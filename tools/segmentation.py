import os
import shutil

import cv2
import numpy as np


def apply_clahe(frame_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame_gray)


def get_hough_seed(frame_gray: np.ndarray) -> tuple[int, int, int]:

    blurred = cv2.GaussianBlur(frame_gray, (11, 11), 0)
    h, w = frame_gray.shape
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(w * 0.3),
        param1=60, param2=30, minRadius=int(w * 0.15), maxRadius=int(w * 0.45),
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        dists = np.sqrt((circles[:, 0] - w / 2) ** 2 + (circles[:, 1] - h / 2) ** 2)
        cx, cy, r = circles[np.argmin(dists)]
        return int(cx), int(cy), int(r)
    return w // 2, h // 2, min(w, h) // 4


def filter_largest_blob(mask_uint8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 2:
        return mask_uint8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    clean = np.zeros_like(mask_uint8)
    clean[labels == largest_label] = 255
    return clean


def _soften_mask(mask_uint8: np.ndarray) -> np.ndarray:

    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    return cv2.GaussianBlur(eroded, (15, 15), 0).astype(np.float32) / 255.0


class SAM2VideoSegmenter:
   

    def __init__(self, model_config: str, checkpoint: str, device: str = "cuda"):
        from sam2.build_sam import build_sam2_video_predictor

        print(f"[SAM 2] Loading predictor on {device.upper()}...")
        self.predictor = build_sam2_video_predictor(model_config, checkpoint, device=device)
        self.device = device
        print("[SAM 2] Ready.")

    def segment(self, frame_bgr: np.ndarray, work_dir: str) -> np.ndarray:
    
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        temp_in = os.path.join(work_dir, "_sam2_frame_cache")
        if os.path.isdir(temp_in):
            shutil.rmtree(temp_in, ignore_errors=True)
        os.makedirs(temp_in, exist_ok=True)
        cv2.imwrite(os.path.join(temp_in, "00000.jpg"), frame_bgr)

        try:
            state = self.predictor.init_state(video_path=temp_in)

            cx, cy, r = get_hough_seed(apply_clahe(frame_gray))
            box = np.array([[cx - int(r * 1.2), cy - int(r * 1.2),
                              cx + int(r * 1.2), cy + int(r * 1.2)]], dtype=np.float32)

            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=np.array([[cx, cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
                box=box,
            )

            raw_mask = None
            for _, _, mask_logits in self.predictor.propagate_in_video(state):
                raw_mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                break
            if raw_mask is None:
                raise RuntimeError("SAM 2 returned no mask.")

            clean = filter_largest_blob((raw_mask * 255).astype(np.uint8))
            return _soften_mask(clean)
        finally:
            shutil.rmtree(temp_in, ignore_errors=True)


def segment_cell(frame_bgr: np.ndarray, work_dir: str, sam: SAM2VideoSegmenter | None = None) -> np.ndarray:

    if sam is not None:
        return sam.segment(frame_bgr, work_dir)

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy, r = get_hough_seed(apply_clahe(frame_gray))
    mask = np.zeros(frame_gray.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return _soften_mask(mask)
