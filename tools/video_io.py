"""Reading a seed frame for the generator (from a video, an .nd2 stack, or
a plain image) and small frame utilities."""

import os
import random

import cv2
import numpy as np


def get_random_frame(filepath: str, seed: int | None = None) -> np.ndarray:
    """Returns a single BGR uint8 frame from `filepath`.

    - `.mp4` (or other OpenCV-readable video): a random frame.
    - `.nd2`: a random slice of its first (time) axis.
    - a plain image (`.png`/`.jpg`/`.tif`/...): that image, as-is.

    `seed` makes the random pick reproducible; `None` uses whatever global
    random state is current.
    """
    rng = random.Random(seed)
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".mp4", ".avi", ".mov"):
        cap = cv2.VideoCapture(filepath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no readable frames: {filepath}")
        frame_idx = rng.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read frame {frame_idx} from {filepath}")
        return frame

    if ext == ".nd2":
        import nd2

        with nd2.ND2File(filepath) as f:
            volume = f.asarray()
            if volume.ndim < 2:
                raise ValueError(f".nd2 file has no 2D image data: {filepath}")
            if volume.ndim == 2:
                frame = volume
            elif volume.ndim == 3:
                frame = volume[rng.randint(0, volume.shape[0] - 1)]
            else:
                frame = volume[rng.randint(0, volume.shape[0] - 1), 0]
            frame_norm = cv2.normalize(frame.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
            return cv2.cvtColor(frame_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not read image: {filepath}")
        return frame

    raise ValueError(f"Unsupported file type: {ext}")


def resize_frame(frame_bgr: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    """Resizes `frame_bgr` to `target_size` = (height, width); `None` leaves
    it untouched. Done once, right after reading the seed frame and before
    segmentation/simulation, so the whole pipeline -- and the ground-truth
    flow it produces -- runs natively at `target_size` with no later
    rescaling."""
    if target_size is None:
        return frame_bgr
    target_h, target_w = target_size
    h, w = frame_bgr.shape[:2]
    if (h, w) == (target_h, target_w):
        return frame_bgr
    interp = cv2.INTER_AREA if (target_h * target_w) < (h * w) else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (target_w, target_h), interpolation=interp)


def clean_video_name(path: str) -> str:
    """A filesystem-safe name for `path`, usable as an output folder name."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace(" ", "_").replace("(", "").replace(")", "")
