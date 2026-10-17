
import os

import cv2
import numpy as np

from configs import default_params as config
from physics import advect_density_compressible, generate_hybrid_viscous_field
from segmentation import SAM2VideoSegmenter, segment_cell
from video_io import clean_video_name, get_random_frame, resize_frame


def run_simulation(video_path: str, out_dir: str, frames: int,
                    sam: SAM2VideoSegmenter | None = None, seed: int | None = None) -> str:
    
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)

    seed_frame_full = get_random_frame(video_path, seed=seed)
    mask_full = segment_cell(seed_frame_full, out_dir, sam=sam)

    seed_frame = resize_frame(seed_frame_full, config.TARGET_SIZE)
    if config.TARGET_SIZE is not None and mask_full.shape != tuple(config.TARGET_SIZE):
        target_h, target_w = config.TARGET_SIZE
        mask = cv2.resize(mask_full.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask = np.clip(mask, 0.0, 1.0)
    else:
        mask = mask_full

    np.save(os.path.join(out_dir, "mask.npy"), (mask > 0.5).astype(np.uint8))

    h, w = seed_frame.shape[:2]
    image = cv2.cvtColor(seed_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    density = 1.0 - image  # synthetic "density" -- chromatin is dark, so bright = low density

    video_path_out = os.path.join(out_dir, "sim.mp4")
    writer = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*config.MP4_FOURCC), config.MP4_FPS, (w, h))

 
    U_rand = np.random.randn(h, w, frames)
    V_rand = np.random.randn(h, w, frames)
    U_rand -= U_rand.mean(axis=-1, keepdims=True)
    V_rand -= V_rand.mean(axis=-1, keepdims=True)

    for i in range(frames):
        U, V = generate_hybrid_viscous_field(
            density, mask, U_rand[:, :, i], V_rand[:, :, i],
            grad_weight=config.GRAD_WEIGHT, rand_weight=config.RAND_WEIGHT,
            intensity=config.INTENSITY, viscosity=config.VISCOSITY,
            boundary_damping=getattr(config, "BOUNDARY_DAMPING", 0.4),
        )
        density_prev = density.copy()
        density = advect_density_compressible(density, U, V)

        cv2.imwrite(os.path.join(out_dir, f"img_t_{i:03d}.tiff"), (1.0 - density_prev).astype(np.float32))
        cv2.imwrite(os.path.join(out_dir, f"img_tp1_{i:03d}.tiff"), (1.0 - density).astype(np.float32))
        np.save(os.path.join(out_dir, f"flow_{i:03d}.npy"),
                np.stack([U.astype(np.float32), V.astype(np.float32)], axis=0))

        frame_u8 = ((1.0 - density) * 255.0).clip(0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR))

    writer.release()
    return out_dir


def main() -> None:
    sam = None
    if config.SEGMENTATION_METHOD == "sam":
        try:
            sam = SAM2VideoSegmenter(config.SAM_MODEL_CONFIG, config.SAM_CHECKPOINT, config.SAM_DEVICE)
        except Exception as e:
            print(f"[!] SAM 2 not available ({e}); falling back to Hough.")

    video_path = config.INPUT_VIDEO
    out_dir = os.path.join(config.OUTPUT_DIR, clean_video_name(video_path), "sim_00")
    print(f"[+] Generating {config.FRAMES_PER_SIM} frames from {os.path.basename(video_path)} -> {out_dir}")
    run_simulation(video_path, out_dir, config.FRAMES_PER_SIM, sam=sam, seed=getattr(config, "SEED", None))
    print(f"[+] Done -> {out_dir}")


if __name__ == "__main__":
    main()
