One example simulation from generate_videos_walkthrough.ipynb: the real
video in Datasets/RealCells_sample.zip's complete_sample_video/, seed=42,
SAM 2 segmentation, config.FRAMES_PER_SIM=200 (the real dataset's default).

sim.mp4                 the full 200-frame preview video
img_t_NNN.tiff           step NNN's starting frame   (float32, [0, 1])
img_tp1_NNN.tiff         step NNN's frame after it    (float32, [0, 1])
flow_NNN.npy             step NNN's ground truth      (float32, shape (2, H, W) -> [U, V])

Only 11 of the 200 (img_t, img_tp1, flow) triplets are included here
(steps 0, 20, 40, ..., 180, 199 -- early through late), to keep this
folder small; the full run is ~200MB. Regenerate all 200 yourself with
generate_videos_walkthrough.ipynb or generate_synthetic_video.py (same
video, seed, and config reproduce this exact simulation).
