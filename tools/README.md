# Synthetic video generation

Builds a synthetic chromatin-motion video with exact, dense ground-truth
flow from a real cell frame: segment the cell, simulate a compressible,
viscous velocity field confined to it, and advect the frame's density by
that field. This is the process that generated `Datasets/SynthCells_sample.zip`.

## Structure

```
physics.py                       the velocity field + density advection (the actual simulation)
segmentation.py                   cell segmentation: SAM 2 (seeded by Hough) or Hough alone
video_io.py                       reading a seed frame (video/.nd2/image), resizing, naming
generate_synthetic_video.py       orchestrates the above for one video -> one simulation

configs/
    default_params.py            all parameters (segmentation method, physics, output) -- edit
                                  this to change behavior, or point the scripts at a different
                                  module with the same names

generate_videos_walkthrough.ipynb   step-by-step notebook, with a rendered example
expected_output/                    one example simulation's output (see Example below)
```

## A note on the physics

`physics.py` is not the version in the original project's
`generate_fluid_dataset_v3.py` -- it's `physics_v3_original.py`, a version
from ~2.5 hours before the real dataset was built, recovered after the
in-repo version turned out to have been edited a month later in a way that
silently stopped reproducing it. The difference is one line in the
boundary-condition kernel (`(1, 1)` vs `(3, 3)`) -- see `physics.py`'s
module docstring for the full story and the numbers that confirm which one
actually matches real samples. If you only take one thing from that
docstring: this generator's default behavior is a (likely accidental)
no-op boundary condition, kept deliberately because it's what the dataset
was actually built with.

## Requirements

Everything here needs only what the rest of this repo already requires
(`numpy`, `opencv-python`, `scipy`, `torch`). Segmentation defaults to
SAM 2, which additionally needs:

- SAM 2: `pip install "git+https://github.com/facebookresearch/sam2.git"`,
  plus a checkpoint (e.g. `sam2.1_hiera_small.pt`, from the
  [official SAM 2 releases](https://github.com/facebookresearch/sam2#model-description))
  -- set `SAM_CHECKPOINT` in `configs/default_params.py` to its path.

Set `SEGMENTATION_METHOD = "hough"` in `configs/default_params.py` to avoid
the SAM 2 dependency entirely (a plain Hough circle as the mask, no
checkpoint needed) -- `generate_synthetic_video.py` also does this
automatically if SAM 2 / torch aren't installed or no checkpoint is
configured.

`.nd2` input additionally needs the `nd2` package.

## Usage

```python
from generate_synthetic_video import run_simulation
from segmentation import SAM2VideoSegmenter

sam = SAM2VideoSegmenter("configs/sam2.1/sam2.1_hiera_s.yaml", "/path/to/sam2.1_hiera_small.pt")
run_simulation("real_cell_video.mp4", "output/sim_00", frames=200, sam=sam, seed=42)
```

Writes `img_t_NNN.tiff` / `img_tp1_NNN.tiff` (float32, `[0, 1]`) and
`flow_NNN.npy` (float32, shape `(2, H, W)` -- `[U, V]`) for each of the
`frames` steps, plus `sim.mp4` for a quick look -- the same triplet format
`Datasets/load_sample.py` reads out of `SynthCells_sample.zip`.

Or edit `configs/default_params.py`'s `INPUT_VIDEO` / `OUTPUT_DIR` and run
`python3 generate_synthetic_video.py` for one simulation from the command
line.

## Example

`generate_videos_walkthrough.ipynb` walks through every step -- picking a
random real frame (fixed seed) from the real video already bundled in
`Datasets/RealCells_sample.zip`, segmenting it, running the simulation, and
plotting the result alongside its ground-truth flow -- with its outputs
already rendered if you just want to read it.

`expected_output/` has that same simulation's actual output: the full
`sim.mp4` preview plus 11 of its 200 `(img_t, img_tp1, flow)` triplets
(spread across the run, early to late -- the other 189 are left out only to
keep this folder small; see `expected_output/README.txt`). Same video,
seed, and config reproduce it exactly.
