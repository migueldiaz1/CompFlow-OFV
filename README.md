# CompFlow

Training-free, physics-informed optical flow for intracellular density dynamics in live-cell microscopy (chromatin motion in leukemic cells).

This repository contains the **optical flow / motion estimation code**: the seven density-flow variants proposed and ablated in the paper, the baseline comparison methods benchmarked against them, the evaluation metrics (AEPE, AAE weighted, WE<sub>DC</sub>), and (`tools/`) the synthetic training-data generator. The video preprocessing pipeline (SAM 2 segmentation, rigid registration, temporal CLAHE) is a separate repository: [StableCells](https://github.com/migueldiaz1/StableCells).

This copy additionally bundles `compflow_demo.ipynb`, a runnable walkthrough notebook -- see [Demo notebook](#demo-notebook) below.

## Quickstart

1. Clone the repo and install the core dependencies (`torch`, `numpy`, `opencv-python`, `scipy` -- enough for CompFlow and the classical baselines; see [Requirements](#requirements) for the optional ones):
   ```bash
   git clone <this-repo-url>
   cd <this-repo>
   pip install -r requirements.txt
   ```
   On Windows, use `python` instead of `python3` in the commands below.
2. Run the proposed method on the included sample data (no need for your own images):
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'Datasets')
   from load_sample import load_pair
   from inference import estimate_flow

   img1, img2, gt = load_pair('Datasets/RealCells_sample.zip', 0)
   u, v = estimate_flow(img1, img2, method='CompFlow-Patch', dataset='RealCells')
   print('mean flow magnitude:', (u**2 + v**2).mean() ** 0.5)
   "
   ```
3. To use it on your own two frames instead:
   ```bash
   python3 inference.py your_frame1.png your_frame2.png --method CompFlow-Patch --dataset RealCells --out flow.npz
   ```
   `flow.npz` will contain the dense `u`/`v` displacement fields. See [Usage](#usage) below for the full Python API, [Trying it on sample data](#trying-it-on-sample-data) for the other datasets, and [Choosing parameters: dataset](#choosing-parameters-dataset) to pick the right calibration for your kind of data.

## Structure

```
CompFlow/                our method and its ablation variants (Table 4 of the paper)
    common.py             shared density-construction / bilinear-warp helpers
    pi_flow.py             PI-Flow            (continuity + viscous-dissipation loss)
    non_compflow.py        NonCompFlow        (pure advection, no divergence compensation)
    non_compflow_patch.py  NonCompFlow-Patch  (NonCompFlow + patch/window aggregation)
    non_compflow_fluid.py  NonCompFlow-Fluid  (NonCompFlow + viscous regularizer)
    compflow.py             CompFlow           (Divergence-Aware Warping loss, Eq. 5)
    compflow_patch.py       CompFlow-Patch     (CompFlow + patch/window aggregation, Eq. 6 -- the method used in the paper's main results)
    compflow_fluid.py       CompFlow-Fluid     (CompFlow + viscous regularizer)

Comparison_methods/      external baselines benchmarked in the paper
    horn_schunck.py, lucas_kanade.py, farneback.py   classical differential optical flow
    widim_cc.py                                      PIV cross-correlation (openpiv-python-cpu)
    typhoon.py                                        wavelet-based optical flow (Dérian & Almar 2017)
    ptlflow_method.py                                 RAFT / PWC-Net / DPFlow (via ptlflow, pretrained "things" weights)

Metrics/
    aepe.py            Average Endpoint Error (needs ground truth)
    aae_weighted.py     magnitude-weighted Average Angular Error (needs ground truth)
    we_dc.py            Density-Compensated Warping Error (no ground truth needed, Eq. 13-14)

configs/
    dataset_params.json  calibrated parameters for every (dataset, method) pair, exactly as used by main.py in the paper

Datasets/                 small sample zips to try inference on (see below) -- NOT the full research datasets
    SynthCells_sample.zip  300 pairs (16 curated videos), dense ground truth, + 1 complete 201-frame video
    SQG_sample.zip          30 pairs, with dense ground truth
    RealCells_sample.zip    300 pairs (17 curated videos), no ground truth, + 1 complete video
    load_sample.py          helper to read pairs out of the zips

tools/                    synthetic training-data generator (the process that built SynthCells_sample.zip)
    physics.py             the velocity-field simulation + density advection
    segmentation.py        cell segmentation: SAM 2 (seeded by Hough) or Hough alone
    video_io.py             reading a seed frame, resizing, naming
    generate_synthetic_video.py  orchestrates the above into one simulation
    generate_videos_walkthrough.ipynb  step-by-step notebook, with a rendered example
    configs/default_params.py    its own parameters (segmentation method, physics, output)

inference.py             dataset-aware registry + simple two-frame inference entry point
requirements.txt         core dependencies (optional baseline dependencies listed as comments)

compflow_demo.ipynb       runnable walkthrough notebook (see Demo notebook below)
utils.py                   plotting / zoom-picking helpers used by the notebook
setup.sh, setup.bat        creates the virtual environment + Jupyter kernel the notebook runs in
```

## Requirements

- `torch`, `numpy`, `opencv-python`, `scipy` (`pip install -r requirements.txt`)
- `openpiv-python-cpu` (only for `WIDIM-CC`)
- `PyWavelets` (only for `Typhoon`; imported as `pywt`)
- `ptlflow` (only for `RAFT`/`PWC-Net`/`DPFlow`)
- SAM 2 (only for `tools/`'s synthetic-data generator, and only with its default `SEGMENTATION_METHOD = "sam"`; see `tools/README.md`)

Each comparison method only needs its own dependency, and it is imported only when that method is selected, so `CompFlow/` and the classical baselines work without the optional packages installed.

## Usage

```bash
python3 inference.py frame1.png frame2.png --method CompFlow-Patch --dataset RealCells
```

```python
import cv2
from inference import estimate_flow

img1 = cv2.imread("frame1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("frame2.png", cv2.IMREAD_GRAYSCALE)

u, v = estimate_flow(img1, img2, method="CompFlow-Patch", dataset="RealCells")
```

`method` can be any of: `CompFlow-Patch` (recommended, used in the paper's main results), `CompFlow`, `CompFlow-Fluid`, `PI-Flow`, `NonCompFlow`, `NonCompFlow-Patch`, `NonCompFlow-Fluid`, `Horn-Schunck`, `Lucas-Kanade`, `Farneback`, `WIDIM-CC`, `Typhoon`, `RAFT`, `PWC-Net`, `DPFlow`.

### Choosing parameters: `dataset`

`dataset` selects which of the paper's three **calibrated** parameter sets to use -- exactly what `main.py` passes to each method for that dataset, taken straight from `configs/dataset_params.json`:

- `"SynthCells"` -- synthetic chromatin with dense ground truth
- `"SQG"` -- the external surface quasi-geostrophic turbulence benchmark
- `"RealCells"` -- real patient-derived leukemic cell microscopy (default; only `PI-Flow`, `CompFlow`, `CompFlow-Patch`, and the classical/deep-learning baselines were calibrated/benchmarked here in the paper -- the four ablation-only variants were only ever run on SynthCells/SQG, see Table 4)

Pass your own `params={...}` instead (or on top, to override individual keys) if you want something other than the paper's calibration:

```python
u, v = estimate_flow(img1, img2, method="CompFlow-Patch", dataset=None,
                     params={"blur": 1, "dark_is_dense": True, "iterations": 100,
                             "learning_rate": 0.05, "window_size": 7})
```

### Evaluating a result

```python
import sys; sys.path.insert(0, "Metrics")
from we_dc import compute_warping_error_dc
from aepe import compute_aepe
from aae_weighted import compute_aae_weighted

we_dc = compute_warping_error_dc(img1, img2, u, v, dark_is_dense=True)   # no GT needed
aepe = compute_aepe(u_gt, v_gt, u, v)                                    # needs GT
aae_w = compute_aae_weighted(u_gt, v_gt, u, v)                           # needs GT
```

## Trying it on sample data

`Datasets/` ships small sample zips so you can run the code without needing your own frames. These are **not** the full research datasets -- SynthCells alone is ~100 GB and SQG ~924 MB, both well past what a normal git repository (no LFS) can hold -- just enough pairs (300 / 30 / 300) to try every method and sanity-check the numbers against the paper.

`SynthCells_sample.zip` and `RealCells_sample.zip` each hold two folders: `other_sampled_pairs/` (the pairs `load_pair`/`n_pairs` read, several per video, spread across 16/17 curated videos -- a handful of videos were excluded after a visual QC pass found a flat/textureless simulation shared by one patient's video in both datasets, a failed SAM 2 segmentation, and an edge-cropped cell) and `complete_sample_video/`, one full, untouched video (all 201 simulation frames for SynthCells, or the original 285-frame `.mp4` for RealCells) plus its own `README.txt`, for anyone who wants to build their own pair at a different frame separation (dt) instead of the single-step ones in `other_sampled_pairs/`.

```python
import sys
sys.path.insert(0, "Datasets")
sys.path.insert(0, "Metrics")
from load_sample import load_pair, n_pairs
from inference import estimate_flow
from aepe import compute_aepe
from aae_weighted import compute_aae_weighted

zip_path = "Datasets/SynthCells_sample.zip"
print(n_pairs(zip_path), "pairs available")

img1, img2, gt = load_pair(zip_path, 0)                      # gt = {'u_gt', 'v_gt', 'mask'} or None (RealCells)
u, v = estimate_flow(img1, img2, method="CompFlow-Patch", dataset="SynthCells")

if gt is not None:
    print("AEPE:", compute_aepe(gt["u_gt"], gt["v_gt"], u, v, mask=gt.get("mask")))
    print("AAE_w:", compute_aae_weighted(gt["u_gt"], gt["v_gt"], u, v, mask=gt.get("mask")))
```

Use `"Datasets/SQG_sample.zip"` with `dataset="SQG"`, or `"Datasets/RealCells_sample.zip"` with `dataset="RealCells"` (no `gt` there -- use `compute_warping_error_dc` instead, which needs no ground truth). `RealCells_sample.zip` contains **raw** frame pairs straight from the video (no SAM 2 segmentation / registration / contrast enhancement applied -- that pipeline isn't in this repository yet), so computing `compute_warping_error_dc` on it (no `mask`) is **not** directly comparable to the paper's RealCells WE<sub>DC</sub> (Table 3), which is averaged only inside the SAM 2 cell mask -- without it the background dilutes the error downward. SynthCells/SQG sample metrics (AEPE, AAE<sub>w</sub>, and masked WE<sub>DC</sub>) match the paper's full-dataset numbers closely (within normal sampling variance for a few hundred pairs vs. thousands).

## Demo notebook

`compflow_demo.ipynb` is a runnable, end-to-end walkthrough: load a SynthCells sample pair, look at it (full cell, then three zoomed-in regions), run `CompFlow-Patch` on it, and plot the resulting dense motion map (estimated vs. ground truth).

```bash
./setup.sh        # Linux/macOS
setup.bat         # Windows
```

This creates `venv/` (reusing your system's `torch`/`numpy`/`opencv`/`scipy` if already installed, installing whatever's missing from `requirements.txt`) and registers it as the standard **"Python 3"** Jupyter kernel. Then open `compflow_demo.ipynb` (`venv/bin/jupyter notebook`/`venv\Scripts\jupyter notebook`, `jupyter lab`, or your IDE) and select that kernel. `venv/` itself is not tracked by git (see `.gitignore`) -- it's a local build artifact, regenerated by the setup script.

## Notes

- `dark_is_dense=True` in a method's params means "dark = dense" (standard brightfield/fluorescence chromatin imaging convention, where condensed chromatin appears darker). Set it to match your imaging convention -- SynthCells/RealCells use `True`, SQG uses `False` (it isn't chromatin at all).
- `CompFlow-Patch`'s `window_size=3` for RealCells is the value grounded in the SynthCells ground-truth sweep (best AEPE/AAE<sub>w</sub>), not a RealCells-specific WE<sub>DC</sub> minimum (see Section 4.3 of the paper).
- Calibration/grid-search scripts, figures, and the K-sensitivity analysis are not included in this repository (research-only, not needed to run the method).
