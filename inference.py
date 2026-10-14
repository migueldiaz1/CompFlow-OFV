import importlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "CompFlow"))
sys.path.insert(0, os.path.join(_HERE, "Comparison_methods"))
sys.path.insert(0, os.path.join(_HERE, "Metrics"))

# Methods are imported lazily (on first use) so that CompFlow and the classical
# baselines run with only torch/numpy/opencv/scipy installed; WIDIM-CC, Typhoon
# and the ptlflow models pull in their own optional dependency only when chosen.
MODULES = {
    "CompFlow-Patch": "compflow_patch",
    "CompFlow": "compflow",
    "CompFlow-Fluid": "compflow_fluid",
    "PI-Flow": "pi_flow",
    "NonCompFlow": "non_compflow",
    "NonCompFlow-Patch": "non_compflow_patch",
    "NonCompFlow-Fluid": "non_compflow_fluid",
    "Horn-Schunck": "horn_schunck",
    "Lucas-Kanade": "lucas_kanade",
    "Farneback": "farneback",
    "WIDIM-CC": "widim_cc",
    "Typhoon": "typhoon",
    "RAFT": "ptlflow_method",
    "PWC-Net": "ptlflow_method",
    "DPFlow": "ptlflow_method",
}

# pip package to suggest when a method's optional dependency is missing
_OPTIONAL_DEPS = {
    "widim_cc": "openpiv-python-cpu",
    "typhoon": "PyWavelets",
    "ptlflow_method": "ptlflow",
}


def _load_module(method):
    name = MODULES[method]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if name in _OPTIONAL_DEPS:
            raise ModuleNotFoundError(
                f"'{method}' needs an optional dependency that is not installed ({e.name}). "
                f"Install it with: pip install {_OPTIONAL_DEPS[name]}"
            ) from e
        raise

# Parameters per (dataset, method) come from configs/dataset_params.json --
# exactly what main.py uses for each of the three datasets in the paper
# (SynthCells, SQG, RealCells), one JSON object per dataset. RealCells only
# lists PI-Flow / CompFlow / CompFlow-Patch plus the classical/deep-learning
# baselines: the four NonCompFlow-*/CompFlow-Fluid ablation variants were
# only calibrated and benchmarked on SynthCells/SQG in the paper (Table 4),
# never run on RealCells.
CONFIG_PATH = os.path.join(_HERE, "configs", "dataset_params.json")
with open(CONFIG_PATH) as _f:
    DATASET_PARAMS = json.load(_f)

_FLAG_NAMES = {"OPTFLOW_FARNEBACK_GAUSSIAN"}


def _resolve_flags(params):
    """Farneback's 'flags' is stored as a cv2 constant NAME in the json
    (readable/portable); resolve it to the actual cv2 int here."""
    if isinstance(params.get("flags"), str) and params["flags"] in _FLAG_NAMES:
        params = dict(params)
        import cv2
        params["flags"] = getattr(cv2, params["flags"])
    return params


def get_params(dataset, method):
    if dataset not in DATASET_PARAMS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASET_PARAMS)}")
    if method not in DATASET_PARAMS[dataset]:
        raise ValueError(f"'{method}' was not calibrated/benchmarked on '{dataset}' in the paper. "
                         f"Available methods for '{dataset}': {list(DATASET_PARAMS[dataset])}")
    return _resolve_flags(DATASET_PARAMS[dataset][method])


def estimate_flow(img1, img2, method="CompFlow-Patch", dataset="RealCells", params=None, device=None):
    """dataset selects which of the paper's three calibrated parameter sets
    to use (SynthCells / SQG / RealCells, see configs/dataset_params.json).
    Pass `params` to override/replace them with your own."""
    if method not in MODULES:
        raise ValueError(f"Unknown method '{method}'. Available: {list(MODULES)}")
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    p = get_params(dataset, method) if dataset is not None else {}
    p = dict(p)
    if params:
        p.update(params)
    if not p:
        raise ValueError("No parameters: pass either `dataset` (one of "
                         f"{list(DATASET_PARAMS)}) or an explicit `params` dict.")

    mod = _load_module(method)
    state = mod.prepare(p, device=device)
    out = mod.run(state, p, img1, img2, device=device)
    u, v = out[0], out[1]
    return u, v


if __name__ == "__main__":
    import argparse
    import numpy as np
    import cv2

    parser = argparse.ArgumentParser(description="CompFlow inference: estimate dense motion between two frames.")
    parser.add_argument("frame1", help="path to the first grayscale frame")
    parser.add_argument("frame2", help="path to the second grayscale frame")
    parser.add_argument("--method", default="CompFlow-Patch", choices=list(MODULES),
                        help="which method to run (default: CompFlow-Patch, the method proposed in the paper)")
    parser.add_argument("--dataset", default="RealCells", choices=list(DATASET_PARAMS),
                        help="which calibrated parameter set to use (default: RealCells, live-cell microscopy)")
    parser.add_argument("--out", default=None, help="optional .npz output path to save (u, v)")
    args = parser.parse_args()

    img1 = cv2.imread(args.frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.frame2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("could not read one of the input frames")

    u, v = estimate_flow(img1, img2, method=args.method, dataset=args.dataset)
    print(f"method={args.method}  dataset={args.dataset}  |w| mean={np.hypot(u, v).mean():.4f} px/frame")

    if args.out:
        np.savez_compressed(args.out, u=u, v=v)
        print(f"-> {args.out}")
