import io
import zipfile

import cv2
import numpy as np

# Pair files (pair_NNN_img1.png, ...) live either at the zip root (older,
# smaller samples) or under other_sampled_pairs/ (newer, bigger samples
# that also ship a complete_sample_video/ folder alongside -- see each
# zip's own complete_sample_video/README.txt). Auto-detected, so existing
# code calling load_pair/n_pairs doesn't need to change either way.
_PAIR_PREFIXES = ("", "other_sampled_pairs/")


def _pair_prefix(zf):
    names = zf.namelist()
    for prefix in _PAIR_PREFIXES:
        if any(n == f"{prefix}pair_000_img1.png" for n in names):
            return prefix
    raise FileNotFoundError(f"no pair_000_img1.png found at any of {_PAIR_PREFIXES} in this zip")


def load_pair(zip_path, idx):
    """Returns (img1, img2, gt) for pair `idx` in a sample zip. gt is a
    dict with 'u_gt'/'v_gt' (and 'mask' for SynthCells) or None for zips
    without ground truth (RealCells_sample.zip)."""
    with zipfile.ZipFile(zip_path) as zf:
        prefix = _pair_prefix(zf)

        def read_img(name):
            data = zf.read(name)
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

        img1 = read_img(f"{prefix}pair_{idx:03d}_img1.png")
        img2 = read_img(f"{prefix}pair_{idx:03d}_img2.png")

        gt = None
        gt_name = f"{prefix}pair_{idx:03d}_gt.npz"
        if gt_name in zf.namelist():
            with zf.open(gt_name) as f:
                npz = np.load(io.BytesIO(f.read()))
                gt = {k: npz[k] for k in npz.files}

    return img1, img2, gt


def n_pairs(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        prefix = _pair_prefix(zf)
        suffix = "_img1.png"
        return len({n.split(suffix)[0] for n in zf.namelist()
                   if n.startswith(prefix) and n.endswith(suffix) and n.count("/") == prefix.count("/")})
