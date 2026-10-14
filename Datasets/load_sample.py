import io
import zipfile

import cv2
import numpy as np


def load_pair(zip_path, idx):
    """Returns (img1, img2, gt) for pair `idx` in a sample zip. gt is a
    dict with 'u_gt'/'v_gt' (and 'mask' for SynthCells) or None for zips
    without ground truth (RealCells_sample.zip)."""
    with zipfile.ZipFile(zip_path) as zf:
        def read_img(name):
            data = zf.read(name)
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)

        img1 = read_img(f"pair_{idx:03d}_img1.png")
        img2 = read_img(f"pair_{idx:03d}_img2.png")

        gt = None
        gt_name = f"pair_{idx:03d}_gt.npz"
        if gt_name in zf.namelist():
            with zf.open(gt_name) as f:
                npz = np.load(io.BytesIO(f.read()))
                gt = {k: npz[k] for k in npz.files}

    return img1, img2, gt


def n_pairs(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return len({n.split("_img1.png")[0] for n in zf.namelist() if n.endswith("_img1.png")})
