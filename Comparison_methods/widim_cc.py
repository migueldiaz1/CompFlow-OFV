
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from openpiv_cpu.cpu_process import piv_cpu


DEFAULT_PARAMS = {
    "min_search_size": 16,
    "search_size_iters": 1,
    "overlap_ratio": 0.5,
}


def prepare(params, device="cuda"):
    return {}


def run(state, params, img1_u8, img2_u8, device="cuda"):
    min_search_size = params.get("min_search_size", DEFAULT_PARAMS["min_search_size"])
    kwargs = {k: v for k, v in {**DEFAULT_PARAMS, **params}.items()
              if k not in ("min_search_size",)}
    if "search_size_iters" in kwargs and isinstance(kwargs["search_size_iters"], list):
        kwargs["search_size_iters"] = tuple(kwargs["search_size_iters"])

    piv = piv_cpu(img1_u8.shape, min_search_size, **kwargs)
    u_sparse, v_sparse = piv(img1_u8.astype(np.float64), img2_u8.astype(np.float64))
    x, y = piv.coords

    h, w = img1_u8.shape
    xd, yd = np.meshgrid(np.arange(w), np.arange(h))
    y_coords, x_coords = y[:, 0], x[0, :]
    yd_clipped = np.clip(yd, y_coords[0], y_coords[-1])
    xd_clipped = np.clip(xd, x_coords[0], x_coords[-1])
    interp_u = RegularGridInterpolator((y_coords, x_coords), u_sparse, method="cubic",
                                        bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((y_coords, x_coords), v_sparse, method="cubic",
                                        bounds_error=False, fill_value=None)
    destination_points = np.stack([yd_clipped.ravel(), xd_clipped.ravel()], axis=-1)
    u = interp_u(destination_points).reshape(h, w)
    v = interp_v(destination_points).reshape(h, w)

    return u.astype(np.float32), v.astype(np.float32), {}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h = w = 256
    base = (rng.random((h, w)) * 255).astype(np.uint8)
    shifted = np.roll(base, shift=(1, 2), axis=(0, 1))
    state = prepare(DEFAULT_PARAMS, device="cpu")
    u, v, _ = run(state, DEFAULT_PARAMS, base, shifted, device="cpu")
    print(f"u mean={u.mean():.4f}  v mean={v.mean():.4f}  (true shift: dy=1, dx=2)")
