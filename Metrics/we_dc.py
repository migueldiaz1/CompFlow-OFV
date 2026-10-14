
import cv2
import numpy as np


def _warp_compressible_density(density, U, V, dt=1.0):
    height, width = density.shape[:2]

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = np.float32(X - U)
    map_y = np.float32(Y - V)
    advected_density = cv2.remap(
        density, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    dU_dx = cv2.Sobel(U, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dV_dy = cv2.Sobel(V, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    divergence = dU_dx + dV_dy

    final_density = advected_density - (advected_density * divergence * dt)
    final_density = np.clip(final_density, 0, 1)
    return final_density


def predict_frame2(img1_gray, u_est, v_est, dark_is_dense=True):
    img1_f = img1_gray.astype(np.float32)
    u = u_est.astype(np.float32)
    v = v_est.astype(np.float32)

    density_t = (1.0 - img1_f / 255.0) if dark_is_dense else (img1_f / 255.0)
    predicted_density = _warp_compressible_density(density_t, u, v, dt=1.0)
    return ((1.0 - predicted_density) if dark_is_dense else predicted_density) * 255.0


def compute_warping_error_dc(img1_gray, img2_gray, u_est, v_est, mask=None, dark_is_dense=True):
    img2_f = img2_gray.astype(np.float32)
    img2_pred = predict_frame2(img1_gray, u_est, v_est, dark_is_dense=dark_is_dense)

    diff = np.abs(img2_f - img2_pred.astype(np.float32))
    if mask is not None:
        mask_bool = mask > 0
        if np.any(mask_bool):
            return float(np.mean(diff[mask_bool]))
    return float(np.mean(diff))
