
import numpy as np


def compute_aae_weighted(u_gt, v_gt, u_est, v_est, mask=None):
    u_gt = u_gt.astype(np.float32)
    v_gt = v_gt.astype(np.float32)
    u_est = u_est.astype(np.float32)
    v_est = v_est.astype(np.float32)

    if mask is not None:
        m = mask > 0
        if not np.any(m):
            return float('nan')
        u_gt, v_gt, u_est, v_est = u_gt[m], v_gt[m], u_est[m], v_est[m]

    if u_gt.size == 0:
        return float('nan')

    mag_gt = np.sqrt(u_gt ** 2 + v_gt ** 2)
    weight = mag_gt
    if not np.any(weight > 0):
        return float('nan')

    mag_gt_eps = mag_gt + 1e-6
    mag_est_eps = np.sqrt(u_est ** 2 + v_est ** 2) + 1e-6
    dot = np.clip((u_gt / mag_gt_eps) * (u_est / mag_est_eps) + (v_gt / mag_gt_eps) * (v_est / mag_est_eps), -1.0, 1.0)
    angle = np.arccos(dot) * 180.0 / np.pi

    return float(np.sum(angle * weight) / np.sum(weight))
