
import numpy as np


def compute_aepe(u_gt, v_gt, u_est, v_est, mask=None):
    u_gt = u_gt.astype(np.float32)
    v_gt = v_gt.astype(np.float32)
    u_est = u_est.astype(np.float32)
    v_est = v_est.astype(np.float32)

    epe = np.sqrt((u_est - u_gt) ** 2 + (v_est - v_gt) ** 2)
    if mask is not None:
        m = mask > 0
        if not np.any(m):
            return float('nan')
        return float(np.mean(epe[m]))
    return float(np.mean(epe))
