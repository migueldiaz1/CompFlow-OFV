
import cv2
import numpy as np

LK_WINDOW_SIZE = 5


SOBEL_K5_GAIN = 128.0


def dense_lucas_kanade_flow(prev_gray, curr_gray, win=None):
    if win is None:
        win = LK_WINDOW_SIZE

    prev = prev_gray.astype(np.float64)
    curr = curr_gray.astype(np.float64)

    Ix = cv2.Sobel(prev, cv2.CV_64F, 1, 0, ksize=5) / SOBEL_K5_GAIN
    Iy = cv2.Sobel(prev, cv2.CV_64F, 0, 1, ksize=5) / SOBEL_K5_GAIN
    It = curr - prev

    kernel = np.ones((win, win), dtype=np.float64)
    S_Ix2 = cv2.filter2D(Ix * Ix, -1, kernel)
    S_Iy2 = cv2.filter2D(Iy * Iy, -1, kernel)
    S_IxIy = cv2.filter2D(Ix * Iy, -1, kernel)
    S_IxIt = cv2.filter2D(Ix * It, -1, kernel)
    S_IyIt = cv2.filter2D(Iy * It, -1, kernel)

    det = S_Ix2 * S_Iy2 - S_IxIy ** 2
    valid = np.abs(det) > 1e-6

    u = np.zeros_like(prev)
    v = np.zeros_like(prev)
    u[valid] = (S_Iy2[valid] * (-S_IxIt[valid]) - S_IxIy[valid] * (-S_IyIt[valid])) / det[valid]
    v[valid] = (S_Ix2[valid] * (-S_IyIt[valid]) - S_IxIy[valid] * (-S_IxIt[valid])) / det[valid]

    return u.astype(np.float32), v.astype(np.float32)


def prepare(params, device='cuda'):
    return None


def run(state, params, img1_u8, img2_u8, device='cuda'):
    return dense_lucas_kanade_flow(img1_u8, img2_u8, win=params.get('win'))
