
import cv2
import numpy as np


SOBEL_K5_GAIN = 128.0


HS_ALPHA = 0.3
HS_ITERATIONS = 300


def horn_schunck_flow(prev_gray, curr_gray, alpha=None, num_iter=None):
    if alpha is None:
        alpha = HS_ALPHA
    if num_iter is None:
        num_iter = HS_ITERATIONS

    prev = prev_gray.astype(np.float64) / 255.0
    curr = curr_gray.astype(np.float64) / 255.0

    Ix = cv2.Sobel(prev, cv2.CV_64F, 1, 0, ksize=5) / SOBEL_K5_GAIN
    Iy = cv2.Sobel(prev, cv2.CV_64F, 0, 1, ksize=5) / SOBEL_K5_GAIN
    It = curr - prev

    u = np.zeros_like(prev)
    v = np.zeros_like(prev)

    kernel_avg = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.float64) / 4.0

    alpha2 = alpha ** 2

    for _ in range(num_iter):
        u_avg = cv2.filter2D(u, -1, kernel_avg)
        v_avg = cv2.filter2D(v, -1, kernel_avg)

        num = Ix * u_avg + Iy * v_avg + It
        den = alpha2 + Ix ** 2 + Iy ** 2

        u = u_avg - Ix * (num / den)
        v = v_avg - Iy * (num / den)

    return u.astype(np.float32), v.astype(np.float32)


def prepare(params, device='cuda'):
    return None


def run(state, params, img1_u8, img2_u8, device='cuda'):
    return horn_schunck_flow(img1_u8, img2_u8, alpha=params.get('alpha'), num_iter=params.get('num_iter'))
