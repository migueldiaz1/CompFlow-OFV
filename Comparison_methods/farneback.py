
import cv2

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=1, winsize=9,
    iterations=10, poly_n=2, poly_sigma=1.5,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
)


def farneback_flow(prev_gray, curr_gray, params=None):
    if params is None:
        params = FARNEBACK_PARAMS
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **params)
    return flow[..., 0], flow[..., 1]


def prepare(params, device='cuda'):
    return None


def run(state, params, img1_u8, img2_u8, device='cuda'):
    return farneback_flow(img1_u8, img2_u8, params=params)
