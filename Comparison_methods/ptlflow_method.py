
import numpy as np
import torch

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter


from loguru import logger as _ptlflow_logger
_ptlflow_logger.disable("ptlflow")


def _gray_to_rgb(img_u8):
    return np.repeat(img_u8[:, :, None], 3, axis=2)


def prepare(params, device='cuda'):
    model_name = params['model_name']
    checkpoint = params.get('checkpoint', 'things')
    model = ptlflow.get_model(model_name, ckpt_path=checkpoint)
    model = model.to(device).eval()
    return {'model': model}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    model = state['model']
    img1 = _gray_to_rgb(img1_u8)
    img2 = _gray_to_rgb(img2_u8)

    io_adapter = IOAdapter(model.output_stride, img1.shape[:2], cuda=(device != 'cpu'))
    inputs = io_adapter.prepare_inputs(images=[img1, img2])

    with torch.no_grad():
        preds = model(inputs)
    preds = io_adapter.unscale(preds)

    flow = preds['flows'][0, 0].detach().cpu().numpy()
    u, v = flow[0], flow[1]
    return u.astype(np.float32), v.astype(np.float32), {}


RAFT_DEFAULT_PARAMS = {'model_name': 'raft', 'checkpoint': 'things'}
PWCNET_DEFAULT_PARAMS = {'model_name': 'pwcnet', 'checkpoint': 'things'}
DPFLOW_DEFAULT_PARAMS = {'model_name': 'dpflow', 'checkpoint': 'things'}


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = w = 256
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
    shifted = np.roll(base, shift=(1, 2), axis=(0, 1))

    for name, settings in (('RAFT', RAFT_DEFAULT_PARAMS), ('PWC-Net', PWCNET_DEFAULT_PARAMS), ('DPFlow', DPFLOW_DEFAULT_PARAMS)):
        state = prepare(settings, device=device)
        u0, v0, _ = run(state, settings, base, base, device=device)
        u1, v1, _ = run(state, settings, base, shifted, device=device)
        print(f"{name}: same image -> u={u0.mean():.3f} v={v0.mean():.3f} (expected ~0); "
              f"known shift (u=2,v=1) -> u={u1[20:-20, 20:-20].mean():.3f} "
              f"v={v1[20:-20, 20:-20].mean():.3f}")
