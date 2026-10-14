
import torch
import torch.nn.functional as F

from common import build_density

_KERNEL_X = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]) / 8.0
_KERNEL_Y = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]) / 8.0


def _integer_shift(img, dy, dx):
    if dy == 0 and dx == 0:
        return img
    h, w = img.shape[-2:]
    pad_top, pad_bottom = max(-dy, 0), max(dy, 0)
    pad_left, pad_right = max(-dx, 0), max(dx, 0)
    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    y0, x0 = dy + pad_top, dx + pad_left
    return padded[:, :, y0:y0 + h, x0:x0 + w]


def compute_we_window_motion(rho1, rho2, params, device='cuda'):
    window_size = params.get('window_size', 9)
    assert window_size % 2 == 1, "window_size must be odd"
    r = window_size // 2
    _, _, h, w = rho1.shape

    noise_scale = 1e-2
    u = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
    v = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
    optimizer = torch.optim.Adam([u, v], lr=params['learning_rate'])
    iterations = params['iterations']

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    rho2_shift = {(dy, dx): _integer_shift(rho2, dy, dx) for dy, dx in offsets}

    for _ in range(iterations):
        optimizer.zero_grad()

        loss = 0.0
        for dy, dx in offsets:
            map_x = (xx + dx) - u.squeeze(1)
            map_y = (yy + dy) - v.squeeze(1)
            map_x_norm = 2.0 * map_x / (w - 1) - 1.0
            map_y_norm = 2.0 * map_y / (h - 1) - 1.0
            grid = torch.stack((map_x_norm, map_y_norm), dim=-1)

            final_density = F.grid_sample(rho1, grid, mode="bilinear", padding_mode="border", align_corners=True)
            loss = loss + F.mse_loss(final_density, rho2_shift[(dy, dx)])

        loss = loss / len(offsets)
        loss.backward()
        optimizer.step()

    return u, v


DEFAULT_PARAMS = {
    'blur': 1,
    'dark_is_dense': False,
    'iterations': 100,
    'learning_rate': 0.1,
    'window_size': 9,
}


def prepare(params, device='cuda'):
    return {}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    rho1 = build_density(img1_u8, params, device)
    rho2 = build_density(img2_u8, params, device)
    u, v = compute_we_window_motion(rho1, rho2, params, device=device)
    u_np = u.detach().cpu().numpy().squeeze()
    v_np = v.detach().cpu().numpy().squeeze()
    extra = {'rho1_np': rho1.detach().cpu().numpy().squeeze(), 'rho2_np': rho2.detach().cpu().numpy().squeeze()}
    return u_np, v_np, extra


if __name__ == "__main__":
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = w = 64
    rng = np.random.default_rng(0)
    base = rng.random((h, w), dtype=np.float32)
    shifted = np.roll(base, shift=(1, 2), axis=(0, 1))
    base_u8 = (base * 255).astype(np.uint8)
    shifted_u8 = (shifted * 255).astype(np.uint8)

    state = prepare(DEFAULT_PARAMS, device=device)
    u, v, _ = run(state, DEFAULT_PARAMS, base_u8, shifted_u8, device=device)
    print(f"u mean={u.mean():.4f}  v mean={v.mean():.4f}  (true shift: dy=1, dx=2)")
