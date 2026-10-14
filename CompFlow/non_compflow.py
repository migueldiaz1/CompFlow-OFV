
import torch
import torch.nn.functional as F

from common import build_density, bilinear_warp


def compute_we_motion(rho1, rho2, params, device='cuda'):
    _, _, h, w = rho1.shape
    u = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * 1e-2).requires_grad_(True)
    v = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * 1e-2).requires_grad_(True)
    optimizer = torch.optim.Adam([u, v], lr=params['learning_rate'])
    for _ in range(params['iterations']):
        optimizer.zero_grad()
        loss = F.mse_loss(bilinear_warp(rho1, u, v), rho2)
        loss.backward()
        optimizer.step()
    return u, v


def prepare(params, device='cuda'):
    return {}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    rho1 = build_density(img1_u8, params, device)
    rho2 = build_density(img2_u8, params, device)
    u, v = compute_we_motion(rho1, rho2, params, device=device)
    u_np = u.detach().cpu().numpy().squeeze()
    v_np = v.detach().cpu().numpy().squeeze()
    return u_np, v_np, {}


if __name__ == "__main__":
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = w = 64
    rng = np.random.default_rng(0)
    base = (rng.random((h, w)) * 255).astype(np.uint8)
    shifted = np.roll(base, shift=(1, 2), axis=(0, 1))
    params = {'blur': 1, 'dark_is_dense': False, 'iterations': 100, 'learning_rate': 0.05}
    u, v, _ = run(prepare(params, device=device), params, base, shifted, device=device)
    print(f"u mean={u.mean():.4f}  v mean={v.mean():.4f}  (true shift: dy=1, dx=2)")
