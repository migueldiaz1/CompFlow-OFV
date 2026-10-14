
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common import build_density


init_random = True


def warp_compressible_density(density, U, V, dt=1.0):
    B, C, H, W = density.shape
    device = density.device

    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    map_x = xx - U.squeeze(1)
    map_y = yy - V.squeeze(1)

    map_x_norm = 2.0 * map_x / (W - 1) - 1.0
    map_y_norm = 2.0 * map_y / (H - 1) - 1.0

    grid = torch.stack((map_x_norm, map_y_norm), dim=-1)

    advected_density = F.grid_sample(
        density, grid, mode='bilinear',
        padding_mode='border', align_corners=True,
    )

    kernel_x = torch.tensor([[[[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]]]], device=device) / 8.0
    kernel_y = torch.tensor([[[[-1., -2., -1.],
                               [ 0.,  0.,  0.],
                               [ 1.,  2.,  1.]]]], device=device) / 8.0

    U_pad = F.pad(U, (1, 1, 1, 1), mode='replicate')
    V_pad = F.pad(V, (1, 1, 1, 1), mode='replicate')

    dU_dx = F.conv2d(U_pad, kernel_x)
    dV_dy = F.conv2d(V_pad, kernel_y)
    divergence = dU_dx + dV_dy

    final_density = advected_density - (advected_density * divergence * dt)
    final_density = torch.clamp(final_density, min=0.0, max=1.0)

    return final_density


class WEDCFlowLoss(nn.Module):
    def __init__(self, loss_type="mse", huber_delta=0.1, L2regularizer=0.0):
        super().__init__()
        assert loss_type in ("mse", "huber"), "loss_type must be 'mse' or 'huber'"
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.L2regularizer = L2regularizer

    def forward(self, rho1, rho2, u, v, return_components=False):
        warped_rho1 = warp_compressible_density(rho1, u, v, dt=1.0)

        if self.loss_type == "huber":
            loss_wedc = F.huber_loss(warped_rho1, rho2, delta=self.huber_delta)
        else:
            loss_wedc = F.mse_loss(warped_rho1, rho2)

        L2_Reg = torch.mean(u ** 2 + v ** 2)
        total_loss = loss_wedc + self.L2regularizer * L2_Reg

        if return_components:
            return total_loss, {
                'wedc': loss_wedc,
                'l2':   self.L2regularizer * L2_Reg,
            }
        return total_loss


def compute_wedc_motion(rho1, rho2, criterion, params,
                        u_init=None, v_init=None, device='cuda'):
    _, _, h, w = rho1.shape

    if u_init is not None and v_init is not None:
        u = u_init.detach().clone().requires_grad_(True)
        v = v_init.detach().clone().requires_grad_(True)
        iterations = max(5, params['iterations'] // 4)
    else:
        if init_random:
            noise_scale = 1e-2
            u = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device)
                 * noise_scale).requires_grad_(True)
            v = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device)
                 * noise_scale).requires_grad_(True)
        else:
            u = torch.zeros((1, 1, h, w), dtype=torch.float32,
                            requires_grad=True, device=device)
            v = torch.zeros((1, 1, h, w), dtype=torch.float32,
                            requires_grad=True, device=device)
        iterations = params['iterations']

    optimizer = torch.optim.Adam([u, v], lr=params['learning_rate'])
    use_amp = (device == 'cuda') or (isinstance(device, torch.device) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    u_history, v_history, loss_history = [], [], []

    for i in range(iterations):
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss = criterion(rho1, rho2, u, v)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(rho1, rho2, u, v)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_f32 = criterion(rho1.float(), rho2.float(), u.float(), v.float())
            u_history.append(u.abs().mean().item())
            v_history.append(v.abs().mean().item())
            loss_history.append(loss_f32.item())

    return u, v, u_history, v_history, loss_history


DEFAULT_PARAMS = {
    'blur':           1,
    'dark_is_dense':  True,
    'iterations':     100,
    'learning_rate':  0.005,
    'loss_type':      'mse',
    'huber_delta':    0.1,
    'L2regularizer':  0.0,
}


def prepare(params, device='cuda'):
    criterion = WEDCFlowLoss(
        loss_type=params['loss_type'], huber_delta=params['huber_delta'],
        L2regularizer=params['L2regularizer'],
    ).to(device)
    return {'criterion': criterion}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    rho1 = build_density(img1_u8, params, device)
    rho2 = build_density(img2_u8, params, device)
    u, v, *_ = compute_wedc_motion(
        rho1, rho2, state['criterion'], params,
        u_init=None, v_init=None, device=device)
    u_np = u.detach().cpu().numpy().squeeze()
    v_np = v.detach().cpu().numpy().squeeze()

    extra = {
        'rho1_np': rho1.detach().cpu().numpy().squeeze(),
        'rho2_np': rho2.detach().cpu().numpy().squeeze(),
    }
    return u_np, v_np, extra


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = w = 64
    rng = np.random.default_rng(0)
    base = rng.random((h, w), dtype=np.float32)
    rho1 = torch.tensor(base, device=device).view(1, 1, h, w)
    shift = torch.roll(rho1, shifts=(1, 2), dims=(2, 3))
    rho2 = shift.clone()

    criterion = WEDCFlowLoss(loss_type=DEFAULT_PARAMS['loss_type'],
                             huber_delta=DEFAULT_PARAMS['huber_delta'],
                             L2regularizer=DEFAULT_PARAMS['L2regularizer']).to(device)

    u, v, hist_u, hist_v, hist_loss = compute_wedc_motion(
        rho1, rho2, criterion, DEFAULT_PARAMS, device=device)

    print(f"initial loss={hist_loss[0]:.6f}  final loss={hist_loss[-1]:.6f}")
    print(f"u mean={u.mean().item():.4f}  v mean={v.mean().item():.4f}")
