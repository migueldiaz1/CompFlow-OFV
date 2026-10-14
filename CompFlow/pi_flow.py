
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from common import build_density

PIXEL_SIZE_NM = 64.6
init_random = True


class physicsflowloss(nn.Module):
    def __init__(self, continuity=1.0, viscosity=0.5, L2regularizer=0.05, ksize=3, scale=1):

        super(physicsflowloss, self).__init__()

        self.continuity = continuity
        self.viscosity = viscosity
        self.L2regularizer = L2regularizer

        self.padding = ksize // 2

        kx_x, ky_x = cv2.getDerivKernels(dx=1, dy=0, ksize=ksize, ktype=cv2.CV_32F)
        sobel_2D_x = np.outer(ky_x, kx_x)

        norm_factor_x = np.sum(sobel_2D_x[sobel_2D_x > 0])
        sobel_2D_x = (sobel_2D_x / norm_factor_x) * scale

        kx_y, ky_y = cv2.getDerivKernels(dx=0, dy=1, ksize=ksize, ktype=cv2.CV_32F)
        sobel_2D_y = np.outer(ky_y, kx_y)
        norm_factor_y = np.sum(sobel_2D_y[sobel_2D_y > 0])
        sobel_2D_y = (sobel_2D_y / norm_factor_y) * scale

        sobel_x_tensor = torch.tensor(sobel_2D_x, dtype=torch.float32).view(1, 1, ksize, ksize)
        sobel_y_tensor = torch.tensor(sobel_2D_y, dtype=torch.float32).view(1, 1, ksize, ksize)

        self.register_buffer('sobel_x', sobel_x_tensor)
        self.register_buffer('sobel_y', sobel_y_tensor)

    def forward(self, rho_base, d_rho_dt, u, v, return_components=False):

        flux_x = rho_base * u
        flux_y = rho_base * v

        d_flux_x_dx = F.conv2d(flux_x, self.sobel_x, padding=self.padding)
        d_flux_y_dy = F.conv2d(flux_y, self.sobel_y, padding=self.padding)

        continuity_residual = d_rho_dt + d_flux_x_dx + d_flux_y_dy

        loss_continuity = torch.mean(continuity_residual ** 2)

        u_x = F.conv2d(u, self.sobel_x, padding=self.padding)
        u_y = F.conv2d(u, self.sobel_y, padding=self.padding)
        v_x = F.conv2d(v, self.sobel_x, padding=self.padding)
        v_y = F.conv2d(v, self.sobel_y, padding=self.padding)

        loss_viscosity = torch.mean(2.0 * (u_x**2) + 2.0 * (v_y**2) + (u_y + v_x)**2)

        L2_Reg = torch.mean(u**2 + v**2)

        total_loss = (self.continuity * loss_continuity) + (self.viscosity * loss_viscosity) + (self.L2regularizer * L2_Reg)

        if return_components:
            return total_loss, {
                'continuity': self.continuity * loss_continuity,
                'viscosity':  self.viscosity * loss_viscosity,
                'l2':         self.L2regularizer * L2_Reg,
            }
        return total_loss


def build_temporal_kernel(N, sigma=0.5, device='cuda'):

    if N == 2:
        kernel = np.array([-1.0, 1.0])
    else:

        x = np.linspace(-1, 1, N)
        kernel = x * np.exp(-0.5 * (x / sigma)**2)
        positive_sum = np.sum(kernel[kernel > 0])
        if positive_sum > 0:
            kernel = kernel / positive_sum

    return torch.tensor(kernel, dtype=torch.float32, device=device).view(-1, 1, 1, 1)


def compute_buffer_motion(rho_buffer, criterion, params, kernel_t, u_init=None, v_init=None, device='cuda'):

    batch_rho = torch.cat(list(rho_buffer), dim=0)
    d_rho_dt = torch.sum(batch_rho * kernel_t, dim=0, keepdim=True)

    d_rho_dt_sq_mean = float(torch.mean(d_rho_dt ** 2).item())

    rho_base = rho_buffer[len(rho_buffer) // 2]
    _, _, h, w = rho_base.shape

    if u_init is not None and v_init is not None:
        u = u_init.detach().clone().requires_grad_(True)
        v = v_init.detach().clone().requires_grad_(True)

        iterations = max(5, params['iterations'] // 4)
    else:
        if init_random:

            noise_scale = 1e-2
            u = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
            v = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
            iterations = params['iterations']
        else:

            u = torch.zeros((1, 1, h, w), dtype=torch.float32, requires_grad=True, device=device)
            v = torch.zeros((1, 1, h, w), dtype=torch.float32, requires_grad=True, device=device)
            iterations = params['iterations']

    optimizer = torch.optim.Adam([u, v], lr=params['learning_rate'])

    scaler = torch.amp.GradScaler('cuda')

    u_history = []
    v_history = []
    loss_history = []

    continuity_loss_history = []
    viscosity_loss_history = []

    for i in range(iterations):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = criterion(rho_base, d_rho_dt, u, v)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():

            loss_f32, comps_f32 = criterion(
                rho_base.float(), d_rho_dt.float(), u.float(), v.float(),
                return_components=True,
            )
            u_history.append(u.abs().mean().item())
            v_history.append(v.abs().mean().item())
            loss_history.append(loss_f32.item())
            continuity_loss_history.append(comps_f32['continuity'].item())
            viscosity_loss_history.append(comps_f32['viscosity'].item())

    return (u, v, u_history, v_history,
            loss_history, continuity_loss_history, viscosity_loss_history,
            d_rho_dt_sq_mean)


def prepare(params, device='cuda'):
    criterion = physicsflowloss(
        continuity=params['continuity'], viscosity=params['viscosity'],
        L2regularizer=params['L2regularizer'],
    ).to(device)
    kernel_t = build_temporal_kernel(2, device=device)
    return {'criterion': criterion, 'kernel_t': kernel_t}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    rho1 = build_density(img1_u8, params, device)
    rho2 = build_density(img2_u8, params, device)
    buf = deque([rho1, rho2], maxlen=2)
    u, v, *_ = compute_buffer_motion(
        buf, state['criterion'], params, state['kernel_t'],
        u_init=None, v_init=None, device=device)
    u_np = u.detach().cpu().numpy().squeeze()
    v_np = v.detach().cpu().numpy().squeeze()

    extra = {
        'rho1_np': rho1.detach().cpu().numpy().squeeze(),
        'rho2_np': rho2.detach().cpu().numpy().squeeze(),
    }
    return u_np, v_np, extra
