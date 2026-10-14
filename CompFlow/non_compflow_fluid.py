import torch
import torch.nn.functional as F

from common import build_density, bilinear_warp
from pi_flow import physicsflowloss


def _df_viscous_loss(visc_criterion, u, v):
    zeros = torch.zeros_like(u)
    _, components = visc_criterion(zeros, zeros, u, v, return_components=True)
    return components['viscosity']


def _optimize(rho1, rho2, visc_criterion, params, device):
    _, _, h, w = rho1.shape
    noise_scale = 1e-2
    u = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
    v = (torch.randn((1, 1, h, w), dtype=torch.float32, device=device) * noise_scale).requires_grad_(True)
    optimizer = torch.optim.Adam([u, v], lr=params['learning_rate'])

    for _ in range(params['iterations']):
        optimizer.zero_grad()
        loss = F.mse_loss(bilinear_warp(rho1, u, v), rho2) + _df_viscous_loss(visc_criterion, u, v)
        loss.backward()
        optimizer.step()

    return u, v


def prepare(params, device='cuda'):
    visc_criterion = physicsflowloss(continuity=0.0, viscosity=params['viscosity'],
                                     L2regularizer=0.0).to(device)
    return {'visc': visc_criterion}


def run(state, params, img1_u8, img2_u8, device='cuda'):
    rho1 = build_density(img1_u8, params, device)
    rho2 = build_density(img2_u8, params, device)
    u, v = _optimize(rho1, rho2, state['visc'], params, device)
    return u.detach().cpu().numpy().squeeze(), v.detach().cpu().numpy().squeeze(), {}
