
import cv2
import torch
import torch.nn.functional as F


def build_density(img_u8, params, device):
    b = params.get('blur', 1)
    blurred = cv2.GaussianBlur(img_u8, (b, b), 0) if b > 1 else img_u8
    rho = torch.tensor(blurred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
    if params.get('dark_is_dense', False):
        rho = 1.0 - rho
    return rho


def bilinear_warp(rho, u, v):
    B, C, H, W = rho.shape
    device = rho.device
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)
    map_x = xx - u.squeeze(1)
    map_y = yy - v.squeeze(1)
    map_x_norm = 2.0 * map_x / (W - 1) - 1.0
    map_y_norm = 2.0 * map_y / (H - 1) - 1.0
    grid = torch.stack((map_x_norm, map_y_norm), dim=-1)
    return F.grid_sample(rho, grid, mode='bilinear', padding_mode='border', align_corners=True)
