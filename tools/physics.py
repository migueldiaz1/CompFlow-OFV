import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

BOUNDARY_DAMPING = 0.4


def apply_boundary_condition(U, V, mask, damping=BOUNDARY_DAMPING):
    U = U * mask
    V = V * mask

    grad_mask_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_mask_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_mask_x ** 2 + grad_mask_y ** 2)
    normal_x = grad_mask_x / (magnitude + 1e-8)
    normal_y = grad_mask_y / (magnitude + 1e-8)

    dot_product = (U * normal_x) + (V * normal_y)

    edge_weight = magnitude / (magnitude.max() + 1e-8)
    weight = edge_weight * (dot_product < 0) * damping

    U -= dot_product * normal_x * weight
    V -= dot_product * normal_y * weight

    return U, V


def generate_hybrid_viscous_field(density, mask, U_rand, V_rand,
                                   grad_weight=0.0, rand_weight=1.0,
                                   intensity=0.35, viscosity=4.0,
                                   boundary_damping=BOUNDARY_DAMPING):
    grad_x = cv2.Sobel(density, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(density, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    U_grad = -grad_x
    V_grad = -grad_y
    U_grad = U_grad / (np.max(np.abs(U_grad)) + 1e-8)
    V_grad = V_grad / (np.max(np.abs(V_grad)) + 1e-8)

    U_combined = (grad_weight * U_grad) + (rand_weight * U_rand)
    V_combined = (grad_weight * V_grad) + (rand_weight * V_rand)

    if viscosity > 0:
        U_combined = gaussian_filter(U_combined, sigma=viscosity)
        V_combined = gaussian_filter(V_combined, sigma=viscosity)

    U_combined = (U_combined / (np.max(np.abs(U_combined)) + 1e-8)) * intensity
    V_combined = (V_combined / (np.max(np.abs(V_combined)) + 1e-8)) * intensity

    return apply_boundary_condition(U_combined, V_combined, mask, damping=boundary_damping)


def advect_density_compressible(density, U, V, dt=1.0):
    h, w = density.shape[:2]

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(X - U)
    map_y = np.float32(Y - V)

    advected = cv2.remap(
        density, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    dU_dx = cv2.Sobel(U, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dV_dy = cv2.Sobel(V, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    divergence = dU_dx + dV_dy

    result = advected - (advected * divergence * dt)
    return np.clip(result, 0, 1)
