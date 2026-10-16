"""
Physics engine for the synthetic video generator: a compressible, viscous
velocity field confined inside a cell mask, advected onto a density field
(1 - normalized intensity) to produce synthetic chromatin motion with a
known ground-truth flow.

This is the exact physics that generated the paper's real SynthCells
dataset -- not the version currently in the project root
(`generate_fluid_dataset_v3.py`), which was edited a month after that
dataset was built and no longer reproduces it. The difference is one line
in `apply_boundary_condition`:

    kernel = np.ones((1, 1), np.uint8)   # here: a no-op
    kernel = np.ones((3, 3), np.uint8)   # edited version: actually active

Eroding a mask with a 1x1 kernel is the identity operation, so
`mask - erode(mask, kernel)` is zero everywhere -- the boundary-condition
correction (keeping the flow from pushing density out through the cell
membrane) never actually fires. Almost certainly an accidental bug (a 3x3
kernel was probably intended), but it is the version that produced every
real sample: with the 1x1 kernel, |diff|(frame0, frame199) is ~0.009-0.01,
matching real SynthCells samples; the "fixed" 3x3 version activates a
boundary correction that is asymmetric by construction (it only cancels the
outward-pushing component, never corrects inward), which over 200 frames
biases density inward -- a growing dark ring not seen in any real sample,
and measurably different (~0.046).
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def apply_boundary_condition(U, V, mask):
    """Zeroes the flow outside `mask` and cancels the outward-pointing
    component of (U, V) right at the mask's boundary, so density doesn't
    get pushed out through the cell membrane. See the module docstring --
    the 1x1 kernel below makes this a no-op, faithfully reproducing the
    bug that generated the real dataset."""
    U = U * mask
    V = V * mask

    grad_mask_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_mask_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_mask_x ** 2 + grad_mask_y ** 2) + 1e-8
    normal_x = grad_mask_x / magnitude
    normal_y = grad_mask_y / magnitude

    dot_product = (U * normal_x) + (V * normal_y)

    kernel = np.ones((1, 1), np.uint8)  # no-op -- see module docstring
    boundary_zone = (mask - cv2.erode(mask, kernel)) > 0

    pushing_out = (dot_product < 0) & boundary_zone

    U[pushing_out] -= dot_product[pushing_out] * normal_x[pushing_out]
    V[pushing_out] -= dot_product[pushing_out] * normal_y[pushing_out]

    return U, V


def generate_hybrid_viscous_field(density, mask, U_rand, V_rand,
                                   grad_weight=0.0, rand_weight=1.0,
                                   intensity=0.35, viscosity=4.0):
    """Blends a density-gradient-following component with a random
    (turbulent) component, smooths the result with a Gaussian (the
    "viscosity"), rescales to `intensity`, and confines it to `mask`.

    Defaults match the real dataset's config (`grad_weight=0.0`,
    `rand_weight=1.0` -- i.e. purely random/turbulent motion, no
    gradient-following component)."""
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

    return apply_boundary_condition(U_combined, V_combined, mask)


def advect_density_compressible(density, U, V, dt=1.0):
    """Warps `density` by the (U, V) displacement field (semi-Lagrangian,
    backward warp) and applies a compressible correction (mass scales with
    local flow divergence), so density piles up where the flow converges
    and thins out where it diverges, instead of being purely transported."""
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
