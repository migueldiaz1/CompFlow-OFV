"""
utils.py
=========
Plotting / zoom-picking helpers for compflow_demo.ipynb. Pulled out of the
notebook so it stays short and focused on the actual repo API
(estimate_flow / load_pair / n_pairs) rather than matplotlib boilerplate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, ListedColormap, Normalize
from matplotlib.patches import Rectangle


def pick_zoom_regions(mask, size=30, n=3, seed=None, min_sep=40):
    """n interior (y0, x0) top-left corners, size x size, fully inside the
    mask's bounding box and spread at least min_sep apart from each other.
    seed=None (default) -> a different, genuinely random set each run."""
    ys, xs = np.nonzero(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() - size, xs.min(), xs.max() - size
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(2000):
        cy, cx = rng.integers(y0, y1), rng.integers(x0, x1)
        if mask[cy + size // 2, cx + size // 2] == 0:
            continue
        if all(abs(cy - py) + abs(cx - px) > min_sep for py, px in picks):
            picks.append((cy, cx))
        if len(picks) == n:
            break
    return picks


def plot_pair_and_zooms(img1, img2, zoom_regions, zoom_colors, zoom_size):
    """Full cell (Frame 1 / Frame 2, zoom regions marked) on top, each
    zoom's (t0 | t0+1) crop below."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for ax, img, title in ((axes[0, 0], img1, "Frame 1 (full cell)"), (axes[0, 1], img2, "Frame 2 (full cell)")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        for (y0, x0), color in zip(zoom_regions, zoom_colors):
            ax.add_patch(Rectangle((x0, y0), zoom_size, zoom_size, fill=False, edgecolor=color, linewidth=1.8))
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 2].axis("off")

    for i, ((y0, x0), color) in enumerate(zip(zoom_regions, zoom_colors)):
        sl = (slice(y0, y0 + zoom_size), slice(x0, x0 + zoom_size))
        crop = np.concatenate([img1[sl], np.full((zoom_size, 2), 128, dtype=np.uint8), img2[sl]], axis=1)
        ax = axes[1, i]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"zoom {i+1} (t0 | t0+1)", color=color)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)

    fig.tight_layout()
    plt.show()


def density_from_img(img):
    """Chromatin density from intensity: dark = dense (dark_is_dense=True,
    the convention used for SynthCells/RealCells throughout the paper)."""
    return 1.0 - img.astype(np.float32) / 255.0


def color_by_diff(diff, base, amplitude=0.5, smoothness=0.16):
    """diff>0 (gained density) -> green; diff<0 (lost density) -> red.
    tanh instead of a hard threshold: small frame-to-frame noise stays
    visible but faint rather than being clipped to exactly zero."""
    s = amplitude * np.tanh(diff / smoothness)
    mag = np.abs(s)
    R = np.clip(base - s, 0, 1)
    G = np.clip(base + s, 0, 1)
    B = np.clip(base - mag, 0, 1)
    return R, G, B


def overlay_lost_gained(img1, img2, mask):
    d1, d2 = density_from_img(img1), density_from_img(img2)
    diff = d2 - d1
    # background shading = raw intensity, NOT density -- so a dense (dark)
    # region stays dark here too, exactly like it looks in the Frame 1/2
    # panels. Using density for the background instead would make dense
    # (dark) regions render white, which is backwards.
    base = (img1.astype(np.float32) / 255.0 + img2.astype(np.float32) / 255.0) / 2.0
    R, G, B = color_by_diff(diff, base)
    rgb = np.stack([R, G, B], axis=-1)
    rgb *= (mask > 0)[..., None]
    vmax_diff = float(np.percentile(np.abs(diff[mask > 0]), 99))
    return rgb, vmax_diff


def overlay_colormap(vmax_diff, n=256):
    diff = np.linspace(-vmax_diff, vmax_diff, n)
    R, G, B = color_by_diff(diff, base=0.5)
    return ListedColormap(np.stack([R, G, B], axis=-1))


def dense_direction_map(u, v, mask, img_bg, threshold=0.0):
    """Fully dense (every mask pixel above `threshold`, 0 = truly every
    pixel) per-pixel direction color, overlaid on the grayscale frame."""
    magnitude = np.hypot(u, v)
    angle = np.degrees(np.arctan2(v, u)) % 360.0
    hue = angle / 360.0
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    rgb = hsv_to_rgb(hsv)
    visible = (magnitude >= threshold) & (mask > 0)
    return rgb, visible


def colored_quiver(ax, u, v, mask, y0, x0, size, step=2.0, scale=30.0, width=0.0032, n_thickness=3):
    """Arrows colored by direction (hue, same HSV convention as the dense
    map), thickness tiered by magnitude, on a staggered (tresbolillo) grid
    of the given step so arrow spacing looks even instead of blocky."""
    ys1 = np.arange(y0 + step / 2.0, y0 + size, step)
    xs1 = np.arange(x0 + step / 2.0, x0 + size, step)
    Y1, X1 = np.meshgrid(ys1, xs1, indexing="ij")
    Y2, X2 = Y1 + step / 2.0, X1 + step / 2.0
    YY = np.round(np.concatenate([Y1.ravel(), Y2.ravel()])).astype(int)
    XX = np.round(np.concatenate([X1.ravel(), X2.ravel()])).astype(int)
    inside = (YY >= y0) & (YY < y0 + size) & (XX >= x0) & (XX < x0 + size)
    YY, XX = YY[inside], XX[inside]
    uu, vv = u[YY, XX], v[YY, XX]
    visible = mask[YY, XX] > 0
    uu, vv, YY, XX = uu[visible], vv[visible], YY[visible], XX[visible]
    mag = np.hypot(uu, vv)
    angle = np.degrees(np.arctan2(vv, uu)) % 360.0

    cuts = np.quantile(mag, np.linspace(0, 1, n_thickness + 1)) if mag.size else [0, 0]
    for i in range(n_thickness):
        lo, hi = cuts[i], cuts[i + 1]
        sel = (mag >= lo) & (mag <= hi if i == n_thickness - 1 else mag < hi)
        if not np.any(sel):
            continue
        factor = 1.0 + 0.4 * i
        ax.quiver(XX[sel], YY[sel], uu[sel] * scale, vv[sel] * scale, angle[sel],
                  cmap="hsv", norm=Normalize(0, 360), angles="xy", scale_units="xy", scale=1,
                  width=width * factor, headwidth=2.8 + 0.3 * i, headlength=3.4, headaxislength=3.0, pivot="mid")


def plot_motion_map_with_gt(img1, img2, mask, u_est, v_est, u_gt, v_gt,
                            zoom_regions, zoom_colors, zoom_size, suptitle=""):
    """Full breakdown, estimated vs. ground truth side by side: Frame 1,
    Frame 2, density lost/gained, flow direction (CompFlow-Patch | GT),
    |w| (CompFlow-Patch | GT) -- 7 columns. Full-cell row = dense per-pixel
    maps (no arrows); one row per zoom region = colored quiver arrows,
    same HSV-direction convention throughout. Direction/magnitude GT
    columns share their colorbar scale with their estimated counterpart
    so the two are directly comparable."""
    overlay_rgb, vmax_diff = overlay_lost_gained(img1, img2, mask)

    dense_rgb, dense_vis = dense_direction_map(u_est, v_est, mask, img1, threshold=0.0)
    dense_rgb_gt, dense_vis_gt = dense_direction_map(u_gt, v_gt, mask, img1, threshold=0.0)

    mag = np.where(mask > 0, np.hypot(u_est, v_est), np.nan)
    mag_gt = np.where(mask > 0, np.hypot(u_gt, v_gt), np.nan)
    vmax_mag = float(np.nanmax([np.nanpercentile(mag, 99), np.nanpercentile(mag_gt, 99)]))

    titles = ["Frame 1\n(chromatin density)", "Frame 2\n(chromatin density)",
             "Density\nlost / gained", "Flow direction\n(CompFlow-Patch)",
             "Flow direction\n(ground truth)", "$|\\mathbf{w}|$ (px/frame)\n(CompFlow-Patch)",
             "$|\\mathbf{w}|$ (px/frame)\n(ground truth)"]
    NCOL = 7
    N_ZOOM = len(zoom_regions)
    NROW = 1 + N_ZOOM

    fig = plt.figure(figsize=(NCOL * 4.2, 5.1 * NROW + 0.4))
    gs = fig.add_gridspec(NROW + 1, NCOL, height_ratios=[1] * NROW + [0.09], hspace=0.2, wspace=0.22,
                          top=0.95, bottom=0.015, left=0.02, right=0.98)
    axF = [fig.add_subplot(gs[0, j]) for j in range(NCOL)]
    axZ = [[fig.add_subplot(gs[1 + i, j]) for j in range(NCOL)] for i in range(N_ZOOM)]
    axC = [fig.add_subplot(gs[NROW, j]) for j in range(NCOL)]

    # ---- full-cell row -----------------------------------------------------
    axF[0].imshow(img1, cmap="gray", vmin=0, vmax=255)
    axF[1].imshow(img2, cmap="gray", vmin=0, vmax=255)
    axF[2].imshow(overlay_rgb)
    for j, (rgb, vis) in ((3, (dense_rgb, dense_vis)), (4, (dense_rgb_gt, dense_vis_gt))):
        axF[j].imshow(img1, cmap="gray", vmin=0, vmax=255)
        axF[j].imshow(np.concatenate([rgb, vis[..., None].astype(np.float32)], axis=-1))
    axF[5].imshow(mag, cmap="plasma", vmin=0, vmax=vmax_mag)
    axF[6].imshow(mag_gt, cmap="plasma", vmin=0, vmax=vmax_mag)

    for ax, title in zip(axF, titles):
        ax.set_title(title, fontsize=12, fontweight="bold")
        for (y0, x0), color in zip(zoom_regions, zoom_colors):
            ax.add_patch(Rectangle((x0, y0), zoom_size, zoom_size, fill=False, edgecolor=color, linewidth=1.6))

    # ---- one row per zoom region --------------------------------------------
    for i, (y0, x0) in enumerate(zoom_regions):
        color = zoom_colors[i]
        ext = (x0 - 0.5, x0 + zoom_size - 0.5, y0 + zoom_size - 0.5, y0 - 0.5)
        sl = (slice(y0, y0 + zoom_size), slice(x0, x0 + zoom_size))
        ax = axZ[i]

        ax[0].imshow(img1[sl], cmap="gray", vmin=0, vmax=255, extent=ext)
        ax[1].imshow(img2[sl], cmap="gray", vmin=0, vmax=255, extent=ext)
        ax[2].imshow(overlay_rgb[sl], extent=ext)
        ax[3].imshow(img1[sl], cmap="gray", vmin=0, vmax=255, extent=ext)
        colored_quiver(ax[3], u_est, v_est, mask, y0, x0, zoom_size, step=2.0, scale=30.0, width=0.0032)
        ax[4].imshow(img1[sl], cmap="gray", vmin=0, vmax=255, extent=ext)
        colored_quiver(ax[4], u_gt, v_gt, mask, y0, x0, zoom_size, step=2.0, scale=30.0, width=0.0032)
        ax[5].imshow(mag[sl], cmap="plasma", vmin=0, vmax=vmax_mag, extent=ext)
        ax[6].imshow(mag_gt[sl], cmap="plasma", vmin=0, vmax=vmax_mag, extent=ext)

        ax[0].set_ylabel(f"zoom {i+1}", fontsize=12, fontweight="bold", color=color)
        for a in ax:
            a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
            for sp in a.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.8)

    # ---- shared colorbars ----------------------------------------------------
    for j in (0, 1):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 255), cmap="gray_r"),
                          cax=axC[j], orientation="horizontal")
        cb.set_ticks([0, 255]); cb.set_ticklabels(["low", "high"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax_diff, vmax_diff), cmap=overlay_colormap(vmax_diff)),
                      cax=axC[2], orientation="horizontal")
    cb.set_ticks([-vmax_diff, 0, vmax_diff]); cb.ax.set_xticklabels([f"-{vmax_diff:.2f}\n(lost)", "0", f"{vmax_diff:.2f}\n(gained)"])
    for j in (3, 4):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 360), cmap="hsv"),
                          cax=axC[j], orientation="horizontal", ticks=[0, 90, 180, 270, 360])
    for j in (5, 6):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax_mag), cmap="plasma"),
                          cax=axC[j], orientation="horizontal")
        cb.set_ticks([0, vmax_mag])

    for ax in [axF] + axZ:
        for a in ax:
            a.set_facecolor("black")
            a.set_xticks([]); a.set_yticks([])

    fig.suptitle(suptitle, fontsize=14, y=0.995)
    plt.show()
