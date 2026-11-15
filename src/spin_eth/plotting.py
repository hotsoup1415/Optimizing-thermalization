"""Figures used across the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import PowerNorm

from .constants import KH_G, KH_H, PRX_G, PRX_H

#: Reference points overlaid on (g, h) landscapes.
REFERENCE_POINTS = (
    {"g": PRX_G, "h": PRX_H, "label": "PRX MC", "color": "red", "marker": "o", "ms": 8},
    {"g": KH_G, "h": KH_H, "label": "KH", "color": "cyan", "marker": "s", "ms": 8},
)

PHI_RATIO_LABEL = r"$\Phi / \Phi_{\min}$"


def plot_landscape(
    data,
    g_values,
    h_values,
    title="",
    cbar_label=r"$\Phi$",
    n_contours=50,
    gamma=0.35,
    vmax_percentile=80,
    cmap="viridis",
    highlight_points=REFERENCE_POINTS,
    ax=None,
):
    """Heat map of a (g, h) landscape (rows = h) with fine contours near the minimum.

    Draws a standalone figure, or into ``ax`` when given (for panel layouts).
    """
    data = np.array(data, dtype=float)
    G, H = np.meshgrid(g_values, h_values)

    vmin = np.nanmin(data)
    vmax = np.nanpercentile(data, vmax_percentile)
    if vmax <= vmin:
        vmax = np.nanmax(data)
    norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure
    small = 1.0 if standalone else 0.8
    im = ax.pcolormesh(G, H, data, shading="auto", cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax, extend="max", fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # Dense contours in the lowest 30% of the range, coarser above.
    levels_fine = np.linspace(vmin, vmin + 0.3 * (vmax - vmin), n_contours // 2)
    levels_coarse = np.linspace(vmin + 0.3 * (vmax - vmin), vmax, n_contours // 2)
    levels = np.unique(np.concatenate([levels_fine, levels_coarse]))
    cs = ax.contour(G, H, data, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, levels[::4], inline=True, fontsize=7 * small, fmt="%.3f")

    ih, ig = np.unravel_index(np.nanargmin(data), data.shape)
    ax.plot(g_values[ig], h_values[ih], "*", color="yellow", markersize=14 * small,
            markeredgecolor="black", markeredgewidth=1.0,
            label=f"min ({g_values[ig]:.3f}, {h_values[ih]:.3f}) = {data[ih, ig]:.4g}")

    for pt in highlight_points or ():
        ax.plot(pt["g"], pt["h"], marker=pt.get("marker", "o"), color=pt.get("color", "red"),
                markersize=pt.get("ms", 8) * small, markeredgecolor="white", markeredgewidth=1.0)
        if "label" in pt:
            ax.text(pt["g"] + 0.02, pt["h"] + 0.02, pt["label"], color=pt.get("color", "red"),
                    fontsize=9 * small, weight="bold")

    ax.set_xlabel("g", fontsize=12 * small)
    ax.set_ylabel("h", fontsize=12 * small)
    ax.set_title(title, fontsize=13 * small)
    ax.legend(loc="upper left", fontsize=9 * small)
    if standalone:
        fig.tight_layout()
        plt.show()
    return fig


def landscape_panels(panels, ncols, cbar_label=r"$\Phi$", panel_size=(5.2, 3.6), suptitle=None):
    """Grid of :func:`plot_landscape` panels.

    ``panels`` is a list of ``(data, g_values, h_values, title)`` tuples.
    """
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    for ax, (data, g, h, title) in zip(axes.flat, panels):
        plot_landscape(data, g, h, title=title, cbar_label=cbar_label, ax=ax)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    plt.show()
    return fig


def plot_min_scaling(L_values, minima, title, ylabel=r"min $\Phi$", ref_label=None):
    """Semilog plot of landscape minima vs L with a C 2^-L guide.

    ``minima`` maps a series label to a list of minima (one per L). The guide
    is normalized to the first point of the first series.
    """
    L_arr = np.asarray(L_values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, vals in minima.items():
        ax.semilogy(L_arr, vals, "o-", ms=7, label=label)
    first = np.asarray(next(iter(minima.values())), dtype=float)
    guide = 2.0 ** (-L_arr)
    guide *= first[0] / guide[0]
    ax.semilogy(L_arr, guide, "k--", lw=1.5, label=ref_label or r"$C \cdot 2^{-L}$")
    ax.set_xlabel("L", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(L_values)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    plt.show()
    return fig


def plot_mode_scan(
    grid,
    alphas,
    betas,
    evals,
    evecs,
    names,
    phi_ref,
    title,
    center_label,
    markers=(),
    note=None,
    savepath=None,
):
    """Plane scan along the two softest Hessian modes: heat map plus two 1D profiles.

    ``grid[i, j]`` is Phi at alpha = alphas[i] (mode 1), beta = betas[j] (mode 2),
    shown relative to ``phi_ref``. ``markers`` adds points to the heat map as
    dicts with keys ``alpha``, ``beta``, ``label`` and optional matplotlib style.
    """
    v1, v2 = evecs[:, 0], evecs[:, 1]
    rel = grid / phi_ref
    ia0 = int(np.argmin(np.abs(alphas)))
    ib0 = int(np.argmin(np.abs(betas)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{title}\n"
        f"mode 1 ($\\lambda$={evals[0]:.3f}): " + ", ".join(f"{n}={c:+.2f}" for n, c in zip(names, v1))
        + f"\nmode 2 ($\\lambda$={evals[1]:.3f}): " + ", ".join(f"{n}={c:+.2f}" for n, c in zip(names, v2)),
        fontsize=9, y=1.02)

    ax = axes[0]
    im = ax.imshow(rel, origin="lower", extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                   aspect="equal", cmap="RdYlGn_r", vmin=1.0, vmax=min(rel.max(), 2.5),
                   interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(PHI_RATIO_LABEL, fontsize=10)
    levels = [lv for lv in (1.005, 1.01, 1.05, 1.10, 1.20, 1.50) if lv <= rel.max()]
    if levels:
        cs = ax.contour(betas, alphas, rel, levels=levels, colors="k", linewidths=0.7, alpha=0.7)
        ax.clabel(cs, fmt="%.2fx", fontsize=7, inline=True)
    ax.plot(0, 0, "D", color="red", ms=12, mew=1.8, markeredgecolor="white", label=center_label, zorder=9)
    grid_min_ia, grid_min_ib = np.unravel_index(np.argmin(grid), grid.shape)
    a_min, b_min = alphas[grid_min_ia], betas[grid_min_ib]
    if abs(a_min) > 0.01 or abs(b_min) > 0.01:
        ax.plot(b_min, a_min, "^", color="orange", ms=9, mew=1.5, markeredgecolor="white",
                label=f"grid low ({a_min:+.2f},{b_min:+.2f})", zorder=7)
    for m in markers:
        style = {"marker": "*", "color": "royalblue", "ms": 16, "mew": 1.5, "markeredgecolor": "white", "zorder": 10}
        style.update({k: v for k, v in m.items() if k not in ("alpha", "beta")})
        ax.plot(m["beta"], m["alpha"], ls="none", **style)
    if note:
        ax.text(0.05, 0.05, note, transform=ax.transAxes, fontsize=8, color="gray")
    ax.axhline(0, c="gray", lw=0.3, ls="--", zorder=1)
    ax.axvline(0, c="gray", lw=0.3, ls="--", zorder=1)
    ax.set_xlabel(r"$\beta$ (mode 2)", fontsize=10)
    ax.set_ylabel(r"$\alpha$ (mode 1)", fontsize=10)
    ax.set_title("2D landscape", fontsize=10)
    ax.legend(fontsize=8)

    for ax, x, prof, color, sym, other, k in (
        (axes[1], alphas, rel[:, ib0], "b-o", r"$\alpha$", r"$\beta$", 0),
        (axes[2], betas, rel[ia0, :], "g-o", r"$\beta$", r"$\alpha$", 1),
    ):
        ax.plot(x, prof, color, ms=4, lw=1.5, label=r"$\Phi / \Phi_{\min}$")
        ax.axvline(0, c="r", lw=1.2, ls="--", label=center_label)
        ax.axhline(1.0, c="gray", lw=0.5, ls=":")
        ax.set_xlabel(f"{sym}  (mode {k + 1},  {other}=0)", fontsize=10)
        ax.set_ylabel(PHI_RATIO_LABEL, fontsize=10)
        ax.set_title(f"Profile along mode {k + 1}  ($\\lambda$={evals[k]:.3f})", fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=110, bbox_inches="tight")
    plt.show()
    return fig
