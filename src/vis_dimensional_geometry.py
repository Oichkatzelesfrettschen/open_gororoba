from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from gemini_physics.dimensional_geometry import sample_dimensional_range


@dataclass(frozen=True)
class RenderSpec:
    width_px: int = 3160
    height_px: int = 2820
    dpi: int = 100
    facecolor: str = "#0d0f14"


def _apply_dark_style(spec: RenderSpec) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": spec.facecolor,
            "axes.facecolor": spec.facecolor,
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "grid.color": "#374151",
            "font.size": 16,
            "font.family": "sans-serif",
        }
    )


def _safe_real(x: np.ndarray) -> np.ndarray:
    """
    For plotting, project complex-valued results to reals.

    For positive radii, the analytic-continuation formulas should be real-valued away from poles.
    If numerical noise produces small imaginary parts, drop them; otherwise mark as NaN.
    """
    y = np.real_if_close(x, tol=1e6)
    if np.iscomplexobj(y):
        imag = np.abs(np.imag(y))
        real = np.real(y)
        real[imag > 1e-8] = np.nan
        return real
    return np.asarray(y, dtype=float)


def _plot_range(d_min: float, d_max: float, *, out_png: str, out_csv: str) -> None:
    spec = RenderSpec()
    _apply_dark_style(spec)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ds, vols, areas = sample_dimensional_range(d_min, d_max, n=4001, r=1.0)
    v = _safe_real(vols)
    s = _safe_real(areas)

    # Identify poles (gamma poles) in this plotting window for visual markers.
    poles = []
    for k in range(0, 64):
        pole = -2.0 * k
        if d_min <= pole <= d_max:
            poles.append(pole)

    fig_w = spec.width_px / spec.dpi
    fig_h = spec.height_px / spec.dpi
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=spec.dpi)
    ax_v, ax_s, ax_vlog, ax_slog = axes.ravel()

    for ax in (ax_v, ax_s, ax_vlog, ax_slog):
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
        for pole in poles:
            ax.axvline(pole, color="#ff5cff", alpha=0.15, linewidth=1.0)

    ax_v.plot(ds, v, color="#00f7ff", linewidth=2.0)
    ax_v.set_title(r"$V_d(1) = \pi^{d/2} / \Gamma(d/2 + 1)$", fontsize=18)
    ax_v.set_xlabel("Dimension d")
    ax_v.set_ylabel("Volume (signed)")

    ax_s.plot(ds, s, color="#ffb000", linewidth=2.0)
    ax_s.set_title(r"$S_{d-1} = 2\pi^{d/2} / \Gamma(d/2)$", fontsize=18)
    ax_s.set_xlabel("Dimension d")
    ax_s.set_ylabel("Surface area (signed)")

    ax_vlog.plot(ds, np.log10(np.abs(v) + 1e-30), color="#00f7ff", linewidth=2.0)
    ax_vlog.set_title(r"$\log_{10} |V_d(1)|$", fontsize=18)
    ax_vlog.set_xlabel("Dimension d")
    ax_vlog.set_ylabel("log10(abs(volume))")

    ax_slog.plot(ds, np.log10(np.abs(s) + 1e-30), color="#ffb000", linewidth=2.0)
    ax_slog.set_title(r"$\log_{10} |S_{d-1}|$", fontsize=18)
    ax_slog.set_xlabel("Dimension d")
    ax_slog.set_ylabel("log10(abs(area))")

    fig.suptitle(
        f"Analytic Continuation of Ball Volume and Sphere Area (d in [{d_min}, {d_max}])",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=spec.dpi, facecolor=spec.facecolor)

    data = np.column_stack([ds, v, s])
    np.savetxt(out_csv, data, delimiter=",", header="d,ball_volume_r1,unit_sphere_area", comments="")


def main() -> None:
    _plot_range(
        -4.0,
        16.0,
        out_png="data/artifacts/images/dimensional_geometry_-4_to_16.png",
        out_csv="data/csv/dimensional_geometry_-4_to_16.csv",
    )
    _plot_range(
        0.0,
        32.0,
        out_png="data/artifacts/images/dimensional_geometry_0_to_32.png",
        out_csv="data/csv/dimensional_geometry_0_to_32.csv",
    )


if __name__ == "__main__":
    main()
