#!/usr/bin/env python3
"""Visualize E-183 MaNGA harmonic halo stacking results.

Produces a 3-panel figure:
  Panel 1: Stacked NFW velocity residual profile delta(x) for D=16
  Panel 2: Fourier power at predicted ZD wavenumbers across CD dimensions
  Panel 3: 1/sqrt(N) sensitivity decay curve with known milestones

Replicates the Fourier analysis from cosmology_core/src/harmonic_stacking.rs
(NDFT at k_n = 2*pi*n/n_modes, normalized by valid-bin count).

Usage:
  python3 src/scripts/visualization/vis_e183_manga_stacking.py
  python3 src/scripts/visualization/vis_e183_manga_stacking.py \
      --results-dir data/results/e183 \
      --output data/results/e183/e183_manga_stacking.pdf

Reference: E-183 (harmonic_stacking.rs::detection_threshold, fourier_analysis_cd)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


SIGMA_V_FRAC = 0.05    # 5% NFW velocity uncertainty (matches Rust binary)
MIN_PER_BIN = 10       # same minimum as stacking run

# Known N milestones for the decay curve
# (label, N, marker, color, alpha)
N_MILESTONES = [
    ("SPARC (E-180)", 93, "^", "#e08000", 1.0),
    ("MaNGA 769 (interim)", 769, "s", "#4a90d9", 0.7),
    ("MaNGA 1056 (current)", 1056, "o", "#2ca02c", 1.0),
    ("MaNGA 2500 (target)", 2500, "*", "#d62728", 1.0),
]

CD_DIMS_COLORS = {
    16:   "#1f77b4",
    64:   "#ff7f0e",
    256:  "#2ca02c",
    1024: "#9467bd",
}


def detection_threshold(n: int, sigma_v_frac: float = SIGMA_V_FRAC) -> float:
    """Mirror of cosmology_core::harmonic_stacking::detection_threshold.

    alpha_zd_min = 2 * sigma_v_frac / (0.5 * sqrt(N)) = 4 * sigma_v_frac / sqrt(N)
    """
    if n <= 0:
        return math.inf
    return 4.0 * sigma_v_frac / math.sqrt(n)


def fourier_power_ndft(
    x: np.ndarray,
    delta: np.ndarray,
    n_contributing: np.ndarray,
    n_modes: int,
    min_per_bin: int = MIN_PER_BIN,
) -> tuple[np.ndarray, np.ndarray]:
    """NDFT at predicted ZD wavenumbers k_n = 2*pi*n/n_modes.

    Replicates harmonic_stacking.rs::fourier_analysis_cd exactly.
    Returns (power, phase) arrays of length n_modes.
    """
    power = np.zeros(n_modes)
    phase = np.zeros(n_modes)
    valid = n_contributing >= min_per_bin

    for n in range(1, n_modes + 1):
        k = 2.0 * math.pi * n / n_modes
        x_v = x[valid]
        d_v = delta[valid]
        count = valid.sum()
        if count == 0:
            continue
        re = np.sum(d_v * np.cos(k * x_v)) / count
        im = np.sum(d_v * np.sin(k * x_v)) / count
        power[n - 1] = re * re + im * im
        phase[n - 1] = math.atan2(im, re)

    return power, phase


def n_modes_from_dim(cd_dim: int) -> int:
    """n_modes = dim/2 - 1 (mirrors CdDimensionParams::new)."""
    return max(1, cd_dim // 2 - 1)


def load_stacking_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def draw_residual_panel(ax: plt.Axes, df: pd.DataFrame, n_galaxies: int) -> None:
    """Panel 1: stacked NFW residual profile (D=16)."""
    x = df["x"].to_numpy()
    dy = df["delta_stack"].to_numpy()
    err = df["delta_stack_err"].to_numpy()

    ax.axhline(0.0, color="0.5", lw=0.8, ls="--", zorder=1)
    ax.fill_between(x, dy - err, dy + err, alpha=0.25, color="#1f77b4", zorder=2)
    ax.plot(x, dy, color="#1f77b4", lw=1.2, zorder=3, label=f"N = {n_galaxies}")

    # Mark NFW scale radius (x = 1)
    ax.axvline(1.0, color="#e08000", lw=0.8, ls=":", zorder=1, alpha=0.8)
    ax.text(1.05, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] != 0 else 0.01,
            r"$x = r_s$", color="#e08000", fontsize=7, va="top")

    ax.set_xlabel(r"$x = r / r_s$", fontsize=9)
    ax.set_ylabel(r"$\Delta(x) = (v_{obs} - v_{NFW}) / v_{NFW}$", fontsize=8)
    ax.set_title(f"MaNGA Stacked NFW Residual  (D=16,  N={n_galaxies})", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0.4, 10.5)


def draw_fourier_panel(
    ax: plt.Axes,
    results_dir: Path,
    dims: list[int],
) -> None:
    """Panel 2: Fourier power at predicted ZD wavenumbers for each CD dim."""
    noise_floor = SIGMA_V_FRAC ** 2 / MIN_PER_BIN
    ax.axhline(noise_floor, color="0.5", lw=0.8, ls="--", zorder=1,
               label=f"noise floor (~{noise_floor:.2e})")

    for dim in dims:
        csv_path = results_dir / f"manga_stack_D{dim}.csv"
        df = load_stacking_csv(csv_path)
        if df is None:
            continue

        n_modes = n_modes_from_dim(dim)
        x = df["x"].to_numpy()
        dy = df["delta_stack"].to_numpy()
        nc = df["n_contributing"].to_numpy()

        power, _ = fourier_power_ndft(x, dy, nc, n_modes)
        mode_indices = np.arange(1, n_modes + 1)
        k_vals = 2.0 * math.pi * mode_indices / n_modes

        color = CD_DIMS_COLORS.get(dim, "gray")
        ax.semilogy(k_vals, power, "o-", ms=4, lw=1.2, color=color,
                    label=f"D={dim} ({n_modes} modes)", alpha=0.85)

    ax.set_xlabel(r"$k_n = 2\pi n / n_{modes}$  (units of $r_s^{-1}$)", fontsize=9)
    ax.set_ylabel("Fourier power  $|A_n|^2$", fontsize=9)
    ax.set_title("Fourier Power at Predicted ZD Wavenumbers", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0.0, 2.0 * math.pi + 0.3)


def draw_scaling_panel(ax: plt.Axes, current_n: int) -> None:
    """Panel 3: 1/sqrt(N) threshold decay with empirical milestones."""
    n_range = np.logspace(np.log10(80), np.log10(3500), 300)
    thr_theory = 4.0 * SIGMA_V_FRAC / np.sqrt(n_range)
    ax.loglog(n_range, thr_theory, "-", color="0.4", lw=1.2, zorder=1,
              label=r"$4\,\sigma_v / \sqrt{N}$  (theory)")

    # Target alpha_zd
    ax.axhline(0.004, color="#d62728", lw=0.9, ls="--", zorder=1, alpha=0.7,
               label=r"$\alpha_{zd} = 0.004$  target")

    for label, n, marker, color, alpha in N_MILESTONES:
        thr = detection_threshold(n)
        kw = dict(color=color, zorder=3, alpha=alpha, ms=8, lw=0)
        if n == current_n:
            kw["ms"] = 11
            kw["markeredgecolor"] = "black"
            kw["markeredgewidth"] = 1.0
        ax.loglog(n, thr, marker=marker, label=f"{label}  ({thr:.4f})", **kw)

    ax.set_xlabel("Number of stacked galaxies  $N$", fontsize=9)
    ax.set_ylabel(r"Detection threshold  $\alpha_{zd}^{min}$", fontsize=9)
    ax.set_title(r"Sensitivity Scaling  $\propto 1/\sqrt{N}$", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize E-183 MaNGA harmonic stacking results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results/e183"),
        help="Directory containing manga_stack_D*.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: <results-dir>/e183_manga_stacking.pdf)",
    )
    args = parser.parse_args()

    output = args.output or (args.results_dir / "e183_manga_stacking.pdf")
    dims = [16, 64, 256, 1024]

    # Load D=16 result for primary panels
    df16 = load_stacking_csv(args.results_dir / "manga_stack_D16.csv")
    if df16 is None:
        print(f"ERROR: manga_stack_D16.csv not found in {args.results_dir}")
        print("Run: make run-e183")
        return

    n_galaxies = int(df16["n_contributing"].max())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(
        f"E-183: MaNGA Harmonic Halo Stacking  (N={n_galaxies} galaxies)",
        fontsize=10, y=1.01,
    )

    draw_residual_panel(axes[0], df16, n_galaxies)
    draw_fourier_panel(axes[1], args.results_dir, dims)
    draw_scaling_panel(axes[2], n_galaxies)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
