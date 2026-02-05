from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ELEMENTS_118 = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

ELEMENT_TO_INDEX = {el: i for i, el in enumerate(ELEMENTS_118)}


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


FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def composition_vector(formula: str) -> np.ndarray:
    counts = np.zeros(len(ELEMENTS_118), dtype=float)
    total = 0.0
    for sym, num in FORMULA_TOKEN_RE.findall(formula):
        if sym not in ELEMENT_TO_INDEX:
            continue
        c = float(num) if num else 1.0
        counts[ELEMENT_TO_INDEX[sym]] += c
        total += c
    if total <= 0:
        return counts
    return counts / total


def pca_embedding(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, explained_variance_ratio) where:
    - x is (n_samples, n_features)
    - scores is (n_samples, k)
    """
    x0 = x - np.mean(x, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    scores = x0 @ vt[:k].T
    # Explained variance ratio from singular values.
    var = (s**2) / (x.shape[0] - 1)
    evr = var / np.sum(var)
    return scores, evr[:k]


def distance_preservation_spearman(x_hi: np.ndarray, x_lo: np.ndarray, *, n_pairs: int = 5000) -> float:
    rng = np.random.default_rng(0)
    n = x_hi.shape[0]
    if n < 3:
        return float("nan")

    pairs = rng.integers(0, n, size=(n_pairs, 2))
    mask = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[mask]

    dhi = np.linalg.norm(x_hi[pairs[:, 0]] - x_hi[pairs[:, 1]], axis=1)
    dlo = np.linalg.norm(x_lo[pairs[:, 0]] - x_lo[pairs[:, 1]], axis=1)
    rho, _p = stats.spearmanr(dhi, dlo)
    return float(rho)


def main() -> None:
    spec = RenderSpec()
    _apply_dark_style(spec)

    df = pd.read_csv("data/csv/materials_jarvis_subset.csv")
    df = df.dropna(subset=["formula"]).copy()
    if len(df) == 0:
        raise SystemExit("No usable rows with formula found in materials_jarvis_subset.csv")

    x = np.vstack([composition_vector(str(f)) for f in df["formula"].values])

    # Use whatever numeric properties are present as color/size signals.
    color = pd.to_numeric(df.get("formation_energy_peratom"), errors="coerce").to_numpy()
    size = pd.to_numeric(df.get("optb88vdw_bandgap"), errors="coerce").to_numpy()

    color = np.nan_to_num(color, nan=np.nanmedian(color))
    size = np.nan_to_num(size, nan=np.nanmedian(size))
    size = 30.0 + 70.0 * (size - np.min(size)) / max(1e-12, (np.max(size) - np.min(size)))

    out_rows = []
    for k in (4, 8, 16, 32):
        scores, evr = pca_embedding(x, k=min(k, x.shape[1]))
        rho = distance_preservation_spearman(x, scores, n_pairs=5000)
        out_rows.append(
            {
                "k": k,
                "spearman_distance_preservation": rho,
                "explained_variance_ratio_sum": float(np.sum(evr)),
            }
        )

        fig_w = spec.width_px / spec.dpi
        fig_h = spec.height_px / spec.dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=spec.dpi)
        sc = ax.scatter(
            scores[:, 0],
            scores[:, 1],
            c=color,
            s=size,
            cmap="viridis",
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(
            f"Materials (JARVIS subset): PCA projection from 118D composition -> {k}D -> 2D",
            fontsize=22,
            fontweight="bold",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Formation energy per atom (eV/atom)")
        ax.text(
            0.02,
            0.02,
            f"EVR sum (first {k}): {float(np.sum(evr)):.3f}\n"
            f"Spearman(d_hi, d_{k}): {rho:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            bbox={"facecolor": "#111827", "edgecolor": "#374151", "alpha": 0.7, "pad": 8},
        )

        out_path = Path(f"data/artifacts/images/materials_pca_{k}d.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=spec.dpi, facecolor=spec.facecolor)
        plt.close(fig)

    out_df = pd.DataFrame(out_rows).sort_values("k")
    out_csv = Path("data/csv/materials_embedding_benchmarks.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    print("Wrote: data/artifacts/images/materials_pca_{4,8,16,32}d.png")


if __name__ == "__main__":
    main()

