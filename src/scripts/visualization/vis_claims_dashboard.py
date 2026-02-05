"""Claims Evidence Matrix dashboard.

Color-coded grid of C-001 through C-024 showing verification status.
Reads statuses from docs/CLAIMS_EVIDENCE_MATRIX.md when available,
otherwise uses hardcoded statuses from the last known audit.

Output: data/artifacts/images/claims_dashboard_3160x2820.png
"""
# SCRIPT_CONTRACT: {"inferred":true,"inputs":[],"network":"forbidden","network_env":"","outputs":["data/artifacts/images/claims_dashboard_3160x2820.png"],"version":1}

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
WIDTH_PX = 3160
HEIGHT_PX = 2820
DPI = 316
FACECOLOR = "#0a0a0a"
FONT_FAMILY = "DejaVu Sans"

np.random.seed(42)

# ---------------------------------------------------------------------------
# Status categories and their display colors.
# ---------------------------------------------------------------------------
STATUS_COLORS = {
    "Verified":           "#22c55e",  # green
    "Partially verified": "#eab308",  # yellow
    "Unverified":         "#ef4444",  # red
    "Refuted":            "#171717",  # near-black (with red border)
    "Speculative":        "#3b82f6",  # blue
    "Modeled (Toy)":      "#8b5cf6",  # purple (treated as partial)
}

# Hardcoded fallback statuses from the latest audit (2026-01-30).
FALLBACK_STATUSES: dict[str, str] = {
    "C-001": "Verified",
    "C-002": "Verified",
    "C-003": "Verified",
    "C-004": "Verified",
    "C-005": "Partially verified",
    "C-006": "Verified",
    "C-007": "Speculative",
    "C-008": "Unverified",
    "C-009": "Unverified",
    "C-010": "Speculative",
    "C-011": "Speculative",
    "C-012": "Refuted",
    "C-013": "Verified",
    "C-014": "Verified",
    "C-015": "Verified",
    "C-016": "Verified",
    "C-017": "Verified",
    "C-018": "Verified",
    "C-019": "Partially verified",
    "C-020": "Refuted",
    "C-021": "Refuted",
    "C-022": "Modeled (Toy)",
    "C-023": "Modeled (Toy)",
    "C-024": "Verified",
}

# Short descriptions for each claim.
CLAIM_TITLES: dict[str, str] = {
    "C-001": "CD non-assoc at 8D+",
    "C-002": "16D sedenion ZDs",
    "C-003": "42 assessors / 7 BKs",
    "C-004": "PSL(2,7) action",
    "C-005": "Reggiani manifolds",
    "C-006": "GWTC-3 snapshot",
    "C-007": "BH mass clumping",
    "C-008": "Frac Schrodinger",
    "C-009": "TN entropy scaling",
    "C-010": "Metamat absorber",
    "C-011": "Gravastar equiv",
    "C-012": "Dark energy neg-dim",
    "C-013": "GoTo automorphemes",
    "C-014": "Annihilator dim=4",
    "C-015": "4 ZD partners",
    "C-016": "m3 trilinear 42/168",
    "C-017": "XOR-bucket heuristic",
    "C-018": "Wheels axioms",
    "C-019": "Wheels + CD ZDs",
    "C-020": "Legacy 16D adjacency",
    "C-021": "1024D lattice map",
    "C-022": "Surreal time dyad",
    "C-023": "Lie fiber policy",
    "C-024": "C++ CD kernels",
}


def _parse_statuses_from_md(md_path: Path) -> dict[str, str]:
    """Try to parse claim statuses from the markdown table."""
    statuses: dict[str, str] = {}
    if not md_path.exists():
        return statuses
    text = md_path.read_text(encoding="utf-8")
    # Match rows like: | C-001 | ... | **Verified (math)** | ...
    pattern = re.compile(
        r"\|\s*(C-\d{3})\s*\|"    # claim ID
        r"[^|]*\|"                 # claim text
        r"[^|]*\|"                 # where stated
        r"\s*\*\*([^*]+)\*\*"     # **Status (details)**
    )
    for m in pattern.finditer(text):
        cid = m.group(1)
        raw = m.group(2).strip()
        # Normalize to our categories
        low = raw.lower()
        if "refuted" in low:
            statuses[cid] = "Refuted"
        elif "verified" in low and "partial" not in low and "unverified" not in low:
            statuses[cid] = "Verified"
        elif "partial" in low:
            statuses[cid] = "Partially verified"
        elif "unverified" in low:
            statuses[cid] = "Unverified"
        elif "speculative" in low:
            statuses[cid] = "Speculative"
        elif "modeled" in low or "toy" in low:
            statuses[cid] = "Modeled (Toy)"
        else:
            statuses[cid] = "Unverified"
    return statuses


def _get_statuses() -> dict[str, str]:
    md_path = Path("docs/CLAIMS_EVIDENCE_MATRIX.md")
    parsed = _parse_statuses_from_md(md_path)
    # Merge: parsed overrides fallback where available.
    result = dict(FALLBACK_STATUSES)
    result.update(parsed)
    return result


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": FACECOLOR,
        "axes.facecolor": FACECOLOR,
        "axes.edgecolor": FACECOLOR,
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY],
    })


def main() -> None:
    _apply_style()
    statuses = _get_statuses()

    claim_ids = sorted(statuses.keys(), key=lambda c: int(c.split("-")[1]))
    n = len(claim_ids)
    cols = 6
    rows = (n + cols - 1) // cols

    fig_w = WIDTH_PX / DPI
    fig_h = HEIGHT_PX / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-0.5, rows + 2.0)
    ax.axis("off")
    ax.invert_yaxis()

    # Title
    ax.text(cols / 2, -0.3, "Claims Evidence Dashboard",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="white")

    cell_w = 0.88
    cell_h = 0.75

    for idx, cid in enumerate(claim_ids):
        col = idx % cols
        row = idx // cols
        cx = col + 0.5
        cy = row + 1.0

        status = statuses.get(cid, "Unverified")
        color = STATUS_COLORS.get(status, "#6b7280")
        edge_color = "#ef4444" if status == "Refuted" else "#374151"
        face_alpha = 0.85 if status != "Refuted" else 0.3

        rect = mpatches.FancyBboxPatch(
            (cx - cell_w / 2, cy - cell_h / 2), cell_w, cell_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=edge_color, linewidth=1.5,
            alpha=face_alpha,
        )
        ax.add_patch(rect)

        # Claim ID
        ax.text(cx, cy - 0.18, cid, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        # Short title
        title = CLAIM_TITLES.get(cid, "")
        ax.text(cx, cy + 0.05, title, ha="center", va="center",
                fontsize=7, color="white", alpha=0.9)
        # Status label
        ax.text(cx, cy + 0.22, status, ha="center", va="center",
                fontsize=7, color="white", fontstyle="italic", alpha=0.8)

    # Legend
    legend_y = rows + 1.5
    legend_items = [
        ("Verified", "#22c55e"),
        ("Partially verified", "#eab308"),
        ("Unverified", "#ef4444"),
        ("Refuted", "#171717"),
        ("Speculative", "#3b82f6"),
        ("Modeled (Toy)", "#8b5cf6"),
    ]
    for li, (label, color) in enumerate(legend_items):
        lx = li * 1.0 + 0.3
        ax.plot(lx, legend_y, "s", markersize=12, color=color,
                markeredgecolor="#555555", markeredgewidth=0.8)
        ax.text(lx + 0.12, legend_y, label, va="center", fontsize=8,
                color="#d1d5db")

    out_path = Path("data/artifacts/images/claims_dashboard_3160x2820.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=FACECOLOR)
    plt.close(fig)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
