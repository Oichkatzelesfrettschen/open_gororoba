<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Visualization Standards and Guidelines

This document is a visualization supplement to `AGENTS.md`.
Governance rules (ASCII-only, warnings-as-errors, source-first, provenance,
citations, and workflow policy) are defined in `AGENTS.md`.
If there is any conflict, `AGENTS.md` is authoritative.

## Governance Alignment

- ASCII-only: use ASCII punctuation and symbols in plot scripts, captions, and docs.
- Warnings-as-errors: run Python checks with `PYTHONWARNINGS=error`.
- Source-first: do not treat `convos/` as authoritative evidence for visual claims.
- Provenance: do not ship untracked external assets; record sources and hashes.
- Citations: do not delete existing citations; only append or clarify.

Quick sync checks:

```bash
python3 bin/ascii_check.py --check
PYTHONWARNINGS=error make lint
```

Goal: generated visuals should be high-resolution, rigorous, reproducible, and
dark-mode optimized.

## 1. Technical Specifications

- Resolution: strictly `3160 x 2820` pixels.
- DPI: minimum 100 (for example, `31.6 x 28.2` inches at 100 DPI).
- Format: PNG (preferred) or PDF (vector export).
- Color profile: dark mode is mandatory.
- Background: `#0d0f14` (Deep Night) or `#121212` (OLED Black).
- Foreground/text: `#ffffff` or `#e0e0e0`.
- Grid: `#374151` or `#555555`, low alpha (`0.3-0.5`).
- Palette: high-contrast cyan/magenta/lime/amber overlays that remain legible.

## 2. Stylistic Requirements

- Complexity: prefer multi-layer visuals over single unannotated traces.
- Titles: large, descriptive, and specific.
- Labels: every axis must include a label and unit.
- Insets/overlays: annotate key events or thresholds when interpretation benefits.
- Aesthetics: avoid default matplotlib style; remove top/right spines; use alpha blending where overlap is heavy.

## 3. Python Implementation Snippet

Use this template for plotting scripts:

```python
import matplotlib.pyplot as plt

W, H = 3160, 2820
DPI = 100
FIG_W, FIG_H = W / DPI, H / DPI

plt.rcParams.update(
    {
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
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

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = plt.gca()

# ... plotting logic ...

plt.tight_layout()
out_path = "data/artifacts/images/grand_visual_name.png"
plt.savefig(out_path, dpi=DPI, facecolor="#0d0f14")
```

## 4. Artifact and Claims Contract

- Save outputs under `data/artifacts/images/`.
- Use stable, descriptive filenames (for example, include dataset and parameter slice).
- If a figure supports a tracked claim, include claim IDs (`C-nnn`) in caption text or nearby documentation.
- Keep figure inputs reproducible from checked-in code and declared datasets.

## 5. Verification

Before finalizing:

- Confirm resolution is exactly `3160 x 2820`.
- Confirm text remains legible at full size.
- Confirm plot is visually coherent (not cluttered, clear layering).
- Run `python3 bin/ascii_check.py --check`.
- Run `PYTHONWARNINGS=error make lint` for changed Python plotting code.
