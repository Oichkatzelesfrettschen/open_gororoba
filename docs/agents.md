# Visualization Standards & Guidelines

## ASCII-only policy

All repo-authored docs, code, and patches should use ASCII characters only.
- Avoid Unicode punctuation (smart quotes, em dashes, arrows, symbols).
- Use ASCII spellings (e.g., "A_infty", "Cayley-Dickson", "Delta", "<=", "->").
- Source transcripts under `convos/` may contain Unicode; treat them as immutable inputs.
- Enforced by `make ascii-check` (or `python3 bin/ascii_check.py --check`).
- If you need to sanitize files, run `python3 bin/ascii_check.py --fix` and then manually clean up any `<U+....>` placeholders it may introduce in prose headings.

**Goal:** All generated visual artifacts must meet "Grand Quality" standards - high-resolution, aesthetically stunning, rigorous, and dark-mode optimized.

## 1. Technical Specifications
*   **Resolution:** Strictly **3160 x 2820** pixels.
*   **DPI:** Minimum 100 (adjust figure size accordingly, e.g., 31.6 x 28.2 inches at 100 DPI or equivalent).
*   **Format:** PNG (preferred) or PDF for vector export.
*   **Color Profile:** Dark Mode is **mandatory**.
    *   **Background:** `#0d0f14` (Deep Night) or `#121212` (OLED Black).
    *   **Foreground/Text:** White (`#ffffff`) or Off-White (`#e0e0e0`).
    *   **Grid:** Subtle Gray (`#374151` or `#555555`), low alpha (0.3 - 0.5).
    *   **Palettes:** High-contrast neon or pastel overlays (Cyan, Magenta, Lime, Amber) that pop against the dark background.

## 2. Stylistic Requirements
*   **Complexity:** Visuals must be **Multi-Layered**. Never produce a simple single-line chart if an overlay, density map, or comparative annotation can add depth.
*   **Annotations:**
    *   **Titles:** Large, bold, descriptive. Use "Great Fonts" (e.g., Serif for headers, Monospace for code/data).
    *   **Labels:** Every axis must have a label with physical units (LaTeX formatted, e.g., $M_{\odot}$, $\hbar$).
    *   **Insets/Overlays:** Use text boxes to highlight key data points (e.g., "LIGO Peak 1", "n=10").
*   **Aesthetics:** "Beautiful and Gorgeous." Avoid default Matplotlib styles. Remove top/right spines. Use alpha blending for overlapping data.

## 3. Python Implementation Snippet
Use this template for all plotting scripts:

```python
import matplotlib.pyplot as plt

# Global Render Parameters
W, H = 3160, 2820
DPI = 100
FIG_W, FIG_H = W / DPI, H / DPI

plt.rcParams.update({
    "figure.facecolor": "#0d0f14",
    "axes.facecolor": "#0d0f14",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#374151",
    "font.size": 16,
    "font.family": "sans-serif" # or serif for grand titles
})

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
ax = plt.gca()

# ... Plotting Logic ...

plt.tight_layout()
plt.savefig("data/artifacts/images/grand_visual_name.png", dpi=DPI, facecolor="#0d0f14")
```

## 4. Verification
Before finalizing, ensure the image is not cluttered, fonts are legible at full resolution, and the "Vibe Check" passes (does it look like Science Fiction UI?).
