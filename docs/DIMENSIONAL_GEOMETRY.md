<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Dimensional Geometry: Analytic Continuation (-4D -> 32D)

This repo includes a **mathematically standard** notion of "non-integer" and even "negative"
dimension via **analytic continuation** of the Gamma-function formulas for:

- the unit (d-1)-sphere surface area in R^d
- the d-ball volume

These quantities become *meromorphic* in `d` (they have poles where the Gamma function has poles).

## Implemented formulas

Let `d` be the ambient dimension.

- Surface area of the unit (d-1)-sphere:
  \[
  S_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}
  \]
- Volume of the d-ball of radius `r`:
  \[
  V_d(r) = \frac{\pi^{d/2}}{\Gamma(d/2+1)} r^d
  \]

Implementation: `src/gemini_physics/dimensional_geometry.py`.

## Visualizations produced

Run:
```bash
PYTHONWARNINGS=error ./venv/bin/python3 src/vis_dimensional_geometry.py
```

Outputs:
- `data/artifacts/images/dimensional_geometry_-4_to_16.png`
- `data/artifacts/images/dimensional_geometry_0_to_32.png`
- `data/csv/dimensional_geometry_-4_to_16.csv`
- `data/csv/dimensional_geometry_0_to_32.csv`

## Interpretation (scope-safe)

These plots are **pure geometry**. Any mapping from "negative dimension" to a physical
model (e.g., PDE operators, cosmology, materials) must be treated separately and justified
with first-party sources and falsifiable experiments.
