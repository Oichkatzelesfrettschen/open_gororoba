<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Fractional Laplacian: operator choices (Riesz vs spectral vs extension)

This repo uses the phrase "fractional Laplacian" only when the domain, boundary conditions, and
definition are stated explicitly. These choices are not interchangeable.

See also:
- `docs/BIBLIOGRAPHY.md` (primary sources)
- `docs/CONVOS_CONCEPTS_STATUS_INDEX.md` (CX-004)

## 1) Riesz / Fourier-multiplier definition (whole space or periodic)

On R^d (or on a periodic torus model), the standard "Riesz" fractional Laplacian is characterized by
the Fourier symbol:

  FFT[ (-Delta)^s u ](xi) = |xi|^(2s) * FFT[u](xi),  s > 0

Repo implementation (1D periodic grid):
- `src/gemini_physics/fractional_laplacian.py` -> `fractional_laplacian_periodic_1d`
- `tests/test_fractional_laplacian.py`

This is the natural operator for FFT-based fractional diffusion and fractional Schrodinger demos on a
periodic grid.

## 2) Spectral definition on bounded domains (Dirichlet/Neumann boundary)

On a bounded domain, "spectral fractional Laplacian" typically means a fractional power of the
Laplacian with a chosen boundary condition:

If -Delta phi_k = lambda_k phi_k (with Dirichlet or Neumann BCs), then

  (-Delta)^s u = sum_k lambda_k^s <u, phi_k> phi_k

This operator generally differs from the Riesz operator restricted to a bounded domain.

Repo implementation (1D discrete Dirichlet Laplacian on interior grid points):
- `src/gemini_physics/fractional_laplacian.py` -> `fractional_laplacian_dirichlet_1d`
- `tests/test_fractional_laplacian.py`

Note: this implementation is explicitly the fractional power of the discrete Dirichlet Laplacian
diagonalized by the discrete sine transform (DST-I), which is the appropriate "spectral" object on the
grid.

## 3) Extension problem (Caffarelli-Silvestre / Stinga-Torrea)

Caffarelli-Silvestre (and later generalizations) represent (-Delta)^s as a Dirichlet-to-Neumann map for
a local PDE in one higher dimension with a degenerate weight.

This is a powerful framework for analysis and for certain numerical methods, but it is not yet
implemented in this repo as an operator that can be cross-validated against the other definitions.

## 4) Practical guidance (what we currently claim)

- "Fractional Laplacian" in repo code currently means:
  - periodic Fourier-multiplier (Riesz/torus), or
  - discrete spectral Dirichlet fractional power (DST-based).
- We do not claim equivalence between these operators on bounded domains.
- Any "negative dimension PDE" narrative is treated as analytic continuation / dim-reg language only
  and must not be conflated with a literal geometric dimension.
