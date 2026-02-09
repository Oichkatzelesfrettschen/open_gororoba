<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Package Index (2026-02-01)

This is a lightweight inventory of Python packages as discovered by `setuptools.find_packages()`
from `pyproject.toml` (`package-dir = src`).

## Core package

- `gemini_physics` (primary library code; grouped into subpackages below)

## Subpackages

- `gemini_physics.algebra`
- `gemini_physics.audit`
- `gemini_physics.cd`
- `gemini_physics.cosmology`
- `gemini_physics.engineering`
- `gemini_physics.geometry`
- `gemini_physics.kinematics`
- `gemini_physics.materials`
- `gemini_physics.ml`
- `gemini_physics.padic`
- `gemini_physics.quantum_info`
- `gemini_physics.stellar_cartography`

## Additional top-level packages

- `holography` (toy/hypothesis checks for information-geometry links)
- `quantum` (quantum / condensed-matter style toy models and benchmarks)
- `spectral` (linear algebra + spectral toys; inverse-spectral counterexamples live here)
- `tensor_networks` (tensor-network entropy measurements)
- `verification` (repo quality gates and claim verifiers)

## Optional dependency groups (see `pyproject.toml`)

- `dev` (pytest, ruff)
- `analysis` (networkx, ripser, persim, optional scikit-learn)
- `astro` (astroquery/getdist/gwpy/h5py/healpy/psrqpy)
- `particle` (uproot/awkward/particle)
- `quantum` (qiskit stack pinned for Python < 3.13; intended for Docker)
