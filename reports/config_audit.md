<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Config Audit Notes (2026-02-01)

## Build entrypoints

- Makefile path scan: no missing `src/`/`bin/`/`docker/` script references detected (excluding venv tokens).

## Pytest import path

- `pytest.ini` uses `pythonpath` entries for `src/scripts/*` so tests can import script modules by bare name
  (e.g., `fetch_planck_2018_spectra`, `measure_tensor_network_entropy`).
- Name-collision fix applied: `src/scripts/analysis/spectral_triple_toy.py` was renamed to
  `src/scripts/analysis/spectral_triple_toy_report.py` so it no longer shadows `src/spectral_triple_toy.py`.

## Docker entrypoint

- Docker wrapper script lives at `run_quantum_container.sh`; docs updated to match.

## Hygiene

- `__pycache__/` directories under `src/` and `tests/` are expected after running the suite; they should remain
  untracked/ignored and can be removed via `make clean` (or manual deletion) if desired.
