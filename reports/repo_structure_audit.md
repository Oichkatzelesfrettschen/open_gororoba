<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Repo Structure Audit (2026-02-04)

Scope: `/home/eirikr/Playground/ChatGPT Experiments`

## Top-level directories

- `.github/`: 1 files, 1 dirs, ~1.17 KB
- `.mamba/`: 26612 files, 2482 dirs, ~2.17 GB
- `archive/`: 25 files, 7 dirs, ~60.26 KB
- `benchmarks/`: 9 files, 2 dirs, ~51.75 KB
- `bin/`: 30 files, 0 dirs, ~16.80 MB
- `convos/`: 20 files, 0 dirs, ~30.88 MB
- `cpp/`: 360 files, 67 dirs, ~8.51 MB
- `curated/`: 611 files, 6 dirs, ~38.27 MB
- `data/`: 13129 files, 56 dirs, ~3.41 GB
- `docker/`: 1 files, 0 dirs, ~852 B
- `docs/`: 321 files, 19 dirs, ~6.23 MB
- `examples/`: 8 files, 1 dirs, ~37.80 KB
- `legacy/`: 1 files, 0 dirs, ~429 B
- `reports/`: 50 files, 0 dirs, ~1.41 GB
- `src/`: 595 files, 57 dirs, ~3.69 MB
- `tests/`: 242 files, 1 dirs, ~1.10 MB

## Structure signals (quick checks)

- `src/` exists: yes
- `tests/` exists: yes
- `docs/` exists: yes
- `data/` exists: yes
- `data/artifacts/` exists: yes
- `data/artifacts/ARTIFACTS_MANIFEST.csv` exists: yes
- `src/scripts/` exists: yes
- `src/verification/` exists: yes
- `src/gemini_physics/` exists: yes

## Notes: what is `gemini_physics`?

- `src/gemini_physics/` is the primary Python package for domain code (algebra, cosmology,
  materials, etc.). It exists to keep reusable library code separate from runnable scripts.
- `src/scripts/**` holds entrypoints and one-off pipelines; scripts import from `gemini_physics`
  (and other packages) rather than duplicating logic.
- Keeping domain code inside a package (instead of flat files under `src/`) improves testability,
  import hygiene, and reproducibility (e.g., network gating and determinism policies can be
  enforced centrally).

## Notes: current `src/` layout (high level)

- `src/gemini_physics/`: primary package for research code.
- `src/scripts/`: runnable scripts with contract headers (inputs/outputs/network).
- `src/verification/`: verifiers for repo contracts and reproducibility gates.
- Other packages under `src/` (for now) may represent legacy or domain-specific modules that
  are candidates for consolidation into `gemini_physics/` once import paths and tests are
  updated safely (phase-gated refactor).

## Suggested repo type classification

- This is a reproducible research monorepo: code + tests + docs + datasets + generated artifacts
  with strict provenance and offline-default behavior. Best practice is to keep generation
  explicit (artifact targets) and keep smoke/check verification-only.
