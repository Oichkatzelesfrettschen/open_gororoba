<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Curated Artifacts Audit (Scope + Provenance)

This document is a lightweight, reproducible inventory of `curated/` to support ongoing cleanup and
conversion from "black box" artifacts to generated, validated outputs.

## Snapshot (as of this audit)

- Total files: **601**
- Total size: ~**28 MB**
- Dominant type: **CSV** (hundreds), plus a small number of PNGs and Coq build artifacts.

Top-level directory breakdown (counts / intent):
- `curated/01_theory_frameworks/`: theory-adjacent CSVs + Coq files (largest; mixed provenance)
- `curated/02_simulations_pde_quantum/`: simulation/plot artifacts (mixed provenance)
- `curated/03_benchmarks_numeric/`: numeric benchmark CSVs (mixed provenance)
- `curated/04_observational_datasets/`: "observational" CSVs (requires first-party cross-checking)
- `curated/05_summaries_indexes/`: summary/index artifacts (mixed provenance)

## Provenance hash log

`curated/PROVENANCE.local.json` records SHA256/size/mtime for every file under `curated/`.

To regenerate:
```bash
python3 bin/record_external_hashes.py --root curated --output curated/PROVENANCE.local.json
```

## Current policy

- Treat `curated/` as **non-authoritative** unless:
  1) a committed script generates the artifact, and
  2) the artifact is verified by tests or a verifier under `src/verification/`.
- Prefer new reproducible artifacts under `data/csv/` and `data/artifacts/images/`.

See also:
- `docs/LACUNAE.md` for the "black box artifacts" diagnosis.
- `curated/README.md` for directory-level caveats.
