# Experiments Portfolio Shortlist

This is a paper-facing shortlist of the most reproducible experiment artifacts currently present in the repo, with a method summary, a figure/table target (pgfplots-friendly), and one validation check per artifact.

## Theme A: Sedenion Zero-Divisors (Combinatorics -> Linear Algebra -> Geometry)

### A1) de Marrais box-kites replication (42 assessors; production rules)
- **Artifacts:** `docs/DE_MARRAIS_REPLICATION.md`, `tests/test_de_marrais_boxkites.py`, `tests/test_de_marrais_automorphemes.py`
- **Method summary:** Implement the assessor/box-kite combinatorics from de Marrais (2000) as explicit index-level rules on sedenion basis units, then validate enumerations (42 assessors; 7 box-kites; trefoil/zigzag structure) with deterministic tests. Extend to "automorphemes" derived from O-trips and verify incidence/coverage constraints.
- **pgfplots target:** Table: counts of assessors/box-kites/face-types; columns = `{object,count,derivation_rule,test_name}`.
- **Validation check:** `PYTHONWARNINGS=error make test` must reproduce all enumerations without relying on curated artifacts.

### A2) Reggiani diagonal-form ZDs (84 elements; annihilator nullity 4/4)
- **Artifacts:** `docs/REGGIANI_REPLICATION.md`, `src/gemini_physics/reggiani_replication.py`, `tests/test_reggiani_standard_zero_divisors.py`, `data/csv/reggiani_*`
- **Method summary:** Construct the 84 diagonal-form sedenion zero divisors derived from the 42 assessors as explicit vectors `(e_low +/- e_high)`. For each element, compute left/right multiplication matrices and estimate nullspaces via SVD to obtain annihilator nullities; confirm the invariant distribution and the 4-partner spanning property within diagonal-form candidates.
- **pgfplots target:** Histogram of nullity pairs `(dim Ann_L, dim Ann_R)` with counts; plus a table of representative ZDs with partner sets.
- **Validation check:** Regenerate CSVs and the "grand" nullity plot via `PYTHONWARNINGS=error make artifacts-reggiani` and verify with `make verify`.

## Theme B: A-infinity / m3 Diagnostics (Exact Arithmetic; Fano combinatorics)

### B1) CD(O)->S trilinear operation (m3) table; 42/168 split
- **Artifacts:** `src/gemini_physics/m3_cd_transfer.py`, `tests/test_m3_cd_transfer.py`, `src/m3_table_cd.py`
- **Method summary:** Define a specific contraction datum `(i,p,h)` from sedenions `S=O\oplusO` to octonions `O` and compute the rooted-tree trilinear operation `m3(x,y,z)=p \mu(h \mu(i x, i y), i z) - p \mu(i x, h \mu(i y, i z))` on octonion basis elements. Prove/verify combinatorial rules: among distinct triples there are exactly 42 scalar outputs and 168 pure-imaginary outputs; scalar outputs correspond exactly to Fano-plane lines and the sign flips with permutation parity.
- **pgfplots target:** Table keyed by ordered triple `(i,j,k)` with columns `{kind,coeff,index}`; plus an aggregated table of counts by kind.
- **Validation check:** `PYTHONWARNINGS=error venv/bin/python3 -m pytest -q tests/test_m3_cd_transfer.py` must pass (system pytest plugins may fail under warnings-as-errors).

## Theme C: Materials-Data Grounding (Reproducible subset; embedding sanity checks)

### C1) JARVIS subset fetch + embedding experiments
- **Artifacts:** `docs/MATERIALS_DATASETS.md`, `src/fetch_materials_jarvis_subset.py`, `src/materials_embedding_experiments.py`, `tests/test_materials_jarvis.py`
- **Method summary:** Fetch a deterministic subset of JARVIS entries (seeded sampling) and run embedding/visualization experiments with explicit provenance. Validate schemas and ensure plots follow the repo "grand quality" rendering spec.
- **pgfplots target:** Scatter of reduced embeddings (PCA/TSNE), colored by a chosen property (band gap, formation energy), with an accompanying table summarizing per-class statistics.
- **Validation check:** Ensure the dataset hash is recorded in `data/external/PROVENANCE.local.json` and reruns are deterministic under fixed seed.

