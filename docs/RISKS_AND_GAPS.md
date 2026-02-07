# Risks and Gaps (Open Items)

## Scientific validity risks

- Several documents assert strong physics conclusions without a falsifiable model, null-model comparisons, or
  first-party citations. Treat these as hypotheses until verified.
- Some plots appear "interpretive" (curves overlaid as theory guides). These must be clearly labeled as such
  and separated from observational data plots.

## Reproducibility risks

- Large `curated/` and `data/` corpora contain many files with unclear provenance.
- Some scripts depend on optional libraries and/or network downloads; they should record provenance and be
  robust to missing data.

## Engineering gaps

- `make lint` currently gates `src/gemini_physics/` and `tests/` only; expanding to all scripts will require
  phased cleanup.
- The "C++ accelerator" roadmap item is still open (Numba likely covers many cases, but the need should be
  benchmark-driven).
- Qiskit workflows are containerized; host installs may not be supported on bleeding-edge Python versions.

## Recommended next milestones

1. Convert each claim in `docs/CLAIMS_EVIDENCE_MATRIX.md` into a small test + citation update.
2. Add a second materials dataset backend for cross-validation (OQMD or NOMAD).
3. Add a reproducible GWTC-3 refetch + checksum workflow and tie LIGO plots directly to it.
4. Execute and track missing cosmology/ephemeris/geodesy pillars in
   `docs/ULTRA_ROADMAP.md` Section H.
