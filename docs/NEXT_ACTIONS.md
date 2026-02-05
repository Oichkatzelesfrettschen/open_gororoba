# Next Actions (Prioritized)

This is the short actionable list after the second-pass audit (2026-01-27).

## A. Claims -> Evidence (highest impact)

1) **Box-kites / 42 assessors / PSL(2,7) replication**
   - Implement a small, deterministic enumeration that reproduces de Marrais counts and structures.
   - Add tests under `tests/` that assert reproduced counts/identities.
   - (Done as a prerequisite) Add an explicit zero-divisor identity test in 16D (`tests/test_cayley_dickson_properties.py`).

2) **Reggiani (arXiv:2411.18881) alignment**
   - Extract the paper's exact definitions (`Z(S)`, `ZD(S)` etc.) into `docs/SEDENION_ATLAS.md`.
   - Add a minimal computational check aligned to those definitions.

3) **GWTC-3 provenance hardening**
   - Add a fetch script that downloads GWTC-3 "confident" events from GWOSC, records URL+date+checksum,
     and writes a cached CSV under `data/external/` with a machine-readable provenance JSON.

4) **Mass "clumping" hypothesis test**
   - Replace narrative "clumping implies modes" with a falsifiable statistical test:
     null models, selection effects, sensitivity checks, and a pre-registered threshold.

## B. Quality gates (engineering)

5) **Phased `ruff` expansion**
   - Add a second lint gate scope (e.g. `src/verification/**` + a curated list of scripts).
   - Track repo-wide lint counts via `make lint-all-stats` and burn them down incrementally.

6) **Grand image policy**
   - Decide if "grand" applies to all artifacts or only those tagged in filenames.
   - Optionally add `make verify-grand --enforce-all` as a strict CI step once the repo is upgraded.

7) **External data provenance**
   - Extend `data/external/PROVENANCE.local.json` into a richer provenance registry:
     source URL, license, query params, access date, and checksum.

## C. Experiments portfolio (paper synth)

8) Create `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`:
   - shortlist 5-10 artifacts
   - 1-2 paragraph method summary per artifact
   - one reproducibility check per artifact
