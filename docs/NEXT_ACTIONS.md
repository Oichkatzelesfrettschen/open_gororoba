# Next Actions (Prioritized)

Updated 2026-02-07 after Sprint 6 completion. Previous audit: 2026-02-07 (Sprint 5).
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for full architecture and GR port plan.
Execution detail for missing datasets lives in `docs/ULTRA_ROADMAP.md` Section H.

## Sprint 6 Summary (2026-02-07)

**Test count**: 1525 unit + 14 doc-tests = 1539 total, 0 clippy warnings.

Completed this sprint:
- D1/D2: Dataset endpoint validation -- 22/30 providers tested, 3 broken URLs fixed
  (WMM 2025 NOAA path, GGM05S ICGEM hash, GRACE-FO /sp/ -> /getseries/ + RL06.3).
- EHT: Expanded from 2 CSV-only to 6 multi-format providers (UVFITS+CSV+TXT).
  All public EHT releases covered: M87 2017/2018, Sgr A*, 3C279, Cen A, M87 Legacy.
- Roadmap reconciliation: Section 4.1 (6 source-first items), 4.2 (viz hygiene),
  7.1 (ruff obsolete), 7.2 (Reggiani done), 7.6 (Section H complete), 8 (motif census done).
- Test counts synchronized across ROADMAP.md, TODO.md, NEXT_ACTIONS.md.

## Sprint 5 Summary (2026-02-07)

- P1-P5: Deleted 10 Python files with confirmed Rust equivalents, ported 4 analysis scripts.
- P6-P8: Built Rust `claims-audit` and `claims-verify` binaries (replaces 11 Python scripts).
- P9: Deleted 20 Python claims scripts (-2996 lines).
- D3: Added 21 synthetic offline tests for catalog parsers (CHIME, Fermi GBM, Pantheon+, DESI BAO).
- BT1-BT2: Audited and batch-updated 118 backfill claims (all already Verified in matrix).
- E1-E3: Convos audit (104 claims, 30 new, 5 refutations), CX-026..028 added.
- QG1: Full quality gate passed.

---

## Completed Workstreams

| Workstream | Status | Sprint | Key Metric |
|-----------|--------|--------|------------|
| A. Claims -> Evidence | DONE | S4-S5 | 459 claims, 118 backfill verified |
| B. Quality gates | DONE (except B.7 provenance CI) | S5-S6 | 0 clippy warnings, 1539 tests |
| C. Experiments portfolio | Open | -- | -- |
| D. Dataset pillars | DONE | S5-S6 | 30 providers, 21/21 Section H |
| E. GR module expansion | DONE | S3-S4 | 18 modules, 394 tests |
| F. GPU ultrametric | DONE | S4 | 82/472 sig at FDR<0.05 |
| G. Convos extraction | DONE | S5 | 104 claims, 5 refutations |

---

## Open Items (Forward-Looking)

### High Priority

1) **Mass "clumping" hypothesis test** (A.4)
   - Replace narrative "clumping implies modes" with a falsifiable statistical test:
     null models, selection effects, sensitivity checks, and a pre-registered threshold.
   - Requires: GWTC-3 mass distributions from data_core catalogs.

2) **Experiments portfolio shortlist** (C.9)
   - Create `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`:
     shortlist 5-10 artifacts, 1-2 paragraph method summary per artifact,
     one reproducibility check per artifact.

3) **Primary-source citation for every claim** (ROADMAP 7.2)
   - Systematic sweep through CLAIMS_EVIDENCE_MATRIX.md.
   - Add missing WHERE STATED and bibliographic references.

### Medium Priority

4) **Fast basis-element multiplication table generator** (ROADMAP 4.2)
   - 16D/32D with cache + checksum.
   - Would accelerate zero-divisor searches and motif enumeration.

5) **External data provenance automation** (B.7)
   - Query params for HEASARC fetches.
   - Automated provenance checks in CI.

6) **Paper-ready LaTeX pipeline** (ROADMAP 7.5)
   - `make latex` build from verified results.
   - Structured "hypotheses + tests + results" format.

### Low Priority / Long-Term

7) ~~**Materials science second dataset** (ROADMAP 7.3)~~ -- DONE
   - AFLOW full-database provider (AFLUX REST API), JARVIS registered as DatasetProvider.
   - Magpie-style composition featurizer (54-dim) + OLS baselines + materials-baseline CLI.

8) **Coq/Rocq formalization** (ROADMAP 7.4)
   - Decide semantics for `has_right`/`reachable_delegation`.
   - Prove a minimal non-trivial theorem end-to-end.

9) **Pole-aware plotting** (ULTRA_ROADMAP C)
   - Optional residue plots / annotated singularities.
