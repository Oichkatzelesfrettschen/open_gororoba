# Next Actions (Prioritized)

Updated 2026-02-07 after Sprint 5 completion. Previous audit: 2026-02-06.
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for full architecture and GR port plan.
Execution detail for missing datasets lives in `docs/ULTRA_ROADMAP.md` Section H.

## Sprint 5 Summary (2026-02-07)

**Test count**: 1525 unit + 14 doc-tests = 1539 total, 0 clippy warnings.

Completed this sprint:
- P1-P5: Deleted 10 Python files with confirmed Rust equivalents, ported 4 analysis scripts.
- P6-P8: Built Rust `claims-audit` and `claims-verify` binaries (replaces 11 Python scripts).
- P9: Deleted 20 Python claims scripts (-2996 lines).
- D3: Added 21 synthetic offline tests for catalog parsers (CHIME, Fermi GBM, Pantheon+, DESI BAO).
- BT1-BT2: Audited and batch-updated 118 backfill claims (all already Verified in matrix).
- E1-E3: Convos audit (104 claims, 30 new, 5 refutations), CX-026..028 added.
- QG1: Full quality gate passed.

## A. Claims -> Evidence (highest impact)

1) **Box-kites / 42 assessors / PSL(2,7) replication** -- DONE
   - Rust: `crates/algebra_core/src/boxkites.rs` (production rules, automorphemes, motif census).
   - Tests: 42 assessors, 7 box-kites at dim=16; 15 components at dim=32.

2) **Reggiani (arXiv:2411.18881) alignment** -- DONE
   - Rust: `crates/algebra_core/src/reggiani.rs` (84 standard ZDs, partner enumeration).
   - Annihilator SVD: `crates/algebra_core/src/annihilator.rs`.

3) **GWTC-3 provenance hardening** -- DONE
   - Rust: `crates/data_core/src/catalogs/gwtc.rs` (combined GWTC catalog, 219 events).
   - Provenance: `data/external/PROVENANCE.local.json` with URL + checksum.

4) **Mass "clumping" hypothesis test**
   - Replace narrative "clumping implies modes" with a falsifiable statistical test:
     null models, selection effects, sensitivity checks, and a pre-registered threshold.

5) **Claims infrastructure** -- DONE
   - Rust `claims-audit` binary: matrix metadata, evidence links, artifact file checks.
   - Rust `claims-verify` binary: 6 check modes (metadata, evidence, where-stated, tasks, domains, providers).
   - 118 backfill TODO items confirmed Verified and batch-updated to DONE.

## B. Quality gates (engineering)

6) **Phased `ruff` expansion**
   - Add a second lint gate scope (e.g. `src/verification/**` + a curated list of scripts).
   - Track repo-wide lint counts via `make lint-all-stats` and burn them down incrementally.

7) **External data provenance** -- PARTIALLY DONE
   - `data/external/PROVENANCE.local.json` is now richer with URL, license, access date, checksum.
   - Still need: query params for HEASARC fetches, automated provenance checks in CI.

8) **Dataset endpoint validation** -- DONE (2026-02-07)
   - Ran `fetch-datasets` against all 30 providers: 22 tested, all pass.
   - Fixed 3 broken geophysical URLs: WMM 2025 (NOAA path change),
     GRACE GGM05S (ICGEM hash change), GRACE-FO (/sp/ -> /getseries/ + RL06.3).
   - 8 large providers (EHT, WMAP, Planck chains, GRAIL, EGM2008, DE440/441)
     untested (multi-GB, on-demand only).
   - Deterministic row-count / column-integrity tests already in D3 (Sprint 5).

## C. Experiments portfolio (paper synth)

9) Create `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`:
   - shortlist 5-10 artifacts
   - 1-2 paragraph method summary per artifact
   - one reproducibility check per artifact

## D. Dataset pillars -- DONE (Section H 21/21)

10) All 21 Section H items complete (2026-02-07).
    18 providers + 6 benchmark families + provenance verifier.
    1517 unit tests, 14 doc-tests, 0 clippy warnings.

## E. GR module expansion (Blackhole C++ port) -- DONE

11) All 30 tasks complete (2026-02-06).  18 gr_core modules, 394 tests.
    See `docs/ROADMAP.md` Section 5 for the full dependency graph and module listing.

## F. GPU ultrametric exploration -- DONE (I-011)

12) CUDA kernel via cudarc 0.19.1 on RTX 4070 Ti.
    10M triples x 1000 permutations x 9 catalogs: 82/472 significant at BH-FDR<0.05.
    Old I-008 conclusion (radio-transient-specific) overturned.
    See `docs/INSIGHTS.md` I-011 for full results.

## G. Convos extraction -- DONE (Sprint 5)

13) 104 claims identified across 7 convo files (6 compass artifacts + 1 large convo).
    56 already in matrix, 18 in CX-index, 30 new.
    5 refutations documented. CX-026..028 added to concept index.
    See `docs/CONVOS_CONCEPTS_STATUS_INDEX.md` audit summary section.
