# Next Actions (Prioritized)

Updated 2026-02-07 after sprint completion. Previous audit: 2026-02-06.
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for full architecture and GR port plan.
Execution detail for missing datasets lives in `docs/ULTRA_ROADMAP.md` Section H.

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

## B. Quality gates (engineering)

5) **Phased `ruff` expansion**
   - Add a second lint gate scope (e.g. `src/verification/**` + a curated list of scripts).
   - Track repo-wide lint counts via `make lint-all-stats` and burn them down incrementally.

6) **Grand image policy**
   - Decide if "grand" applies to all artifacts or only those tagged in filenames.
   - Optionally add `make verify-grand --enforce-all` as a strict CI step once the repo is upgraded.

7) **External data provenance** -- PARTIALLY DONE
   - `data/external/PROVENANCE.local.json` is now richer with URL, license, access date, checksum.
   - Still need: query params for HEASARC fetches, automated provenance checks in CI.

## C. Experiments portfolio (paper synth)

8) Create `docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md`:
   - shortlist 5-10 artifacts
   - 1-2 paragraph method summary per artifact
   - one reproducibility check per artifact

## D. Dataset pillars -- DONE (Section H 21/21)

9) All 21 Section H items complete (2026-02-07).
   18 providers + 6 benchmark families + provenance verifier.
   1464 unit tests, 14 doc-tests, 0 clippy warnings.

## E. GR module expansion (Blackhole C++ port) -- DONE

10) All 30 tasks complete (2026-02-06).  18 gr_core modules, 394 tests.
    See `docs/ROADMAP.md` Section 5 for the full dependency graph and module listing.

## F. GPU ultrametric exploration -- DONE (I-011)

11) CUDA kernel via cudarc 0.19.1 on RTX 4070 Ti.
    10M triples x 1000 permutations x 9 catalogs: 82/472 significant at BH-FDR<0.05.
    Old I-008 conclusion (radio-transient-specific) overturned.
    See `docs/INSIGHTS.md` I-011 for full results.
