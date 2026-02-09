# Next Actions (Prioritized)

Updated 2026-02-09 after Sprint 24 completion.
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for full architecture and research plan.

## Current State (Sprint 24, 2026-02-09)

**Test count**: 2270 total (unit + integration + doc), 0 clippy warnings.
**Claims**: C-001..C-490 (490 total). **Insights**: I-001..I-017. **Binaries**: 40.

## Recent Sprint Summaries

- **S24**: Face sign census at dim=256: 17 regimes, 13.3M triangles in ~16s (C-488).
  Regime-to-GF(2) class correspondence: regimes = edge-count classes + 1, unifying
  motif census, GF(2) projective geometry, and face sign census (C-489). Parity-
  specific edge regularity REFUTED in non-pure regimes -- the Double 3:1 Law's
  algebraic mechanism is deeper than graph-theoretic counting (C-490). Universal
  Double 3:1 Law now verified across 243 components at 5 dimensions.
- **S23**: Face sign census at dim=128: 9 regimes, 821128 triangles in ~1s (C-484).
  Regime count formula corrected from 2*log2(dim/16)+1 to dim/16+1 (C-485). Edge
  count extremal formulas: E_max=C(dim/2-2,2)-(dim/4-1), E_min=3*dim/2-12 (C-486).
  Universal Double 3:1 Law discovered: OneSameTwoOpp=3*AllSame AND TwoSameOneOpp=
  3*AllOpposite in ALL 116 components across dims 16/32/64/128 (C-487). This
  supersedes C-483 (pure-regime case).
- **S22**: Face sign census extended to dim=64: 5-regime structure with 48328 triangles
  (C-482). Pure-regime 3:1 ratio law proved via antibalanced signed graph theory
  (C-483, verified across 24 pure-regime components at dims 16/32/64).
  Special heptacross at dim=32 identified (XOR label = dim/4). 84-edge pure-min components
  at dim=64 exactly reproduce dim=16's 42:14 census (embedded sedenion sub-algebra).
- **S21**: GF(2) separating degree extended to dim=128 (degree 5, +1 per doubling confirmed
  at 3 doublings, C-480 updated). Generic face sign census: dim=16 reproduces C-479,
  dim=32 discovers 3-regime structure with 2408 triangles (C-481). FaceSignPattern moved
  to boxkites.rs (canonical home). Parity-clique/XOR partner at dim=128 confirmed
  already verified.
- **S20**: Twist-delta pair correspondence RESOLVED (C-478, Fano Line Pairing Theorem).
  Lanyard sign census VERIFIED (C-479, 42+14 uniform). GF(2) separating degree at
  dim=64 RESOLVED (C-480, degree 4). Stale docs updated.
- **S19**: Ran experiments E-011/E-012/E-013. C-476 (ALP) partially verified
  (sparse graphs only). C-477 (Sky-Limit-Set) partially verified (A_{N-1} best
  Coxeter match). New insight I-017.
- **S18**: Experiment binaries E-011/E-012/E-013 implemented as CLIs with CSV output.
- **S17**: ET discrete billiard (symbolic dynamics, entropy, phase sweep),
  sky-limit-set Coxeter correspondence, cross-stack locality ALP formalization.
- **S16**: Multiplication coupling rho(b) resolved: C-466 REFUTED -- only identity
  has consistent linear coupling. Rank-adaptive reduced-subspace algorithm.
- **S15**: NullModelStrategy trait, CodebookNullTest, Baire-codebook bridge, Thesis I.
- **S14**: Monograph Layers 0-4: TypedCarrier, EncodingDictionary, elevated addition,
  InvariantSuite cross-validation.
- **S13**: Terminology normalization, glossary, Dynkin Convention 10.
- **S12**: Registry integrity, LaTeX superstructure, HDF5 export (hdf5-metno),
  algebra_core subdirectory reorg, mdbook chapters.
- **S11**: docpipe crate (pdfium-render), 19 papers extracted, TOML registry, mdbook skeleton.
- **S10**: De Marrais emanation architecture (L1-L18), 113 tests, 8 new claims.
- **S9**: Lattice codebook filtration, 8 monograph theses A-H.
- **S8**: CdMultTable, Hartigan dip test, mass-clumping, license GPL-2.0-only.

---

## Completed Workstreams

| Workstream | Status | Sprint | Key Metric |
|-----------|--------|--------|------------|
| A. Claims -> Evidence | DONE | S4-S9 | 467 claims, 118 backfill verified |
| B. Quality gates | DONE (except B.7 provenance CI) | S5-S9 | 0 clippy warnings, 1693 tests |
| C. Experiments portfolio | DONE | S8 | 10 experiments documented |
| D. Dataset pillars | DONE | S5-S6 | 30 providers, 21/21 Section H |
| E. GR module expansion | DONE | S3-S4 | 18 modules, 394 tests |
| F. GPU ultrametric | DONE | S4 | 82/472 sig at FDR<0.05 |
| G. Convos extraction | DONE | S5, S9 | 104 claims + 3 extracts, convos/ removed |
| H. Lattice codebook | DONE | S9 | 8 theses verified, I-015 |
| I. Emanation architecture | DONE | S10 | L1-L18, 113 tests, I-016 |
| J. Monograph layers 0-5 | DONE | S13-S15 | TypedCarrier through NullModel |
| K. Rho(b) coupling | DONE (REFUTED) | S16 | C-466 refuted, identity only |
| L. ALP + Sky-Limit-Set | DONE (PARTIAL) | S17-S19 | C-476/C-477 partially verified, I-017 |

---

## Open Items (Forward-Looking)

### Registry and Documentation Governance

0) **Complete TOML centralization for markdown mirrors** (Sprint 13)
   - `registry/knowledge_sources.toml` now indexes all tracked markdown files.
   - `registry/knowledge/docs/*.toml` now stores raw markdown capture (non-authoritative landing layer).
   - Curated operational registries added: `registry/roadmap.toml`, `registry/todo.toml`,
     `registry/next_actions.toml`, `registry/requirements.toml`.
   - Claims support registries now normalized:
     `registry/claims_tasks.toml`, `registry/claims_domains.toml`,
     `registry/claim_tickets.toml`.
   - `make registry` is now TOML-first by default; legacy markdown ingest is
     explicit via `make registry-ingest-legacy`.
   - Exporters now generate TOML-driven markdown mirrors under
     `docs/generated/*_REGISTRY_MIRROR.md` for insights, experiments, roadmap,
     todo, next-actions, requirements, claims-tasks, claims-domains, and claim-tickets.
   - Exporters also regenerate claims-support legacy mirrors:
     `docs/CLAIMS_TASKS.md`, `docs/claims/INDEX.md`,
     `docs/claims/CLAIMS_DOMAIN_MAP.csv`, and `docs/claims/by_domain/*.md`.
   - Mirror freshness is now enforced by
     `src/verification/verify_registry_mirror_freshness.py`.
   - Registry generators now emit deterministic stamps to avoid one-time timestamp churn.
   - Remaining policy decision: fully replace legacy narrative docs or keep
     them as non-authoritative overlays.
   - Keep markdown as human-facing mirrors, generated where practical.

### High Priority

1) ~~**Multiplication coupling rho(b) in GL(8,Z)** (C-466)~~ -- **REFUTED** (Sprint 16).
   Only identity element b=0 yields zero residual in rank-adaptive reduced-subspace algorithm.

2) **Primary-source citation for every claim** (ROADMAP 7.2)
   - Systematic sweep through CLAIMS_EVIDENCE_MATRIX.md.
   - Add missing WHERE STATED and bibliographic references.
   - Sprint 8 did 10 claims; ~200+ remain without primary sources.

3) **Paper-ready LaTeX pipeline** (ROADMAP 7.5)
   - `make latex` build from verified results.
   - Structured "hypotheses + tests + results" format.
   - Priority target: ultrametric core mining (I-013) or lattice codebook (I-015).

### Medium Priority

4) ~~**Twist-delta pair correspondence** (I-016)~~ -- **RESOLVED** (Sprint 20, C-478).
   - Root cause: `twist_transition_table()` used HashSet (nondeterministic target selection).
   - Fix: select the S-pairing (vent assessor pair with XOR=S, delta-consistent).
   - Fano Line Pairing Theorem: 4 vent assessors admit 3 complementary pairings whose
     XOR values are exactly the Fano line {S, perp[0], perp[1]}. All 3 roles realized.

5) ~~**Full lanyard classification from signed graph** (I-016)~~ -- **RESOLVED** (Sprint 20, C-479).
   - Cross-BK census: 56 faces = 42 TwoSameOneOpp + 14 AllOpposite, 0 Blues, 0 OneSameTwoOpp.
   - Perfectly uniform: 6+2 per BK. Only 2 of 4 sign patterns realized.
   - TwoSameOneOpp = trefoil lanyards, AllOpposite = triple-zigzag lanyards.

6) ~~**GF(2) separating degree at dim=64** (I-012 open question)~~ -- **RESOLVED** (Sprint 20, C-480).
   - dim=32: degree 3 (cubic). dim=64: degree 4 (quartic). Grows +1 per doubling.
   - At dim=64: 4 classes (9+8+7+7), PG(4,2) with 5-bit labels.
   - At degree 4: all 15 non-zero GF(2)^4 signatures achievable.

5) ~~**Extend parity-clique and XOR partner to dim=128+**~~ -- **VERIFIED** (Sprint 9-12).
   - Parity-clique REFUTED at dims 16, 32, 64, 128 (density 1.6% at 128D, 50% cross-parity).
   - XOR partner UNIVERSAL at dims 16, 32, 64, 128 (3968/3968 valid at 128D, mask=8).
   - dim=256 ZD adjacency is computationally infeasible (132M evaluations).

6) **External data provenance automation** (B.7)
   - Query params for HEASARC fetches.
   - Automated provenance checks in CI.

### Low Priority / Long-Term

7) ~~**Mass "clumping" hypothesis test** (A.4)~~ -- DONE (Sprint 8: mass-clumping binary)
8) ~~**Experiments portfolio shortlist** (C.9)~~ -- DONE (Sprint 8)
9) ~~**Fast basis-element multiplication table generator** (ROADMAP 4.2)~~ -- DONE (Sprint 8: CdMultTable)
10) ~~**Materials science second dataset** (ROADMAP 7.3)~~ -- DONE

11) **Coq/Rocq formalization** (ROADMAP 7.4)
    - Decide semantics for `has_right`/`reachable_delegation`.
    - Prove a minimal non-trivial theorem end-to-end.

12) **Lex-prefix filtration and octonion subalgebra connection** (Sprint 9 open)
    - The 8D lattice dimension matches the octonion dimension.
    - Is this a coincidence or does the octonion subalgebra structure
      directly constrain the lattice embedding?

13) **GPU tensor network contraction** (ROADMAP 8)
    - Extend cudarc GPU path beyond ultrametric testing.
    - Target: PEPS boundary contraction for larger system sizes.
