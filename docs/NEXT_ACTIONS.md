# Next Actions (Prioritized)

Updated 2026-02-08 after Sprint 9 completion.
**See also:** [`docs/ROADMAP.md`](ROADMAP.md) for full architecture and research plan.

## Sprint 9 Summary (2026-02-08)

**Test count**: 1693 total (unit + integration + doc), 0 clippy warnings.
**Claims**: C-001..C-467 (467 total). **Insights**: I-001..I-015.

Completed this sprint:
- Monograph theses A-H: 8/8 verified (lattice codebook filtration)
- Novel discovery: all filtration transitions are lexicographic prefix cuts
- S_base = 2187 = 3^7 (base universe), Lambda_32 = pinned corner
- 3 conversation extracts (inverse CD, wheel taxonomy, sedenion ZD)
- 12 conversation files removed (convos/ fully processed)
- 10 new claims (C-458..C-467), Insight I-015

## Sprint 8 Summary (2026-02-07)

- CdMultTable generator, Hartigan dip test, Casimir DE, PEPS entropy
- mass-clumping binary, EXPERIMENTS_PORTFOLIO_SHORTLIST.md, MATH_CONVENTIONS.md
- Cubic Anomaly (I-012), External data cross-validation (I-014)
- License: GPL-2.0-only. 1670 tests.

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

---

## Open Items (Forward-Looking)

### High Priority

1) **Multiplication coupling rho(b) in GL(8,Z)** (C-466, Sprint 9 open question)
   - The scalar shadow pi(b) = signum(sum) is verified for addition mode.
   - The full multiplication coupling rho(b) acting on lattice vectors remains open.
   - This is the key missing piece for a complete lattice-algebra dictionary.

2) **Primary-source citation for every claim** (ROADMAP 7.2)
   - Systematic sweep through CLAIMS_EVIDENCE_MATRIX.md.
   - Add missing WHERE STATED and bibliographic references.
   - Sprint 8 did 10 claims; ~200+ remain without primary sources.

3) **Paper-ready LaTeX pipeline** (ROADMAP 7.5)
   - `make latex` build from verified results.
   - Structured "hypotheses + tests + results" format.
   - Priority target: ultrametric core mining (I-013) or lattice codebook (I-015).

### Medium Priority

4) **GF(2) separating degree at dim=64** (I-012 open question)
   - At dim=32, the 8/7 motif split needs a cubic GF(2) polynomial.
   - At dim=64 (4 motif classes, PG(4,2)), minimum separating degree is unknown.
   - Does the degree grow with the doubling level?

5) **Extend parity-clique and XOR partner to dim=128+** (Sprint 9 open)
   - Parity-clique verified at dims 16, 32. XOR partner verified at dim=64.
   - dim=256 ZD adjacency is computationally infeasible (132M evaluations).
   - dim=128 may be tractable with optimized code paths.

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
