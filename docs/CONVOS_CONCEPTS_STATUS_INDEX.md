# Convos -> Concepts -> Status -> What to Implement Next

This document distills convo-derived mathematical ideas into a reproducible engineering backlog.

Non-negotiable rule: text in `convos/` is not treated as authoritative. Every claim becomes a hypothesis
that must be tied to (a) first-party sources, and (b) a repo test/artifact that can be reproduced under
`PYTHONWARNINGS=error`.

Convos policy: files under `convos/` are treated as immutable inputs (they may contain Unicode).
All repo-authored synthesis docs/code should remain ASCII-only.

For standardized terminology used across docs/code/tests, see `docs/TERMINOLOGY_GLOSSARY.md`.

Convos chunk audit logs live under `docs/convos/` (start with `docs/convos/audit_1_read_nonuser_lines_cont.md`).

## Status legend
- Implemented + tested: backed by unit tests and/or artifact verifiers.
- Implemented (untested): code exists but lacks tests or invariants.
- Prototype only: exploratory script/notebook-style code; not integrated into build system.
- Unimplemented: only described in docs/convos.
- Rejected / reframed: claim is false as stated; replaced with a correct, testable statement.

## Index (high-level)

| Concept ID | Standard terms | Convo pointers | Repo pointers | Status | What to implement next (minimal next step) |
|---|---|---|---|---|---|
| CX-001 | A_infty-algebra, Stasheff identities, Homological Perturbation Lemma (HPL), Kadeishvili transfer, m3 (ternary operation) | `convos/3_further_explorations.md:26964` | `src/gemini_physics/m3_cd_transfer.py`, `tests/test_m3_cd_transfer.py`, `src/export_m3_table.py` | Implemented + tested | Add artifact export; then decide whether we can legitimately claim an A_infty structure (needs a precise dg source + Stasheff identity checks). |
| CX-002 | Cayley-Dickson zero divisors; 2-blades (ei+/-ej); 4-blades; motif census (cuboctahedron / K_{2,2,2}) | `convos/4_read_every_nonuser_line.md:8960` | `src/gemini_physics/cd_motif_census.py`, `tests/test_cd_motif_census.py`, `src/export_cd_motif_census.py`, `docs/MOTIF_CENSUS_SCHEMA.md` | Implemented + tested (16D/32D snapshots) | Extend to 64D/128D (exact) and 256D (sampled) with deterministic policies; add a single plot of component-size distribution vs dimension. |
| CX-003 | XOR-bucket heuristic, XOR-balanced quadruples, parity signings | `convos/4_read_every_nonuser_line.md:8968` | `src/gemini_physics/cd_xor_heuristics.py`, `tests/test_cd_xor_heuristics.py` | Implemented + tested | Extend from 2-blades->2-blades to 2-blades->selected 4-blades; keep the claim "necessary, not sufficient". |
| CX-004 | Fractional Laplacian (Riesz vs spectral), Caffarelli-Silvestre extension, fractional Schrodinger | `convos/3_further_explorations.md:22254` | `src/gemini_physics/fractional_laplacian.py`, `tests/test_fractional_laplacian.py`, `docs/FRACTIONAL_LAPLACIAN_OPERATOR_CHOICES.md`, `src/quantum/fractional_schrodinger.py`, `src/nonlocal_long_range_ZZ.py` | Implemented + tested (foundations) | Add a minimal extension (Caffarelli-Silvestre / Stinga-Torrea) toy solver and a cross-formulation consistency test on a tiny grid. |
| CX-005 | Negative dimension as analytic continuation / dimensional regularization, Parisi-Sourlas | `convos/2_exploration_cont.md:921` | `docs/NEGATIVE_DIMENSION_CLARIFICATIONS.md`, `docs/BIBLIOGRAPHY.md`, `src/neg_dim_pde.py`, `src/neg_dim_spectrum.py` | Clarified / reframed | Audit docs for any remaining "negative dimension" overclaims and route them through the clarification policy. |
| CX-006 | p-adic analysis, dyadic rationals, Cantor-set / devil's staircase, Vladimirov operator | `convos/3_further_explorations.md:3112` | `src/gemini_physics/padic.py`, `tests/test_padic_and_cantor.py`, `docs/BIBLIOGRAPHY.md`, `docs/latex/MASTER_SYNTHESIS.tex` | Implemented + tested (foundations) | Add a Vladimirov/Kozyrev wavelet "hello world" (defer until scope + domain are fixed). |
| CX-007 | Nilpotent orbits, Jordan form, Bala-Carter, Albert algebra H3(O), Freudenthal determinant | `convos/2_exploration_cont.md:710` | `src/gemini_physics/nilpotent_orbits.py`, `tests/test_nilpotent_orbits.py`, `docs/SEDENION_NILPOTENCY.md` | Implemented + tested (toy) | Add (1) a sourced note on Bala-Carter usage for exceptional groups, and (2) a separately-scoped Albert/Freudenthal module with unit tests. |
| CX-008 | Harper-Hofstadter model, Chern numbers, TKNN integers, Fukui-Hatsugai-Suzuki (FHS), Diophantine gap labeling, Streda formula | `convos/1_read_each_nonuser_line.md:5141` | `src/quantum/harper_chern.py`, `src/quantum/advanced/projector_fhs_polar.py`, `data/artifacts/images/harper_chern_map_highres.png` | Prototype only | Refactor to a small library module + add tests for the Diophantine congruence on a tiny grid (q <= 9) under warnings-as-errors. |
| CX-009 | Kramers-Kronig relations, Hilbert transform, (M)SSKK, Cole-Cole, Havriliak-Negami, causality/passivity | `convos/1_read_each_nonuser_line.md:6146` | `src/quantum/advanced/kk_mskk2.py`, `data/artifacts/images/kk_mskk2_bars_fixed_3160x2820.png` | Prototype only | Extract a tested KK kernel module; add a synthetic-model regression test that quantifies finite-window bias vs tail-extension. |
| CX-010 | Surreal numbers (Conway), birthday, simplicity theorem (scope-warning) | `convos/1_read_each_nonuser_line.md:30` | (none) | Unimplemented | Add a short, sourced doc note: surreals are a valid number system but not a Cayley-Dickson precursor; any "pre-Cayley" usage must be labeled metaphor/speculation. |
| CX-011 | Dimensional continuation (analytic continuation in dimension; Gamma/Beta identities), sphere volumes | `convos/2_exploration_cont.md:4982` | `src/gemini_physics/dimensional_geometry.py`, `tests/test_dimensional_geometry.py`, `src/vis_dimensional_geometry.py` | Implemented + tested | Extend plots with pole annotations (optional) and tie any "negative D" narrative strictly to dim-reg/analytic-continuation policy. |
| CX-019 | Reproducible scientific visualization; Matplotlib mathtext constraints; warnings-as-errors plotting | `convos/1_read_nonuser_lines_cont.md:25636`, `convos/1_read_nonuser_lines_cont.md:26462`, `convos/1_read_nonuser_lines_cont.md:26467`, `convos/1_read_nonuser_lines_cont.md:29605`, `convos/1_read_nonuser_lines_cont.md:30324` | `src/quantum/advanced/pseudospectrum_slice.py`, `src/quantum/advanced/fractal_css_expanded.py` | Unimplemented | Add a short repo policy note + helper patterns: no divide-by-zero in demo grids, no Matplotlib deprecations/warnings under `PYTHONWARNINGS=error`, avoid unsupported MathText (`\\text{...}`), keep OpenAI image tooling optional and out of `make check`. |
| CX-020 | Artifact atlas / "plates" as reproducible scripts (codex layout) | `convos/1_read_nonuser_lines_cont.md:30406` | `src/verification/verify_generated_artifacts.py`, `src/export_cd_motif_census.py`, `src/export_de_marrais_boxkites.py`, `src/export_reggiani_annihilator_stats.py` | Unimplemented | If we adopt "plates": define each plate as a deterministic script that writes versioned artifacts under `data/artifacts/` and add a single `make atlas` target to regenerate and validate them. |
| CX-012 | E8 root system, root lattice, quasicrystal projection (cut-and-project, 2D/3D projections) | `convos/2_exploration_cont.md:6987` | `src/gemini_physics/e8_lattice.py`, `src/e6_refinement.py`, `docs/LACUNAE.md` | Prototype only | Refactor to an import-safe `e8_roots` module and add tests: 240 roots, correct norms, inner-product multiset, and invariance under sign/permutation generators. |
| CX-013 | Associator statistics, bit-bucket (popcount) stratification, associator heatmap/tensor field | `convos/2_exploration_cont.md:9020` | `src/gemini_physics/optimized_algebra.py`, `src/sedenion_field_sim.py` | Prototype only | Add a deterministic associator map export (16D/32D): bucket triple indices by popcount, compute mean associator norm, write CSV + plot, add a regression test on output shape/ranges. |
| CX-014 | Spectral triple, noncommutative geometry, Connes distance, Dixmier trace, zeta-function regularization | `convos/3_further_explorations.md:399` | (none) | Unimplemented (scope pending) | Add a scope doc that separates (a) toy commutative spectral triple sanity checks vs (b) any claims about "surreal"/CD algebras; implement only (a) first. |
| CX-015 | Geometry over F1 (Tits, Soule, Connes-Consani, Lorscheid blueprints) | `convos/1_read_each_nonuser_line.md:9473` | (none) | Rejected / reframed | Add a sourced note: F1 is not a field; any use is philosophical scaffolding only, not a computational module claim. |
| CX-016 | Sectorial operators, semigroups, resolvent bounds, pseudospectrum | `convos/1_read_each_nonuser_line.md:6000` | `src/quantum/advanced/pseudospectrum_slice.py`, `data/artifacts/images/pseudospectrum_slice_3160x2820.png` | Prototype only | Extract a small, tested linear-operator discretization (periodic 1D) and validate basic pseudospectrum monotonicity under parameter sweeps. |
| CX-017 | wheels (division-by-zero algebra, Carlstrom); wheel graph (graph theory); wheeled operads/PROPs (Markl/Merkulov/Shadrin) | `convos/1_read_nonuser_lines_cont.md:L5` | `docs/WHEELS_DIVISION_BY_ZERO.md`, `crates/algebra_core/src/wheels.rs` | Implemented + tested | Carlstrom WheelQ with all 8 axioms verified on canonical test set. Source doc disambiguates wheel (Carlstrom) vs wheel graph vs wheeled operad. Scope boundary: wheels do not explain CD zero-divisors (labeled speculative unless sourced). |
| CX-018 | Algebraic integer rings (Gaussian/Eisenstein), Hurwitz quaternions, octonion integer lattices ("octavians") | `convos/1_read_nonuser_lines_cont.md:L60` | (none) | Unimplemented (docs-first) | Add a short, sourced doc note defining these rings/lattices and their norm forms; avoid any "negative dimension" framing. |
| CX-021 | RG flow spectral scaling; associator growth exponent; Callan-Symanzik analogy | `convos/3_further_explorations.md`, `convos/2_exploration_cont.md` | `src/gemini_physics/renormalization.py`, C-074 (`tests/test_claim_c074_associator_growth.py`) | Prototype only | Formalize the connection between associator growth exponent (C-074: alpha~1.80) and RG-like scaling; add a rigorous beta-function extraction or declare the analogy heuristic. |
| CX-022 | Triality, Spin(8), outer automorphisms, G2 as triality fixed-point subgroup | `convos/4_read_every_nonuser_line.md`, `convos/2_exploration_cont.md` | `src/gemini_physics/algebra/sedenion_automorphism.py`, C-028 (Verified), `curated/02_simulations_pde_quantum/Triality-Based_Quantum_Information_Structures.csv` | Implemented + tested (via C-028) | Extract triality-specific module from sedenion_automorphism.py; add explicit Spin(8) outer-automorphism verification and D4 Dynkin diagram symmetry test. |
| CX-023 | Fractal CSS codes, Sierpinski stabilizers, fracton topological order, GF(2) rank | `convos/1_read_nonuser_lines_cont.md`, `convos/3_further_explorations.md` | `src/quantum/fracton_code.py`, `src/quantum/advanced/fractal_css_expanded.py` | Prototype only | Refactor into tested module; add distance lower-bound computation; connect to Yoshida (2013) fracton framework with proper citations. |
| CX-024 | p-adic gauge deformations, Vladimirov operator on gauge fields, p-adic AdS/CFT | `convos/3_further_explorations.md` | CX-006 (`src/gemini_physics/padic.py`, `tests/test_padic_and_cantor.py`) | Unimplemented (scope pending) | Requires CX-006 foundations; scope a minimal gauge-deformation toy model or defer with explicit bibliography (Gubser et al. 2017, Heydeman et al. 2018). |
| CX-025 | Inverse Cayley-Dickson (un-doubling / contraction), CD contraction as A-infinity co-operations | `convos/3_further_explorations.md`, `convos/1_read_each_nonuser_line.md` | CX-001 (`src/gemini_physics/m3_cd_transfer.py`, `tests/test_m3_cd_transfer.py`) | Unimplemented (theory-first) | Formalize the "inverse CD" map p: CD(A)->A as an A-infinity co-operation; verify whether m3 from CX-001 arises as a transferred structure from this contraction. Cite Kadeishvili (2005). |
| CX-026 | Quantum inequalities (Ford-Roman-Pfenning), negative energy constraints, warp drive obstructions | `convos/compass_artifact_wf-a59991f9...md`, `convos/compass_artifact_wf-7f987cff...md` | `crates/gr_core/src/constants.rs` (Planck units), C-011 (warp drive constraints) | Unimplemented (literature-first) | Cache Ford-Roman QI bound C=3/(32pi^2); verify tau^4 scaling; connect to existing C-011 warp-drive constraint chain. |
| CX-027 | Light-ion QGP (O-O, Ne-Ne), alpha-clustering nuclear geometry, v3 centrality inversion | `convos/compass_artifact_wf-f3798de2...md` | C-043 (O-O collisions July 2025) | Unimplemented (data pending) | Track CMS HIN-25-008 RAA=0.69, Ne-Ne RAA~0.60; add alpha-clustering geometry (tetrahedral O-16, bowling-pin Ne-20); monitor for ALICE/ATLAS publications through 2026. |
| CX-028 | Subluminal positive-energy warp drives (Bobrick-Martire, Fuchs 2024), Lentz refutation | `convos/compass_artifact_wf-a59991f9...md` | C-011 (warp drive framework), C-401 (White-Casimir warp) | Unimplemented (literature-first) | Cache Bobrick-Martire (2021) and Fuchs et al. (2024) papers; document distinction between subluminal-positive-energy and superluminal-exotic-matter regimes; note acceleration-mechanism absence as open problem. |

---

## Parked / speculative themes from `convos/` (not yet accepted into the backlog)

These appear throughout the convos but are currently out of scope unless/until we can turn them into
precise definitions + first-party citations + reproducible tests/artifacts.

- "Negative-dimensional categories", "fractal category theory", and "meltdown coherence diagrams" (needs a real categorical model and a minimal toy example).
- "Holography/AdS-CFT in negative-fractal spacetimes" (currently metaphorical; treat as speculation unless sourced and formalized).
- "p-adic automorphic forms / L-functions validation" (requires a concrete target object + a reproducible computational comparison).
- "Tensor network / MERA / VR synesthesia tooling" (engineering/UI ideas; not a math validation claim).

---

## CX-001 - m3 / A_infty validation

### What the convos claim (to be validated or corrected)
- There is a "tree-level HPL/transfer" trilinear operation `m3` derived from a CD(O)->S contraction.
- On distinct octonion basis triples (i,j,k with i,j,k in 1..7 all distinct), the outputs split into 42 scalar and 168 vector cases ("42 vs 168").
- The split is connected to Fano-plane combinatorics (and often rhetorically linked to `|PSL(2,7)| = 168`).
- The surrounding narrative sometimes conflates "associator" with "A_infty m3"; this requires careful correction.

### Repo reality (current)
- `src/gemini_physics/m3_cd_transfer.py` computes a specific trilinear operation via
  `p mu( h mu(i x, i y), i z ) - p mu( i x, h mu(i y, i z) )`
  using `i(x)=(x,x)`, `p(a,b)=(a+b)/2`, `h = Id - i p`.
- `tests/test_m3_cd_transfer.py` validates the 42/168 split and the "Fano-line <-> scalar" rule.
- We have not yet:
  - exposed this as a stable library API,
  - proved it satisfies any Stasheff identity set,
  - formally related it to Kadeishvili transfer from a dg associative algebra.

### Minimal reproducible hypotheses (tests)
- H1: For distinct triples (i,j,k) in 1..7, the computed `m3` is never zero.
- H2: Among those 210 triples, exactly 42 outputs are scalar (+/-2e0) and 168 are pure imaginary (+/-2e_l).
- H3: Scalar outputs correspond exactly to unordered Fano lines {i,j,k}.
- H4: For a fixed Fano line, the scalar sign flips with permutation parity.

### What to implement next
- Add a small CSV export (optional) + verifier hook (now unblocked).
- Decide how to treat "A_infty" claims:
  - either implement a precise dg source + transfer (and test Stasheff identities),
  - or reframe the repo language to avoid implying A_infty structure is established.

### First-party sources to cite (core)
- Stasheff (1963): "Homotopy associativity of H-spaces I/II" (definition of A_infty).
- Kadeishvili (1980; arXiv scan available): minimal model / transferred A_infty structure.
- Gugenheim-Lambe-Stasheff (late 80s/early 90s): perturbation theory / explicit transfer formulas.

---

## CX-002 - 32D/64D/128D/256D motif census (ZD graphs)

**Status:** Implemented + tested (motif census + scaling laws + terminology audit)

### What the convos claim
- Build graphs whose nodes are blades (2-blades, selected 4-blades), edges indicate exact zero-products.
- In 16D (sedenions), pure 2-blade ZD components resemble 12-vertex degree-4 motifs (cuboctahedra).
- Adding XOR-balanced 4-blades fuses motifs into ~18-vertex components; K_{2,2,2} (octahedra) did not appear in an all-plus 4-blade sweep.
- In 32D (pathions), XOR-bucketed 2-blade mining yields many clean cuboctahedra.
- Proposed extension: repeat for 64D/128D/256D (sometimes using "Chingons/Routons" naming) and track motif density vs dimension.

### Minimal reproducible hypotheses (tests)
- H1: VERIFIED. In 16D and 32D, nontrivial ZD edges among 2-blades exist. Tested via `motif_components_for_cross_assessors()`.
- H2: VERIFIED. At dim=16, all 7 components are octahedral (K_{2,2,2}). At dim=32+, cuboctahedra appear. Component structure changes completely at each doubling.
- H3: VERIFIED. XOR-bucket heuristics have zero false negatives (perfect recall). Precision quantified in CX-003: 168/315 = 53.3% at dim=16.

### Implementation artifacts
- `crates/algebra_core/src/boxkites.rs`: `cross_assessors()`, `diagonal_zero_products_exact()`, `MotifComponent`, `motif_components_for_cross_assessors()`.
- Exact census extended to dim=256 (16 regression tests in boxkites.rs).
- 5 verified scaling laws: n_components = dim/2-1, nodes_per_component = dim/2-2, n_motif_classes = dim/16, K2_components = 3+log2(dim), K2_parts = dim/4-1.
- NO octahedra beyond dim=16; structure completely restructures each doubling.

### Terminology audit (source-first pass, 2026-02-06)
- **Sedenion (dim=16) terms**: FULLY ALIGNED with de Marrais (2000, 2004). assessor, box-kite, strut, sail/vent, co-assessor, production rules 1/2/3, automorpheme, "Behind-the-8-Ball Theorem", edge sign types, O-trips, strut table -- all match primary sources.
- **Higher-dim naming**: Pathion/Chingon/Routon (32/64/128D) present in `hypercomplex.rs::AlgebraDim`. Dim=256 has no de Marrais name ("Voudouion" appears in some informal sources but has no first-party citation).
- **Placeholder Substructures series (2002-2008)**: Terms "emanation tables", "meta-fractals", "bit-string recipes" describe infinite-dimensional organizational patterns NOT YET implemented. Our finite census correctly uses neutral generic terms (`CrossPair`, `MotifComponent`).
- **Decision**: No terminology changes needed in existing code. The generic naming for dim>16 is appropriate since de Marrais's nomenclature is informal and non-standardized beyond sedenions.

### First-party sources cited
- de Marrais (2000) box-kites + production rules: replicated in boxkites.rs.
- de Marrais (2004) Box-Kites III: production rules 1/2/3, automorphemes, strut tables.
- de Marrais "Placeholder Substructures" series (2002-2008): emanation tables, Pathions, XOR-driven recipes. CITED in BIBLIOGRAPHY.md (entries 160-171). NOT yet implemented (infinite-dim patterns).

---

## CX-003 - XOR-balanced / parity / 4-blade constructions

### What the convos claim
- CD basis products obey `e_a e_b = +/- e_{a xor b}` (index XOR with sign).
- For sum-of-blades products to cancel to zero, XOR-targets must match in pairs; thus XOR alignment is a necessary condition.
- XOR-balanced 4-tuples (i xor j xor k xor l = 0) are a useful search space for ZD-rich 4-blades.
- Signed 4-blades with even parity are needed to realize certain K_{2,2,2} sign patterns.

### What to implement next
- Implement a small XOR lemma module + tests:
  - check necessary conditions statistically on random sparse blades,
  - avoid overclaiming sufficiency.

---

## CX-004 - Fractional Laplacian foundations

**Status:** Implemented + clarified (Rust spectral_core, source-first pass 2026-02-06)

### What the convos claim
- Interchangeably uses "fractional Laplacian" across (at least) three non-equivalent objects:
  - Riesz fractional Laplacian on R^d (Fourier symbol |xi|^alpha),
  - spectral fractional Laplacian on bounded domains (eigen-expansion),
  - Caffarelli-Silvestre extension (Dirichlet-to-Neumann map).
- Mentions "fractional diffusion" closures and fractional Schrodinger dynamics.

### Repo reality (updated 2026-02-06)
- `crates/spectral_core/src/lib.rs` implements TWO correctly-labeled formulations:
  - **Periodic (Riesz)**: `fractional_laplacian_periodic_{1d,2d,3d}` -- Fourier multiplier |k|^{2s} on torus.
  - **Dirichlet (spectral)**: `fractional_laplacian_dirichlet_{1d,2d}` -- DST-I eigenexpansion with lambda_k^s.
- `crates/spectral_core/src/neg_dim.rs` has `caffarelli_silvestre_eigenvalues()` which uses Fourier-symbol (Riesz) definition. Doc clarified: equivalent to C-S extension for s in (0,1) by their theorem, but does NOT implement the (d+1)-dimensional extension PDE.
- **Cross-validation**: `test_dirichlet_laplacian_1d_consistency` compares s=1 fractional Dirichlet vs direct finite difference. Passes.
- 11 tests verify eigenfunction expansion, roundtrip DST-I, consistency across formulations.

### Terminology clarification applied
- Function names correctly distinguish periodic vs Dirichlet domains.
- `caffarelli_silvestre_eigenvalues` doc now explicitly states it uses Fourier-symbol, not extension solver, and cites the equivalence theorem.
- NOT yet implemented: Caffarelli-Silvestre (d+1)-dimensional extension solver. This is a future enhancement, not a gap in the current formulations.

### First-party sources cited (all in BIBLIOGRAPHY.md)
- Caffarelli-Silvestre (2007): extension characterization. CITED.
- Laskin (2000/2002): fractional quantum mechanics / fractional Schrodinger. CITED.
- Di Nezza-Palatucci-Valdinoci (2012): fractional Sobolev spaces "hitchhiker's guide". CITED.
- Kwasnicki (2017): "Ten equivalent definitions..." (operator-choice taxonomy). CITED in lib.rs.
- Stinga-Torrea (2010): extension framework. CITED.

---

## CX-005 - "Negative dimension" (clarification-first)

### Required clarification
- "Negative dimension" is a standard term in very specific contexts:
  - analytic continuation in dimensional regularization,
  - Parisi-Sourlas "negative dimensions" in supersymmetric formulations of disorder.
- "Negative-dimensional number systems" (dyadics/surreals/p-adics assigned fractional/negative dimension) is not standard and must be labeled speculative unless grounded in a published definition.

### What to implement next
- Move all speculative "dimension assignments for number systems" behind an explicit "Speculation / Metaphor" boundary in docs.
- Keep only the rigorous dim-reg and Parisi-Sourlas material in "validated math" sections.

---

## CX-006 - p-adics / dyadics / Cantor sets

**Status:** Foundations implemented + tested (Rust padic.rs, source-first pass 2026-02-06)

### What to implement next
- Decide scope for ADVANCED features:
  - either implement Vladimirov operator / Kozyrev wavelets (requires Q_p function spaces),
  - or explicitly defer with bibliographic pointers (current choice).
- Foundations are COMPLETE: valuations, absolute values, dyadic rationals, Cantor digits, devil's staircase, p-adic distance, ultrametric check. 14 tests pass.

### First-party sources to cite (core)
- Vladimirov-Volovich-Zelenov (1994): p-adic analysis + mathematical physics.
- Taibleson (1975): Fourier analysis on local fields.

---

## CX-007 - Nilpotent orbits / Jordan forms / exceptional algebras

**Status:** Foundations implemented, no overclaims found (source-first pass 2026-02-06)

### What the convos claim
- Nilpotency claims based on "all eigenvalues are 0" for purported exceptional-algebra representation matrices.
- Mentions Bala-Carter classification and Jordan-form diagnostics.
- Mentions Albert algebra `H3(O)` and a Freudenthal determinant upgrade path.

### Repo reality (updated 2026-02-06)
- `nilpotent_orbits.rs`: Generic nilpotent matrix analysis (Jordan types, nilpotency index, partition enumeration, dominance order). Does NOT claim Bala-Carter or exceptional-algebra features. Correctly labeled "toy" -- works for arbitrary nilpotent matrices. 8 functions, tests pass.
- `e8_lattice.rs`: Freudenthal-Tits magic square fully implemented. All exceptional Lie algebras (F4, E6, E7, E8) derived correctly from division algebra pairs. Dimensions, ranks, root counts verified. No overclaims.
- NOT implemented (correctly absent): Bala-Carter classification, Albert algebra H3(O), Freudenthal determinant, Jordan product on exceptional algebras.
- No overclaims in code -- CX-007 "next actions" correctly identify H3(O) and Bala-Carter as FUTURE work.

### What to implement next (unchanged, deferred)
- Verified nilpotent pipeline: sl_n examples with Jordan form vs theory.
- H3(O) numeric model: elements, Jordan product, trace/quadratic/cubic invariants.
- Freudenthal determinant (later, depends on H3(O)).

### First-party sources cited (both in BIBLIOGRAPHY.md)
- Collingwood-McGovern (1993): nilpotent orbits reference text. CITED in nilpotent_orbits.rs.
- Springer-Veldkamp (2000): octonions, Jordan algebras, exceptional groups. CITED in BIBLIOGRAPHY.md.

---

## CX-021 - RG flow spectral scaling / associator growth exponent

### What the convos claim
- The CD associator growth law `<||A(a,b,c)||^2> = A_inf * (1 - C * d^{-alpha})` (C-074)
  exhibits a power-law approach to the asymptotic value A_inf=2.0 with exponent alpha~1.80.
- Several convo threads draw an analogy to RG flow: the dimension d plays the role of
  an inverse "energy scale", and the growth exponent alpha is loosely compared to a
  beta-function coefficient in a Callan-Symanzik equation.
- The claim is that associator statistics undergo a "dimensional phase transition" near
  dim=8 (where associativity breaks) and flow to a universal fixed point at large dim.

### Repo reality (current)
- `src/gemini_physics/renormalization.py` implements a toy RG flow simulation on a 2D
  scalar field (unrelated to CD algebras; uses gradient-based anomalous dimension).
- C-074 has a fitted growth law with A_inf~2.0, but UQ (uncertainty quantification)
  is pending and the "RG analogy" is not formalized.
- C-087 (Phase 6) independently confirmed E[||A||^2] -> 2.0 via Monte Carlo at dim=128.

### Minimal reproducible hypotheses (tests)
- H1: The exponent alpha from the C-074 fit is stable under resampling (bootstrap CI).
- H2: A discrete "beta function" b(d) = <||A||^2>(2d) - <||A||^2>(d) is monotonically
  decreasing and approaches 0 (fixed-point behavior).
- H3: The analogy to Callan-Symanzik is either (a) formalized via a precise operator
  mapping or (b) explicitly labeled as heuristic/metaphorical.

### What to implement next
- Add bootstrap CI to C-074 fit parameters (A_inf, C, alpha).
- Compute the discrete beta function b(d) and test monotonicity.
- Write a scope note: if no rigorous operator mapping exists, relabel the "RG flow"
  language as metaphorical in all docs.

### First-party sources to cite (core)
- Baez (2002): "The Octonions" (associator properties at each Cayley-Dickson level).
- Wilson & Kogut (1974): RG and the epsilon expansion (for comparison of scaling forms).

---

## CX-022 - Triality / Spin(8) / outer automorphisms

### What the convos claim
- Spin(8) has a unique Z/3 outer automorphism group (triality), permuting the vector,
  left-spinor, and right-spinor representations (all 8-dimensional).
- G2 is the fixed-point subgroup of triality within SO(7) (or equivalently, Aut(O)).
- The S3 factor in Aut(S) = G2 x S3 (C-028) permutes three octonionic subalgebras
  of the sedenions, and this permutation is related to triality.
- Triality underpins the three-generation structure claimed in C-029.

### Repo reality (current)
- C-028 is **Verified**: `src/gemini_physics/algebra/sedenion_automorphism.py` confirms
  three closed octonionic subalgebras and S3 permutation validity.
- `curated/02_simulations_pde_quantum/Triality-Based_Quantum_Information_Structures.csv`
  contains a curated dataset (from Gemini sessions) on triality-based quantum structures.
- No standalone triality module exists; the verification is embedded in sedenion code.

### Minimal reproducible hypotheses (tests)
- H1: The three 8D representations of Spin(8) are verified to be permuted by the
  outer automorphism (explicit matrix construction).
- H2: The fixed-point subalgebra of the triality action on O is exactly G2 (14-dimensional).
- H3: The S3 in C-028 acts on the three O-subalgebras consistently with the triality
  permutation of Spin(8) representations.

### What to implement next
- Extract a small `triality.py` module with explicit Spin(8) outer automorphism matrices.
- Add tests: (a) D4 Dynkin diagram has S3 symmetry, (b) triality permutes three 8D
  irreps, (c) G2 is the stabilizer of the triality action.
- Cross-reference with C-028 and C-029.

### First-party sources to cite (core)
- Baez (2002): "The Octonions", Sec. 3.4 (triality and Spin(8)).
- Adams (1996): "Lectures on Exceptional Lie Groups", Ch. 7 (triality).
- Yokota (2009): "Exceptional Lie Groups" (explicit triality construction).

---

## CX-023 - Fractal CSS codes / Sierpinski stabilizers

### What the convos claim
- Fractal patterns (specifically the Sierpinski triangle / Rule 90 cellular automaton)
  can be used as stabilizer supports for CSS quantum error-correcting codes.
- The resulting "fractal codes" have interesting distance scaling properties related to
  the Hausdorff dimension of the fractal.
- Connections to fracton topological order (Yoshida 2013, Haah 2011): certain fractal
  stabilizer codes realize type-I fracton phases where excitations have restricted mobility.

### Repo reality (current)
- `src/quantum/fracton_code.py` (43 lines): Rule-90 Sierpinski mask visualization only.
  No stabilizer algebra, no distance computation, no tests.
- `src/quantum/advanced/fractal_css_expanded.py` (114 lines): Builds 12 X-stabilizers
  and 12 Z-stabilizers from Sierpinski translations, computes GF(2) ranks, searches for
  low-weight X-logicals. Prototype with visualization output; no tests.
- Neither script is imported by any test or library module.

### Minimal reproducible hypotheses (tests)
- H1: The Sierpinski CSS code with L=33 has k >= 1 logical qubit (rank deficiency).
- H2: The minimum distance d scales as O(L^alpha) with alpha related to the fractal
  dimension log(3)/log(2) ~ 1.585.
- H3: The X- and Z-stabilizers satisfy the CSS commutativity condition (H_X * H_Z^T = 0
  mod 2).

### What to implement next
- Refactor `fractal_css_expanded.py` into a tested library module under `src/quantum/`.
- Add tests for: CSS orthogonality, rank computation, and distance lower bound.
- Add bibliography entries for fracton codes.

### First-party sources to cite (core)
- Yoshida (2013): "Exotic topological order in fractal spin liquids", Phys. Rev. B 88.
- Haah (2011): "Local stabilizer codes in three dimensions without string logical
  operators", Phys. Rev. A 83.
- Bravyi & Haah (2013): "On the energy landscape of 3D spin Hamiltonians with topological order".

---

## CX-024 - p-adic gauge deformations

### What the convos claim
- p-adic numbers can parametrize gauge field deformations, with the Vladimirov operator
  (p-adic fractional derivative) playing the role of a kinetic term.
- "p-adic AdS/CFT" (Gubser et al.) provides a toy holographic duality where the
  Bruhat-Tits tree replaces the continuous AdS bulk.
- Convo threads suggest connecting p-adic analysis to CD algebra deformations: viewing
  the dyadic structure of Cayley-Dickson doubling as a 2-adic phenomenon.

### Repo reality (current)
- CX-006 provides tested p-adic foundations: `src/gemini_physics/padic.py` implements
  basic p-adic arithmetic and Cantor-set constructions, with `tests/test_padic_and_cantor.py`.
- No gauge-field or holographic modules exist.
- The "2-adic CD" connection is mentioned in convos but not formalized anywhere.

### Minimal reproducible hypotheses (tests)
- H1: The Vladimirov operator on the Bruhat-Tits tree (p=2) reproduces known eigenvalues
  (Kozyrev wavelets) on a finite truncation.
- H2: A toy p-adic gauge action S = integral |D_mu phi|_p^2 d_p(x) produces a propagator
  with the expected p-adic power-law decay.
- H3: The "2-adic CD" claim is either formalized (CD doubling = 2-adic extension) or
  explicitly labeled speculative.

### What to implement next
- Requires CX-006 Vladimirov wavelet foundations first.
- Scope decision: either (a) implement a minimal Bruhat-Tits tree + Vladimirov eigenvalue
  check, or (b) defer entirely and restrict to bibliography pointers.
- Add sourced notes distinguishing rigorous p-adic field theory from the speculative
  "2-adic CD" narrative.

### First-party sources to cite (core)
- Gubser, Knaute, Parikh, Samber, Witczak-Krempa (2017): "p-adic AdS/CFT", Comm. Math. Phys.
- Heydeman, Marcolli, Saberi, Stoica (2018): "Tensor networks, p-adic geometry...", arXiv:1812.06529.
- Vladimirov, Volovich, Zelenov (1994): "p-adic Analysis and Mathematical Physics".

---

## CX-025 - Inverse Cayley-Dickson (un-doubling) as A-infinity co-operation

### What the convos claim
- The Cayley-Dickson doubling map i: A -> CD(A) sending x -> (x, 0) has a natural
  left inverse p: CD(A) -> A given by p(a, b) = a (projection to the first half).
- The composition p . mu . (i x i) recovers the original multiplication on A, but
  when applied asymmetrically (e.g., p . mu . (id x i) or similar), it produces
  "contraction operations" that interact with the A-infinity structure from CX-001.
- Specifically, the m3 operation from CX-001 can be viewed as arising from the failure
  of the CD projection to be a strict algebra map: the "error term" in
  p(i(x) * i(y)) - x*y is related to the homotopy transfer data.
- The inverse-CD / un-doubling viewpoint suggests a tower of higher operations
  m_n at each CD level, forming an A-infinity algebra.

### Repo reality (current)
- CX-001 implements m3 via `src/gemini_physics/m3_cd_transfer.py` using the specific
  formula `p mu(h mu(ix, iy), iz) - p mu(ix, h mu(iy, iz))` with h = Id - ip.
- The 42/168 split and Fano-line correspondence are tested.
- No explicit "inverse CD" or "un-doubling" module exists.
- No Stasheff identity checks have been performed on the m3 output.

### Minimal reproducible hypotheses (tests)
- H1: The map p: CD(A) -> A (projection) satisfies p(i(x)*i(y)) = x*y exactly
  (the "section-retraction" identity for the doubling).
- H2: The "contraction defect" p(a*b) - p(a)*p(b) for generic sedenion elements
  (a, b not in the image of i) is nonzero and related to the m3 homotopy.
- H3: Stasheff identity check: m2(m3(a,b,c), d) +/- m3(m2(a,b), c, d) +/- ... = 0
  (the A-infinity relation for m3).

### What to implement next
- Implement an explicit "un-doubling" module with p, i, h operators.
- Test H1 (section-retraction) and H2 (contraction defect characterization).
- Attempt the first Stasheff identity (the A-infinity relation involving m2 and m3).
- If Stasheff fails, document the obstruction and reframe CX-001 accordingly.

### First-party sources to cite (core)
- Kadeishvili (1980/2005): "On the homology theory of fibre spaces" / transferred A-infinity.
- Loday & Vallette (2012): "Algebraic Operads", Ch. 9 (A-infinity algebras and transfer).
- Markl (2006): "Transferring A-infinity structures" (explicit formulas).

---

## CX-026 - Quantum inequalities and negative energy constraints

**Status:** Papers downloaded (2026-02-07)

### What the convos claim
- Ford-Roman quantum inequality bound: integral of <T_00> f(tau)^2 dtau >= -C/tau^4
  with C = 3/(32 pi^2) ~ 0.00949 for 4D massless scalar (Pfenning & Ford 1997).
- This implies minimum bubble wall thickness ~ 100 Planck lengths for warp drives.
- Negative energy requirement for Alcubierre metric: ~ 10^64 kg for 100m bubble at 10c.
- Van Den Broeck (1999) topology modification reduces to few solar masses.
- No quantum inequalities exist along null geodesics in 4D (Fewster & Roman 2003).

### Minimal reproducible hypotheses (tests)
- H1: Verify C = 3/(32 pi^2) numerically and confirm dimensional consistency.
- H2: Derive bubble thickness lower bound from QI and verify ~ 100 l_Planck.
- H3: Reproduce Alcubierre energy estimate for standard parameters.

### Downloaded papers (local, gitignored)
- `papers/arxiv_gr-qc9702026_pfenning_ford_1997_quantum_inequality_restrictions.pdf`
- `papers/arxiv_1207.5805_fewster_2012_lectures_quantum_energy_inequalities.pdf`
- `papers/arxiv_1904.05684_kontou_sanders_2020_quantum_energy_inequalities_review.pdf`
- `papers/arxiv_gr-qc0209036_fewster_roman_2003_null_energy_condition.pdf`

### First-party sources to cite
- Pfenning & Ford (1997): "Quantum inequalities on the energy density in static Robertson-Walker spacetimes".
- Fewster (2012): "Lectures on quantum energy inequalities".
- Kontou & Sanders (2020): Review, CQG 37, 193001.
- Fewster & Roman (2003): null geodesic counterexample.

---

## CX-027 - Light-ion QGP and alpha-clustering

**Status:** Papers and data downloaded (2026-02-07)

### What the convos claim
- O-O collisions at LHC (July 2025, sqrt(s_NN) = 5.36 TeV): CMS HIN-25-008
  measures R_AA = 0.69 +/- 0.04 at pT ~ 6 GeV (>5 sigma significance).
- Ne-Ne collisions: R_AA ~ 0.60 at pT ~ 6 GeV (7 sigma significance).
- O-16 has tetrahedral alpha-clustering: generates enhanced triangular
  eccentricity epsilon_3 -> v_3 (centrality inversion vs Pb-Pb).
- Ne-20 has prolate bowling-pin shape: beta_2 ~ 0.5-0.7 deformation.
- Energy loss scales ~ L (near-linear) in expanding QGP, not L^2
  static medium (Arleo & Falmagne, arXiv:2411.13258).
- Pre-equilibrium quenching onset at ~ 0.2 fm/c (Pablos & Takacs).

### Connection to existing claims
- C-043: O-O collisions at LHC (already tracked, needs R_AA update).
- New claims: Ne-Ne R_AA, v_3 inversion, energy-loss scaling, pre-equilibrium onset.

### Downloaded papers (local, gitignored)
- `papers/arxiv_2510.09864_cms_2025_discovery_suppressed_oo_collisions.pdf` (CMS discovery)
- `papers/arxiv_2509.07008_mazeliauskas_2025_energy_loss_baseline_oo.pdf` (no-quenching baseline)
- `papers/arxiv_2411.13258_arleo_falmagne_2024_nene_predictions.pdf` (Ne-Ne predictions)
- `papers/arxiv_2509.19430_pablos_takacs_2025_alpha_clustering_geometry.pdf` (pre-equilibrium)

### Available datasets (tracked in git)
- `data/external/cms_oo_raa/table3_oo_raa.csv`: CMS O-O R_AA, 44 pT bins (3-103.6 GeV)
  - Min R_AA = 0.695 at pT = 6.2 GeV, recovers to ~0.97 at pT > 80 GeV
  - HEPData DOI: 10.17182/hepdata.165512.v1/t3
- `data/external/cms_oo_raa/table1_pp_spectra.csv`: pp reference spectra
- `data/external/cms_oo_raa/table2_oo_spectra.csv`: OO inclusive spectra
- `data/external/cms_oo_raa/alice_pbpb_raa_0to5pct_baseline.csv`: ALICE Pb-Pb 0-5% R_AA baseline

### First-party sources to cite
- CMS (arXiv:2510.09864): Discovery of O-O jet quenching, HEPData ins3068407.
- Mazeliauskas (arXiv:2509.07008): No-quenching baseline for O-O.
- Arleo & Falmagne (arXiv:2411.13258): energy-loss scaling analysis.
- Pablos & Takacs (arXiv:2509.19430): pre-equilibrium timing via Bayesian inference.

---

## CX-028 - Subluminal positive-energy warp drives

**Status:** Papers downloaded (2026-02-07)

### What the convos claim
- Bobrick & Martire (2021): subluminal positive-energy warp drives are
  constructible but reduce negative energy by only ~2 orders of magnitude
  and lack any acceleration mechanism (passengers arrive slower than light).
- Erik Lentz (2021): claimed hyper-fast soliton with positive energy.
  REFUTED by Santiago, Schuster & Visser (2021) error analysis.
- Fuchs, Helmerich et al. (2024): achieve subluminal constant-velocity
  solution satisfying all classical energy conditions. Most rigorous
  positive-energy result to date, but still subluminal and constant-velocity.
- Finazzi, Liberati & Barcelo (2009): superluminal warp bubbles are
  unstable against quantum backreaction (Hawking-like instability).

### Connection to existing claims
- C-011: Alcubierre warp drive framework (exotic matter requirements).
- C-401: White-Casimir warp conjecture (PARTIAL, needs landscape update).

### Key distinction
The field splits into two regimes:
- **Subluminal + positive energy**: mathematically valid but physically trivial
  (slower than light, no acceleration). Bobrick, Fuchs results live here.
- **Superluminal + exotic matter**: requires NEC violation, quantum-scale
  constraints from Ford-Roman QIs (CX-026), and suffers Finazzi instability.
No bridge between regimes is known.

### Downloaded papers (local, gitignored)
- `papers/arxiv_2105.03079_santiago_schuster_visser_2021_generic_warp_drives.pdf` (Lentz refutation)
- `papers/arxiv_2407.18908_fuchs_2024_constant_velocity_subluminal_warp.pdf` (positive-energy)
- `papers/arxiv_0904.0141_finazzi_liberati_barcelo_2009_semiclassical_instability.pdf` (instability)
- Already in collection: Bobrick & Martire (2021), Lentz (2021), Alcubierre (2000), Smolyaninov (2010)

### First-party sources to cite
- Bobrick & Martire (2021): "Introducing physical warp drives", CQG 38, 105009.
- Lentz (2021): "Breaking the warp barrier" (CQG 38, 075015).
- Santiago, Schuster & Visser (2021): Lentz error analysis (PRD 105, 064038).
- Fuchs et al. (2024): positive-energy constant-velocity solution.
- Finazzi, Liberati & Barcelo (2009): instability result, PRD 79, 124017.

---

## Convos audit summary (2026-02-07)

Full audit of 7 files in `convos/`. Results:

- **104 distinct claims** identified across 6 compass artifacts + 1 large convo.
- **56 already in CLAIMS_EVIDENCE_MATRIX.md** (54%).
- **18 already in CX-index** (17%, CX-001 through CX-025).
- **30 new claims** extracted (29%), clustered into CX-026..CX-028 + individual items.
- **5 refutations** identified:
  1. Winding number formula ill-defined on discrete 5-site graphs.
  2. Lentz positive-energy hyper-fast soliton (error found by Santiago et al.).
  3. G2 holonomy + sedenion extension: no literature connection found.
  4. Debye screening lambda_D quoted as 0.2 fm; correct value 0.3-0.5 fm at T~200 MeV.
  5. Neutron star core temperature overstated (correct: ~10^9-10^7 K, not 0-50 MeV).

### Individual new claims not elevated to CX-level (for matrix tracking)

| ID | Domain | Summary |
|----|--------|---------|
| NEW-004 | Cosmology | Sedenion-based cosmological constant suppression (novel, no precedent) |
| NEW-005 | Cosmology | DESI Year 3 evolving dark energy at 2.8-4.2 sigma (not 5 sigma) |
| NEW-013 | Quantum gravity | Lauscher-Reuter eta_N = -2 yields d_S = 2 UV (asymptotic safety) |
| NEW-014 | Analog gravity | Steinhauer BEC Hawking radiation ~10% temperature agreement |
| NEW-015 | Information theory | Bekenstein bound S <= 2 pi R E / hbar c verified (Casini 2008) |
| NEW-016 | Optics | Spaceplates R = 176 +/- 14 compression (Hogan et al. 2025) |
| NEW-017 | Quantum gravity | d_S -> 2 universal across 6 independent QG frameworks (synthesis) |
| NEW-020 | QCD | Clarke et al. T_CEP = 102 MeV, mu_B_CEP = 428 MeV (published PRD) |
| NEW-026 | History | Pentium FDIV: 16 missing entries, 5 cause errors (nuanced) |

These should be promoted to C-nnn entries in the matrix when they acquire
primary source caching and offline validation tests.
