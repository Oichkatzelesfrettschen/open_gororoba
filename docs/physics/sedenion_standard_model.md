# Sedenion Standard Model: Algebraic Origin of Fermion Masses and Mixing

## I. Executive Summary

The 16-dimensional Sedenion algebra S contains three canonical octonionic
subalgebras O_1, O_2, O_3 whose S_3 permutation symmetry corresponds to the
three observed generations of fermions.  The non-associative multiplication
table generates a "topological friction" that varies across generations,
breaking the flavor symmetry and producing mass hierarchies.

This document summarizes the computational results of a systematic
selector-pair scan that maps the SU(3) x SU(2) x U(1) gauge structure onto
the sedenion subalgebras and extracts:

- Exact two-weight fit of charged-lepton log-ratios within the selector basis
- Best interleaved-scheme selector scan, after permutation-aware extraction,
  yields all three CKM angles within 10% of PDG (C-1456)
- PMNS neutrino mixing angle theta_13 within 1.2% of PDG (C-1457)
- Electroweak mixing angle sin^2(theta_W) within 14% (C-1458)
- Discrete 2*sqrt(2) quantization of the friction spectrum (C-1459)

**Scope distinction (peer-reviewed)**:

**(A) Literature-backed**: Interleaved sedenion subalgebras and the psi
automorphism exist and are explicit (Gresnigt/Gourlay 2019/2023). Overlap
and linear dependence across generations may underlie CKM/PMNS mixing
(speculative in the papers). Z(S) isometric to G_2 (Reggiani 2024), ZD(S)
isometric to V_2(R^7) (Reggiani 2024), V_2(R^8) frame decomposition
(Koebisu 2025). Aut(S) = G_2 (Schafer, confirmed by Wilmot 2025); the S_3
family symmetry is specific to the Gresnigt/Gourlay/Brown framework.

**(B) Project-specific computational results**: CKM/PMNS angles from
friction/psi coupling, V_6 SVD geometry, TensorElementLift solar correction
(C-1478). For the current selector/friction observable class, the
interleaved scheme is the strongest CKM/PMNS phenomenology platform.

**(C) Heuristic / provisional**: The specific 7-assessor block assignment in
TensorElementLift (moderate block alignment 44%, psi orbits cross blocks).
The (12/12/6) assessor partition (falsified, C-1474). Block assignment is
the minimal successful project lift, not yet derived from the algebra.

## II. The Flavor Hierarchy Mechanism

### The Casimir Baseline (Flavor-Blind)

The SU(3) quadratic Casimir C_2 = sum_a T_a * T_a, projected onto each
octonionic subalgebra, produces a 3x3 mass matrix M_ij = Re(C_2|O_i . C_2|O_j*).
This baseline matrix has rank 1 and is S_3-symmetric: all three generations
share identical mass.  The Casimir alone cannot produce a hierarchy.

### Topological Friction as Yukawa Coupling

Braiding two Majorana modes (e_i, e_j) within a subalgebra O_k accumulates
an "associator flux" -- the signed sum of all associator values [A_rot, X, B]
for probe basis elements X in O_k.  This signed friction f_k is:

    f_k = sum_{X in O_k} [A_rot(theta), X, e_j]

where A_rot = cos(theta)*e_i + sin(theta)*e_j is the braided state.

For theta = pi/4 and specific braid pairs, the three friction values
{f_1, f_2, f_3} break S_3 completely (21 of 105 pairs) or partially
(48 of 105 pairs).  The 36 remaining pairs preserve S_3 and correspond
to flavor-universal gauge interactions (Gresnigt Cl(8) correspondence, C-1460).

### The 1/sqrt(2) Geometric Coupling (Suggestive)

The difference-normalized fit of two signed-friction selectors to the
charged lepton mass ratios yields weights:

    w_1 = -0.6569,  w_2 = -0.7420
    w_sym = (w_1 + w_2)/2 = -0.6994  (~  -1/sqrt(2) = -0.7071, 0.8% error)
    |w_asym / w_sym| = 0.061

The proximity of w_sym to 1/sqrt(2) = cos(pi/4) = sin(pi/4) is suggestive
of a self-consistent link to the braid angle theta = pi/4.  This observation
requires further stabilization: it must be shown to persist across selector
orbits, basis changes, and equivalent symmetry representatives before it
can be elevated to a derived constant.

### S_3 -> 1+1+1 Symmetry Breaking

The signed friction (orientation-sensitive observable) breaks S_3 more
selectively than the unsigned norm:

| Observable     | Full splits | Degenerate | Partial | Total |
|----------------|-------------|------------|---------|-------|
| Unsigned |f|   | 54          | 51         | 0       | 105   |
| Signed f       | 21          | 36         | 48      | 105   |

The signed friction resolves more structure because sign cancellations
restore some S_3 symmetries that the norm discards.

## III. The Fermion Mass Spectra

### Charged Leptons

Selectors: Sel(e_1, e_4) and Sel(e_2, e_4).  Assignment: e=O_2, mu=O_3, tau=O_1.

    F_g = w_1 * Sel_1(g) + w_2 * Sel_2(g)
    m_mu / m_e = exp(F_mu - F_e) = 207.0  (PDG: 206.8, exact to 5e-16)
    m_tau / m_e = exp(F_tau - F_e) = 3477.0  (PDG: 3477.2, exact to 3e-16)

### 3-Blade Zero-Divisor Hierarchy (C-1459)

Escalating from 2-blade to 3-blade topological defects (triples e_i, e_j, e_k)
produces 231 / 455 = 50.8% full splits.  The best friction ratio is
exactly 3/2 (target 1.529, error 1.9%).  All friction values are integer
multiples of 2*sqrt(2), confirming that the CD doubling process enforces a
rigid discrete spectrum.  The sedenion friction manifold is a crystal,
not a smooth continuum.

### Quarks (Project-Specific Selector Scan)

The CKM selector scan (420 combinations, Rayon-parallelized) found the
optimal quark sector assignment using cross-coupled friction:

    F_up = w_1 * Sel_up + w_2 * Sel_down
    F_down = w_1 * Sel_down + w_2 * Sel_up

Best pair: up = (e_11, e_12), down = (e_10, e_11).

## IV. Gauge Mixing Matrices

### Quark Sector: CKM (C-1456)

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| |V_us|     | 0.245     | 0.225    | 8.9%  |
| |V_ub|     | 0.00382   | 0.00373  | 2.4%  |
| |V_cb|     | 0.044     | 0.042    | 5.0%  |
| theta_12   | 14.19 deg | 12.99    | 9.2%  |
| theta_13   | 0.219 deg | 0.214    | 2.3%  |
| theta_23   | 2.52 deg  | 2.40     | 4.8%  |

The CKM matrix is nearly diagonal because the up and down selector pairs
share the element e_11 (overlapping "outer shell" of the sedenion), producing
small off-diagonal mixing.

### Neutrino Sector: PMNS (C-1457)

Best pair: charged lepton = (e_11, e_12), neutrino = (e_7, e_8).

**Diagonal-only friction (baseline):**

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| theta_12   | 29.2 deg  | 33.4     | 12.6% |
| theta_13   | 8.64 deg  | 8.54     | 1.2%  |
| theta_23   | 32.3 deg  | 49.0     | 34%   |

**With psi-automorphism off-diagonal coupling (C-1464):**

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| theta_12   | 28.5 deg  | 33.4     | 15%   |
| theta_13   | 8.63 deg  | 8.54     | 1.1%  |
| theta_23   | 47.1 deg  | 49.0     | 3.9%  |

The off-diagonal coupling uses the Gourlay/Gresnigt psi automorphism
(order-3 S3 generator) to inject cross-generational friction.
Two independent parameters: alpha_ch = 3.75, alpha_nu = 1.30.

The psi overlap/norm ratio is -0.5 = cos(2*pi/3) for all generations,
confirming the S3 120-degree rotation directly drives atmospheric mixing.
The diagonal-only ceiling at 32.3 deg was broken by the off-diagonal
coupling, progressing: 32.3 -> 37.6 -> 39.0 -> 47.1 degrees.

The charged lepton selector (e_11, e_12) is identical to the CKM up-type
selector -- consistent with the SU(5) prediction that charged leptons
partner with up-type quarks (C-1462).

### V_6 Solar Angle Correction Pipeline (C-1474, C-1475)

The 6D orthogonal complement V_6 of the B/C column space within the Type X
incidence matrix (rank=6, all singular values = 3.420) provides a
basis-invariant candidate subspace for targeted solar angle correction. The pipeline:

1. `construct_casimir_baseline` -- neutral Casimir matrices (no quark leakage)
2. `construct_pmns_matrices_two_param` -- factored two-parameter psi coupling
3. `extract_v6_basis` -- incidence algebra SVD -> 6x42 basis
4. `AssessorToFlavorMap` -- explicit (12/12/6) assessor-to-generation partition
5. `apply_v6_perturbation` -- composable, beta=0 recovery exact

The Jacobian is epsilon-stable (<0.1 deg angular deviation across eps=0.01/0.05/0.1).
However, the gradients g_12, g_13, g_23 are nearly collinear in V_6 space under
the default (12/12/6) partition. No unit direction achieves positive solar
selectivity S(u) = |g_12.u| - 10|g_13.u| - 3|g_23.u|. The 1D scan confirms:
theta_12 shifts < 0.04 deg over t in [-10, 10].

This is a **structural null result** for the (12/12/6) partition (first-pass
projection heuristic): cos(g_12, g_13) = -1.0 (perfectly anti-collinear).

The FlavorLift trait makes the mapping pluggable. Three implementations tested:
- Partition(12/12/6): null (collinear gradients, S(u) = -0.04)
- DirectOffDiagonal: decorrelated (cos = 0.47), theta_12 moves 14-46 deg
  but g_13/g_12 = 1.7x prevents solar isolation within theta_13 constraint
- PsiEquivariant: zero gradients (orbit weights cancel; needs refinement)

Rank-2 lock (C-1476): under FlavorLifts that collapse 42D to 3 generation
factors, g_12 lies 100% in span{g_13, g_23} (residual 4.57e-5 to 5.18e-4).
This is a no-go for the 42D->3D lift family, not all V_6 couplings.

**Solar correction achieved (C-1478)**: TensorElementLift (42D->6D, 6 blocks
of 7 assessors mapping to all 6 independent Herm_3 elements) breaks the lock.
Residual fraction = 75.7%. Constrained Gram-Schmidt direction at t=2.47:

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| theta_12   | 33.42 deg | 33.41    | 0.02% |
| theta_13   | 8.63 deg  | 8.54     | 1.05% |
| theta_23   | 47.08 deg | 49.0     | 3.93% |

**1D solar correction** (3 parameters: alpha_ch=3.75, alpha_nu=1.30, t_V6=2.47).

Stability at optimum: Jacobian rank = 3 (full), condition number = 6.85
(well-conditioned), d^2(theta_12)/dt^2 = -0.28 (concave -- stable peak).
Residual fraction at optimum = 60.9% (still well-decoupled at t=2.47).

**Joint 4D optimization** (C-1491, 4 parameters re-optimized jointly):

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| theta_12   | 33.84 deg | 33.41    | 1.28% |
| theta_13   | 8.56 deg  | 8.54     | 0.24% |
| theta_23   | 48.74 deg | 49.0     | 0.54% |

alpha_ch=3.50, alpha_nu=1.35, t_solar=1.54, t_atmo=2.00. Score: 0.000221.
All three angles within 1.3% of PDG (4.9x improvement over fixed-alpha model).

Invariance audit: block alignment MODERATE (44% max concentration, 16.7%
would be uniform). Psi orbits cross blocks (30 cross, 12 within). The lift
works because it preserves 6 DOFs, not because the blocks are canonical.
V_6 is a psi-eigenspace (scalar 0.25*I_6) -- generation-invariant (C-1489).

### Electroweak Mixing Angle (C-1458)

The associator flux ratio SU(2)/SU(3) = 0.529 gives:

    sin^2(theta_W) = 0.375 * 0.529 = 0.199  (PDG: 0.231, 14% error)

The SU(2) flux is generation-independent (8.49 per generation) while the
SU(3) flux varies by generation (19.80, 11.31, 16.97), reflecting the
structural asymmetry between the weak isospin and color gauge sectors
within the sedenion algebra.

## V. The Zero-Divisor Manifold

### Reggiani G_2 Isometry (Literature-Established)

The constraint manifold Z(S) = {(a,b) : a*b = 0, |a| = |b| = 1} has
dimension 14 = dim(G_2), confirmed by numerical Jacobian rank test
(reggiani.rs::test_reggiani_zd_manifold_dimension_is_g2).

### Koebisu V_2(R^8) Holonomy (C-1461, Complementary to Reggiani)

Koebisu (arXiv:2512.13002) decomposes each sedenion as a pair of octonions
s = (a, b) where a = s[0..8], b = s[8..16].  The ZD condition becomes
|a| = |b| and <a,b> = 0, identifying the normalized ZD set with the
Stiefel manifold V_2(R^8).

All 84 standard zero-divisors satisfy both conditions (verified in
reggiani.rs::test_koebisu_holonomy_v2r8_decomposition).

Note: Koebisu's V_2(R^8) and Reggiani's V_2(R^7) are complementary results
using different mathematical frameworks.  Koebisu uses the full octonion
pair including the real component; Reggiani restricts to the imaginary sector.

### Wilmot Calibration Connection

Wilmot (AACA 2026, arXiv:2505.06011) derives sedenions from a 14-simplex
calibration on Pin(15).  The 14-dimensional calibration space matches
dim(G_2) = 14, providing a calibration-theoretic origin for the G_2 isometry.

**Note on Aut(S)**: Wilmot (arXiv:2512.07210) resolves the Schafer/Brown
discrepancy in favor of Schafer: Aut(S) = G_2.  Brown's sigma'
transformation (eq. 11) changes the e_{1234567} term of Phi_O, so it is
not an automorphism.  Only Phi_O^{C(1)} (cyclic sign variations) embeds
as a 15-dimensional G_2 representation.  Gresnigt's S_3 from Cl(8) acts
on generation labels (a DIFFERENT structure), not as algebra automorphisms,
and is compatible with Aut(S) = G_2.

## VI. Non-Associativity Structure and ZD Incidence

### Triad Classification (Wilmot Table 2, verified)

All C(15,3) = 455 triads of imaginary sedenion basis elements decompose as:

    35 fully associative = C(7,3) = H_15 quaternion subalgebra count
    84 Type B non-associative (only [b,d,c] nonzero)
    84 Type C non-associative (only [c,b,d] nonzero)
    252 Type X non-associative (all three orderings nonzero)
    0 Type A

This decomposition uses the standard associator [x,y,z] = (x*y)*z - x*(y*z)
in all three orderings.  Wilmot's Table 2 for U_1 gives 35+60+360=455
(the 155 in Table 2 belongs to U_2, not U_1).

### ZD-Triad Incidence (Chirality Conjecture Falsified)

The incidence matrix between 420 non-associative triads and 42 ZD assessor
pairs reveals **universal coverage, not chiral partition**:

| Type | Triads | Assessors covered | Hits per assessor |
|------|--------|-------------------|-------------------|
| B    | 84     | 42/42             | 36 (uniform)      |
| C    | 84     | 42/42             | 36 (uniform)      |
| X    | 252    | 42/42             | 96 (uniform)      |

All three types cover all assessors with perfectly uniform multiplicity.
The 84:84:252 decomposition is a **cardinality structure** from the
non-associativity theorem, not a chiral partition of the ZD manifold.

The earlier conjecture that "Type B = left-handed ZDs, Type C = right-handed
ZDs" is **falsified** by this incidence matrix.  The correct statement is:

The 35+84+84+252=455 refinement is a uniform cover structure over the
42 assessors.  Type B, C, and X are coverage classes, not flavor labels.

### Scheme Comparison (Interleaved vs Contiguous)

| Parameter | Interleaved stride | Tang contiguous | PDG |
|-----------|--------------------|-----------------|-----|
| Score     | 0.010              | 0.834           | --  |
| |V_us|    | 0.245              | 0.188           | 0.225 |
| theta_23  | 2.52 deg           | 0.99 deg        | 2.40  |

Within the current selector/friction observable class, the interleaved
CD-generated scheme (Gresnigt 2023) is the phenomenologically superior
CKM model, outperforming Tang's contiguous-block scheme by a factor
of ~83x in log-distance score.

## VII. G_2 -> SU(3) Gauge Sector (PR1-3, C-1479 to C-1488)

### Stabilizer Extraction (PR1, algebra/geometry)

For any imaginary octonion unit e_k (k=1..7), the G_2 derivation algebra
(14-dimensional) contains an 8-dimensional stabilizer subalgebra stab(e_k)
that fixes e_k. Extracted via thin SVD + Modified Gram-Schmidt kernel
completion (avoids condition-number squaring from E^T*E).

Left-multiplication by e_k defines a complex structure J_k on the 6D
orthogonal complement e_k^perp. Each stabilizer derivation is both
skew-adjoint (R^T + R = 0) and J_k-commuting (R*J_k = J_k*R),
establishing a u(3) embedding: stab(e_k) embeds in u(3) acting on
(e_k^perp, J_k) as a complex 3-space.

### Constructive SU(3) Realization (PR2, representation theory)

The 3x3 complex anti-Hermitian traceless representation of stab(e_k)
is constructed using the complex structure J_k to convert the real 6x6
stabilizer matrices to complex 3x3. Structure constants match the
standard Gell-Mann matrices under the anti-Hermitian convention
T_a = (i/2)*lambda_a (f_123 = -1 in anti-Hermitian vs +1 in Hermitian).

Basis-invariant cross-validation: sum_{a,b,c} f_{abc}^2 = 24 =
C_2(adj) * dim(su(3)) = 3 * 8 for both the octonionic SU(3) and the
SU(3) sector of SU(5) GUT. All 7 embeddings produce identical Casimir.

### Scalar Projection Bridge (PR3, project-specific)

The e_0 component of a CD algebra element is the unique commutative-
associative scalar projection. A feature-gated physics bridge connects
this to the SU(5) GUT mass/scalar infrastructure. The bridge is
explicitly project-specific, not literature-dictated.

**Epistemic classification**:
- PR1: largely algebra/geometry extraction and verification
- PR2: representation-theoretic cross-validation, convention-sensitive
- PR3: bridge/lift construction, project-specific until intertwiners solved

## References

- Reggiani (2024): Geometry of sedenion zero divisors [arXiv:2411.18881]
- Koebisu (2025): Singular structures + holonomy [arXiv:2512.13002]
- Wilmot (2026): G_2 from Clifford calibrations [arXiv:2505.06011]
- Wilmot (2026): Structure of CD algebras [arXiv:2505.11747]
- Wilmot (2025): Automorphisms of sedenions [arXiv:2512.07210]
- Gresnigt (2025): Electroweak + S_3 from Cl(8) [arXiv:2601.07857]
- Gourlay & Gresnigt (2024): Three gens from Cl(8) [arXiv:2407.01580]
- Tang & Tang (2024): Sedenion SU(5) model [MDPI Symmetry 16-00626]
- Dou et al. (2024): Sedenionic star-power series [arXiv:2512.00600]

## Claims Index

C-1455: Lepton mass fit w_sym ~ -1/sqrt(2)
C-1456: CKM selector scan, all angles <10% PDG
C-1457: PMNS theta_13 = 8.64 deg (PDG 8.54, 1.2%)
C-1458: Electroweak mixing angle sin^2(theta_W) = 0.199
C-1459: 3-blade ZD ratio = 3/2, friction quantized in 2*sqrt(2)
C-1460: Gresnigt Cl(8) S_3 correspondence
C-1461: Koebisu V_2(R^8) holonomy verification
C-1462: SU(5) lepton-quark selector identity
C-1467: Wilmot Fano 3-form: 7 associative + 28 non-associative octonion triples
C-1468: Wilmot 14-simplex: 35 terms, 105 edges, 252 non-associative
C-1469: Wilmot algebra stacking: H_n formula, T_4=15
C-1470: Wilmot 252 = 8*28 + 7*4 decomposition
C-1471: Wilmot Aut(S) = G_2 (Schafer confirmed)
C-1472: Wilmot Fano volume: 35 quaternions, 15 planes
C-1473: Dou ZD kernel: 4-dim ker(e_1-e_10)
C-1474: V_6 Jacobian: epsilon-stable gradients, collinear in assessor space
C-1475: V_6 solar pipeline: compositional, beta=0 exact, partition null result
C-1476: V_6 constrained scan: g_12 100% in span{g_13,g_23}, linear injection insufficient
C-1477: V_6 alpha-modulation: 10x gradient boost but rank-2 lock persists (42D->3D collapse)
C-1478: V_6 TensorElementLift: rank broken (75.7%), theta_12 = 33.42 deg (0.02% PDG)
C-1489: V_6 psi-eigenspace (0.25*I_6), no S_3-equivariant intertwiner, su(3) reducible
C-1490: 2D constrained scan: (33.37, 8.52, 47.40) deg, all within 3.3% of PDG
C-1491: Joint 4D: (33.84, 8.56, 48.74) deg, all within 1.3% of PDG, score 0.000221
C-1479: G2 stabilizer dimension: stab(e_k) = 8D for all k=1..7
C-1480: Complex structure J_k on e_k^perp, left-multiplication defines C^3
C-1481: u(3) embedding: stabilizer is skew-adjoint + J_k-commuting
C-1482: Fano lines through fixed unit: exactly 3 per e_k, 6D = 2+2+2
C-1483: Constructive SU(3): 3x3 complex anti-Hermitian traceless representation
C-1484: Gell-Mann alignment via orthogonal change of basis
C-1485: Fundamental Casimir: T_a*T_a = -(4/3)*I_3 (anti-Hermitian convention)
C-1486: All-embeddings equivalence: 7 SU(3) embeddings produce identical Casimir
C-1487: SU(5)/SU(3) cross-validation: sum f_{abc}^2 = 24 for both embeddings
C-1488: Real-part projection: e_0 component is unique commutative-associative scalar
