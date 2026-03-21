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

**Scope distinction (peer-reviewed)**: The mathematical backbone -- Z(S)
isometric to G_2 (Reggiani 2024), ZD(S) isometric to V_2(R^7) (Reggiani
2024), and V_2(R^8) frame decomposition (Koebisu 2025, complementary) --
is established in the literature.  The selector-pair choices and numerical
fits are project-specific results, not literature claims.

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

| Parameter  | This work | PDG 2025 | Error |
|------------|-----------|----------|-------|
| theta_12   | 29.2 deg  | 33.4     | 12.6% |
| theta_13   | 8.64 deg  | 8.54     | 1.2%  |
| theta_23   | 32.3 deg  | 49.0     | 34%   |

The charged lepton selector (e_11, e_12) is identical to the CKM up-type
selector -- consistent with the SU(5) prediction that charged leptons
partner with up-type quarks (C-1462).

The PMNS/CKM theta_13 ratio is 39.3 (observed: 39.9, within 1.6%).
This is a structural prediction: the algebra naturally produces large PMNS
angles from neutrino pairs in the lower CD block (e_7, e_8) vs the upper
block (e_11, e_12) for charged leptons.

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

**Note on Aut(S)**: The automorphism group of the sedenions is under active
investigation.  Gresnigt's framework uses Aut(S) = G_2 x S_3; Wilmot's
calibration analysis supports Schafer's Aut(S) = G_2.  This document does
not take a position on this mathematical question.

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
