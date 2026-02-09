<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Claims Batch Backlog (C-101..C-150) (2026-02-03)

Purpose: planning snapshot for claim-by-claim audits (not evidence).

- Matrix: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Domain map: `docs/claims/CLAIMS_DOMAIN_MAP.csv`
- Claims in range: 49
- Open claims in range: 0

## Open claims (in-range, oldest-first by last_verified)

- (none)

## Details (all claims in range)

| Claim | Domains | Status | Last verified | Claim (short) | Where stated (short) | Evidence / notes (short) |
|---|---|---|---|---|---|---|
| C-101 | algebra | Verified | 2026-01-31 | Flexibility identity A(x,y,x)=0 holds at ALL Cayley-Dickson dimensions throug... | src/scripts/analysis/cd_algebraic_experiments_v9.py | Key metric: 1970-01-01 (unknown). Notes: This is effectively a numerical proo... |
| C-102 | meta | Verified (Monte Carlo; st... | 2026-02-04 | Alternativity ratio\|\|A(x,x,y)\|\|^2 /\|\|A(x,y,z)\|\|^2 converges to approx... | src/scripts/analysis/c102_alt_ratio_convergence_audit.py, data/csv/c102_alt_r... | Offline artifacts reproduce the legacy expZ ratio trend across dims 16..1024:... |
| C-103 | meta | Verified (algorithmic sam... | 2026-02-04 | ZD manifold topology shows sharp percolation transition at angular distance ~... | src/scripts/analysis/c103_zd_topology_percolation_audit.py, data/csv/c103_zd_... | Offline artifacts compute a percolation-like connectivity jump for a determin... |
| C-104 | meta | Verified | 2026-01-31 | Cross-term correlation decay is better modeled by inverse polynomial or log-c... | src/scripts/analysis/cd_algebraic_experiments_v9.py | Key metric: 1970-01-01 (unknown). Notes: The pure power law (gamma=1.85) fits... |
| C-105 | meta | Verified | 2026-01-31 | Associator tensor SV ratio grows as dim^1.65, extending from dim=8 through di... | src/scripts/analysis/cd_algebraic_experiments_v9.py | Key metric: 1970-01-01 (unknown). Notes: The SV ratio grows superlinearly. At... |
| C-106 | algebra | Verified | 2026-01-31 | Non-diagonal zero divisors exist at dim=32 with kernel dimension exactly 4 an... | src/scripts/analysis/cd_algebraic_experiments_v9.py | Key metric: 1970-01-01 (unknown). Notes: Key discovery: the annihilator dimen... |
| C-107 | holography | Verified | 2026-01-31 | Flexibility identity A(x,y,x)=0 holds to machine precision through dim=2048,... | src/scripts/analysis/cd_algebraic_experiments_v10.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: Verifies C-101 with independent samp... |
| C-108 | meta | Verified (Monte Carlo; ca... | 2026-02-03 | Alternativity ratio converges to R_inf = 0.514 +/- 0.003, consistently above... | src/scripts/analysis/c108_alt_ratio_convergence_audit.py, data/csv/c108_alt_r... | Offline extract reproduces the cached v10 fit: r_inf=0.5138852738 and r_inf_e... |
| C-109 | holography | Verified (cached probe +... | 2026-02-04 | Random probing fails to find algebraic ZDs; diagonal ZDs lifted via CD doubli... | src/scripts/analysis/c109_zd_construction_audit.py, data/csv/c109_zd_construc... | Offline artifacts encode both parts of the legacy claim: (1) cached v10 probi... |
| C-110 | meta | Verified | 2026-01-31 | Associator tensor has multilinear rank (dim-1, dim-1, dim-1) with cubic symme... | src/scripts/analysis/cd_algebraic_experiments_v10.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The dim-1 rank means the associator... |
| C-111 | holography | Verified | 2026-01-31 | Complete 13-dimension CD property table (dim=2 through 8192): flexibility and... | src/scripts/analysis/cd_algebraic_experiments_v10.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: This is the definitive CD property t... |
| C-112 | meta | Verified | 2026-01-31 | Alternativity ratio converges to R_inf = 0.507 +/- 0.003 with left/right symm... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: With dim=16384 data included, the te... |
| C-113 | meta | Verified | 2026-01-31 | Moufang ratio\|\|M(a,b,c)\|\|^2/\|\|A(a,b,c)\|\|^2 converges to M_inf = 1.561... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The Moufang-to-associator ratio conv... |
| C-114 | meta | Verified | 2026-01-31 | Full power-associativity x^a  x^b = x^{a+b} holds for all (a,b) with a+b <= 8... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: This extends C-111 from just (aa)a=a... |
| C-115 | holography | Verified | 2026-01-31 | The commutator [a,b] and associator A(a,b,c) are asymptotically orthogonal in... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: In octonions, commutator and associa... |
| C-116 | meta | Verified | 2026-01-31 | L-BFGS-B gradient descent finds non-diagonal ZDs at dim=16, 32, and 64, ALL w... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: Extends C-106 from dim=16,32 to dim=... |
| C-117 | spectral | Verified | 2026-01-31 | Associator tensor spectral gap is nearly constant (~3.0) while effective rank... | src/scripts/analysis/cd_algebraic_experiments_v11.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The near-constant spectral gap means... |
| C-118 | algebra | Verified | 2026-01-31 | The Jordan identity J(x^2, y, x) = (x^2y)x - x^2(yx) = 0 holds at ALL Cayley-... | src/scripts/analysis/cd_algebraic_experiments_v12.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: This is the first systematic test of... |
| C-119 | algebra | Verified | 2026-01-31 | The left and right Bol identities hold exactly through dim=8 (octonions) and... | src/scripts/analysis/cd_algebraic_experiments_v12.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: Bol identities (left: a(b(ac)) = (a(... |
| C-120 | meta | Verified (Monte Carlo; ca... | 2026-02-03 | Diagonal ZD kernels scale as dim/4 (not universally 4). Lifted 16D ZDs also h... | src/scripts/analysis/c120_zd_kernel_scaling_audit.py, data/csv/c120_zd_kernel... | Offline extract shows kernel_dim == dim/4 for all cached v12 rows (5 diagonal... |
| C-121 | meta | Verified | 2026-01-31 | Multi-seed bootstrap: Moufang ratio converges to M_inf = 1.519 +/- 0.013, con... | src/scripts/analysis/cd_algebraic_experiments_v12.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: SUPERSEDES C-113. The v11 single-see... |
| C-122 | meta | Verified | 2026-01-31 | Multi-seed bootstrap: Alternativity ratio converges to R_inf = 0.504 +/- 0.00... | src/scripts/analysis/cd_algebraic_experiments_v12.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: Refines C-112 with multi-seed bootst... |
| C-123 | legacy | Verified (Monte Carlo; ca... | 2026-02-03 | The associator Lie bracket [A(a,b,c), A(d,e,f)] = A1A2 - A2A1 has relative no... | src/scripts/analysis/c123_associator_lie_bracket_audit.py, data/csv/c123_asso... | Offline extract shows last_rel_bracket=1.9834713613 at dim=128 with max_abs_m... |
| C-124 | algebra | Verified | 2026-01-31 | The flexibility identity (ab)(ca) = a((bc)a) holds exactly through dim=8 (oct... | src/scripts/analysis/cd_algebraic_experiments_v13.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: Flexibility fails at exactly the sam... |
| C-125 | holography | Verified | 2026-01-31 | Artin's theorem (2-generated subalgebras are associative) holds through dim=8... | src/scripts/analysis/cd_algebraic_experiments_v13.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The key discovery: while Artin's the... |
| C-126 | meta | Verified | 2026-01-31 | The nested Lie bracket [[A1,A2],A3] has relative norm stabilizing at ~1.95 (r... | src/scripts/analysis/cd_algebraic_experiments_v13.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: CORRECTS C-123's implication that as... |
| C-127 | meta | Verified | 2026-01-31 | The full Jordan product x o y = (xy+yx)/2 satisfies the Jordan identity (x o... | src/scripts/analysis/cd_algebraic_experiments_v13.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: This strengthens C-118 by using the... |
| C-128 | holography, legacy, algebra | Verified (cached v13 expAZ) | 2026-02-04 | The conjugate inverse x^{-1} = conj(x)/\|\|x\|\|^2 gives x  x^{-1} = x^{-1}... | src/scripts/analysis/c128_conjugate_inverse_audit.py, data/csv/c128_conjugate... | Offline extract records max inverse errors <= 6.66e-16 (left and right) acros... |
| C-129 | holography, legacy | Verified (cached v13 expBA) | 2026-02-04 | The associator norm distribution concentrates as dim grows (CV -> 0.01). The... | src/scripts/analysis/c129_associator_distribution_concentration_audit.py, dat... | Offline extract records CV shrinking from 0.307 (dim=8) to 0.0403 (dim=512) w... |
| C-130 | holography, legacy | Verified (Monte Carlo; ca... | 2026-02-03 | The associator norm\|\|A(a,b,c)\|\|-> sqrt(2) because (ab)c and a(bc) become... | src/scripts/analysis/c130_associator_norm_sqrt2_audit.py, data/csv/c130_assoc... | Offline extract reports mean_sq_highdim=1.9883911277 with mean_cos_highdim=0.... |
| C-131 | meta | Verified | 2026-01-31 | Identity violation ratios are universal constants: alt/assoc -> 1/2, mouf/ass... | src/scripts/analysis/cd_algebraic_experiments_v14.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: This establishes a UNIVERSAL RATIO T... |
| C-133 | holography | Verified | 2026-01-31 | The Moufang defect and associator are asymptotically orthogonal (cos -> 0, pe... | src/scripts/analysis/cd_algebraic_experiments_v14.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The Moufang defect, associator, and... |
| C-134 | meta | Verified | 2026-01-31 | ZD pair products at dim=16 show rich combinatorial structure: of 18 tested pa... | src/scripts/analysis/cd_algebraic_experiments_v14.py, data/csv/cd_algebraic_e... | Key metric: 1970-01-01 (unknown). Notes: The ZD pair interaction space has di... |
| C-135 | legacy | Verified (cached v14 expBG) | 2026-02-04 | Power norms\|\|x^n\|\|= 1 EXACTLY (to machine precision) at ALL CD dimensions... | src/scripts/analysis/c135_power_norm_scaling_audit.py, data/csv/c135_power_no... | Offline extract records 112 (dim,power) pairs across dims 4..256 and powers 1... |
| C-136 | meta | Verified (math) | 2026-02-01 | Norm multiplicativity\|\|xy\|\|=\|\|x\|\|\|\|y\|\|holds exactly through dim=8... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBHNormMultiplicativity (5 test... |
| C-137 | holography | Verified (math) | 2026-02-01 | ZD products at dim=32 preserve the norm trichotomy {0, 1, sqrt(2)} from dim=1... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBIZDProductsDim32 (5 tests). N... |
| C-138 | holography, algebra | Verified (math) | 2026-02-01 | 3-generated subalgebras become non-associative at dim=8 (octonions), while 2-... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBJThreeGenSubalgebra (6 tests)... |
| C-139 | meta | Verified (math) | 2026-02-01 | Violation ratios at dim=8192 with 5-seed bootstrap: alt/assoc = 0.497 +/- 0.0... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBKRatioPrecision (4 tests). Al... |
| C-140 | meta | Verified (math) | 2026-02-01 | Associator component entropy: relative entropy ~ 0.84 (not uniform), effectiv... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBLAssocComponentEntropy (5 tes... |
| C-141 | meta | Verified (math) | 2026-02-01 | Mixed product norms:\|\|(ab)c\|\|and\|\|a(bc)\|\|both concentrate near 1.0 fo... | src/scripts/analysis/cd_algebraic_experiments_v15.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v9.py::TestBMMixedProductNorms (5 tests).... |
| C-142 | meta | Verified (math) | 2026-02-01 | Power-associativity x^m  x^n = x^(m+n) holds exactly (machine epsilon) at ALL... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBNPowerAssociativity (5 tests... |
| C-143 | meta | Verified (math) | 2026-02-01 | Left and right multiplication operators L_a, R_a have identical singular valu... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBOLROperatorSpectra (6 tests)... |
| C-144 | meta | Verified (math) | 2026-02-01 | ZD kernel spectrum at dim=64 has 9 distinct values {4, 12, 16, 20, 24, 28, 32... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBPZDKernelDim64 (5 tests). Ke... |
| C-145 | holography | Verified (math) | 2026-02-01 | Four-element products have exactly 5 distinct bracketings. At dim=4 (associat... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBQFourProduct (5 tests). dim=... |
| C-146 | meta | Verified (math) | 2026-02-01 | Inner derivation D(a,b)(x) = [[a,b],x] + [[a,x],b] + [[x,b],a] satisfies the... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBRDerivation (4 tests). Leibn... |
| C-147 | holography | Verified (math) | 2026-02-01 | Alternator-associator decomposition: A(a,b,c) splits into alternating part Al... | src/scripts/analysis/cd_algebraic_experiments_v16.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v10.py::TestBSAlternatorDecomposition (6... |
| C-148 | meta | Verified (math) | 2026-02-01 | The nucleus N(A) of a CD algebra equals the full algebra at dim=4 (quaternion... | src/scripts/analysis/cd_algebraic_experiments_v17.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v11.py::TestBTNucleus (4 tests). Nucleus... |
| C-149 | holography | Verified (math) | 2026-02-01 | Composition defect delta =\|\|xy\|\|^2 -\|\|x\|\|^2\|\|y\|\|^2 is identically... | src/scripts/analysis/cd_algebraic_experiments_v17.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v11.py::TestBUCompositionDefect (5 tests)... |
| C-150 | holography | Verified (math) | 2026-02-01 | Quadruple associator A(a,b,c,d) has\|\|A(a,b,cd)\|\|~\|\|A(a,b,c)\|\|(ratio 0... | src/scripts/analysis/cd_algebraic_experiments_v17.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v11.py::TestBVQuadrupleAssociator (4 test... |
