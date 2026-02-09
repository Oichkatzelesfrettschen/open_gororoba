<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/reports_narratives.toml -->

# Claims Batch Backlog (C-251..C-300) (2026-02-02)

Purpose: planning snapshot for claim-by-claim audits (not evidence).

- Matrix: `docs/CLAIMS_EVIDENCE_MATRIX.md`
- Domain map: `docs/claims/CLAIMS_DOMAIN_MAP.csv`
- Claims in range: 50
- Open claims in range: 0

## Open claims (in-range, oldest-first by last_verified)

- (none)

## Details (all claims in range)

| Claim | Domains | Status | Last verified | Claim (short) | Where stated (short) | Evidence / notes (short) |
|---|---|---|---|---|---|---|
| C-251 | holography, spectral | Verified (math) | 2026-02-01 | Eigenvalue spectrum of L_a: all eigenvalues lie on the unit circle (\|lambda\... | src/scripts/analysis/cd_algebraic_experiments_v34.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v28.py::TestFSLaSpectrum (5 tests). Unit... |
| C-252 | meta | Verified (math) | 2026-02-01 | Commutator algebra dimension: span{[e_i, e_j]} has rank 0 at dim=2 (commutati... | src/scripts/analysis/cd_algebraic_experiments_v34.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v28.py::TestFTCommutatorDimension (6 test... |
| C-253 | holography | Verified (math) | 2026-02-01 | Associator norm scaling:\|\|A(a,b,c)\|\|for unit vectors converges to sqrt(2)... | src/scripts/analysis/cd_algebraic_experiments_v34.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v28.py::TestFUAssociatorNormScaling (5 te... |
| C-254 | meta | Verified (math) | 2026-02-01 | Conjugate reversal: conj(ab) = conj(b)conj(a) holds EXACTLY at ALL CD dims 2-... | src/scripts/analysis/cd_algebraic_experiments_v34.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v28.py::TestFVConjugateReversal (3 tests)... |
| C-255 | spectral | Verified (math) | 2026-02-01 | Left multiplication operator: L_a^2 = L_{a^2} (as matrices) holds EXACTLY at... | src/scripts/analysis/cd_algebraic_experiments_v34.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v28.py::TestFWLaSquared (5 tests). Equal... |
| C-256 | holography, spectral | Verified (math) | 2026-02-01 | R_a and L_a eigenvalue spectra match (sorted\|eigenvalues\|identical) at ALL... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestFXRaLaSpectrum (3 tests). All... |
| C-257 | holography | Verified (math) | 2026-02-01 | Associator norm concentration: CV(\|\|A\|\|) = std/mean decreases monotonical... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestFYAssociatorConcentration (5... |
| C-258 | holography | Verified (math) | 2026-02-01 | Product norm ratio:\|\|ab\|\|/(\|\|a\|\|\|\|b\|\|) = 1.0 EXACTLY at dim<=8 (c... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestFZProductNormRatio (5 tests).... |
| C-259 | algebra | Verified (math) | 2026-02-01 | Doubling-level associator: at dim=16, left-half elements (octonion subalgebra... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestGADoublingAssociator (4 tests... |
| C-260 | meta | Verified (math) | 2026-02-01 | Associator 4-form: <A(a,b,c), d> is alternating (antisymmetric under adjacent... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestGBAssociator4Form (4 tests).... |
| C-261 | meta | Verified (math) | 2026-02-01 | Center Z(A) = full algebra (dim 2) at dim=2 (complex, commutative). Center =... | src/scripts/analysis/cd_algebraic_experiments_v35.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v29.py::TestGCCenterDimension (4 tests).... |
| C-262 | meta | Verified (math) | 2026-02-01 | Associator trilinearity: A(a+b,c,d) = A(a,c,d) + A(b,c,d) and A(alphaa,c,d) =... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGDAssociatorTrilinearity (5 t... |
| C-263 | meta | Verified (math) | 2026-02-01 | Product of inverses: (ab)^{-1} = b^{-1}a^{-1} holds EXACTLY at dim<=8 (compos... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGEProductOfInverses (5 tests)... |
| C-264 | holography | Verified (math) | 2026-02-01 | Commutator-to-associator norm ratio:\|\|[a,b]\|\|/\|\|A(a,b,c)\|\|-> sqrt(2)... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGFCommutatorAssociatorRatio (... |
| C-265 | meta | Verified (math) | 2026-02-01 | Inner derivation D(a,b)(x) = [[a,b],x] - 3A(a,b,x) satisfies the Leibniz rule... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGGInnerDerivation (6 tests).... |
| C-266 | holography | Verified (math) | 2026-02-01 | Flexible nucleus = full algebra at ALL CD dims 4-128. Every element satisfies... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGHFlexibleNucleus (3 tests).... |
| C-267 | holography, algebra | Verified (math) | 2026-02-01 | Moufang identity a(b(ac)) = ((ab)a)c holds EXACTLY at dim<=8 (Moufang loop).... | src/scripts/analysis/cd_algebraic_experiments_v36.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v30.py::TestGIMoufangIdentity (5 tests).... |
| C-268 | meta | Verified (math) | 2026-02-01 | Quadratic identity: x^2 - 2Re(x)x +\|\|x\|\|^2e_0 = 0 holds EXACTLY at ALL CD... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGJQuadraticIdentity (4 tests)... |
| C-269 | meta | Verified (math) | 2026-02-01 | Power-norm for non-unit vectors:\|\|x^n\|\|=\|\|x\|\|^n at ALL CD dims 4-128... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGKPowerNormNonUnit (4 tests).... |
| C-270 | holography | Verified (math) | 2026-02-01 | Di-associator (ax)b - a(xb) = 0 at dim<=4 (associative). Nonzero at dim>=8 wi... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGLDiAssociator (6 tests). Zer... |
| C-271 | holography | Verified (math) | 2026-02-01 | Artin's theorem: subalgebra generated by any two elements is associative at d... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGMArtinSubalgebra (4 tests).... |
| C-272 | meta | Verified (math) | 2026-02-01 | Trace of commutator: Tr(L_{[a,b]}) = 0 at ALL CD dims 4-128. This follows fro... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGNTraceCommutator (3 tests).... |
| C-273 | meta | Verified (math) | 2026-02-01 | Nucleus is NOT an ideal at dim>=8. For n = lambdae_0 (scalar, in nucleus) and... | src/scripts/analysis/cd_algebraic_experiments_v37.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v31.py::TestGONucleusIdeal (5 tests). Ide... |
| C-274 | meta | Verified (math) | 2026-02-01 | Polarization identity: both standard (4<a,b> =\|\|a+b\|\|^2 -\|\|a-b\|\|^2) a... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGPPolarizationIdentity (4 tes... |
| C-275 | meta | Verified (math) | 2026-02-01 | Jordan product power-associativity: (a.b).(a.a) = a.(b.(a.a)) where a.b = (ab... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGQJordanPowerAssociativity (4... |
| C-276 | meta | Verified (math) | 2026-02-01 | Jordan triple product {a,b,c} = a.(b.c) + c.(b.a) - b.(a.c) is symmetric in (... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGRTripleProduct (4 tests). Sy... |
| C-277 | holography | Verified (math) | 2026-02-01 | Basis element squares: e_k^2 = -e_0 for all k>=1 at ALL CD dims 2-128. e_0^2... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGSBasisElementSquares (4 test... |
| C-278 | algebra | Verified (math) | 2026-02-01 | No nilpotent elements: x^n != 0 for any nonzero x at ALL CD dims 8-128. This... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGTNilpotentElements (4 tests)... |
| C-279 | holography | Verified (math) | 2026-02-01 | Alternative nucleus: ALL random elements satisfy both left and right alternat... | src/scripts/analysis/cd_algebraic_experiments_v38.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v32.py::TestGUAlternativeNucleus (5 tests... |
| C-280 | meta | Verified (math) | 2026-02-01 | L_a eigenvalue distribution: at dim<=8, all eigenvalues of L_a (unit a) lie e... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestGVLaEigenvalueDistribution (5... |
| C-281 | meta | Verified (math) | 2026-02-01 | Near-zero-divisor product norm: min(\|\|ab\|\|/(\|\|a\|\|\|\|b\|\|)) = 1.0 ex... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestGWZeroDivisorProductNorm (4 t... |
| C-282 | holography | Verified (math) | 2026-02-01 | Subalgebra embedding chain: R c C c H c O c S c P verified within dim=64. Ele... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestGXSubalgebraEmbedding (4 test... |
| C-283 | holography | Verified (math) | 2026-02-01 | Associator mean norm convergence: mean(\|\|A(a,b,c)\|\|) for unit vectors inc... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestGYAssociatorFrobenius (4 test... |
| C-284 | holography | Verified (math) | 2026-02-01 | Anti-commutator norm ratio:\|\|{a,b}\|\|/(2\|\|a\|\|\|\|b\|\|) = 1.0 at dim=2... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestGZAntiCommutatorNorm (5 tests... |
| C-285 | meta | Verified (math) | 2026-02-01 | Power tower:\|\|x^{2^k}\|\|=\|\|x\|\|^{2^k} remains exact to machine precisio... | src/scripts/analysis/cd_algebraic_experiments_v39.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v33.py::TestHAPowerTower (3 tests). All d... |
| C-286 | meta | Verified (math) | 2026-02-01 | Norm product identity: aconj(a) = conj(a)a = \|\|a\|\|^2  e_0 holds EXACTLY a... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHBNormProduct (4 tests). Both... |
| C-287 | holography | Verified (math) | 2026-02-01 | Imaginary product structure: for pure imaginary a,b (Re=0), Re(ab) = -<a,b> (... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHCImaginaryProduct (4 tests).... |
| C-288 | holography | Verified (math) | 2026-02-01 | L_a eigenvalue conjugate pairing: all eigenvalues of the left-multiplication... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHDEigenvaluePairing (4 tests)... |
| C-289 | meta | Verified (math) | 2026-02-01 | n-fold product norm: \|\|a1a2...an\|\|/ prod(\|\|ai\|\|) = 1.0 exactly at dim... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHENFoldProductNorm (5 tests).... |
| C-290 | meta | Verified (math) | 2026-02-01 | Associator antisymmetry: A(a,b,c) = -A(b,a,c) holds exactly at dim<=8 (altern... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHFAssociatorSwapRatio (5 test... |
| C-291 | meta | Verified (math) | 2026-02-01 | Generated subalgebra dimension: every nonzero element x of a CD algebra gener... | src/scripts/analysis/cd_algebraic_experiments_v40.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v34.py::TestHGGeneratedSubalgebra (3 test... |
| C-292 | holography | Verified (math) | 2026-02-01 | Moufang identity a(b(ac)) = (a(ba))c holds EXACTLY at dim<=8 (Moufang loop) a... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHHMoufangLoopFailure (4 tests... |
| C-293 | meta | Verified (math) | 2026-02-01 | Conjugate anti-automorphism: conj(ab) = conj(b)conj(a) holds at ALL CD dims 2... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHIConjugateAntiAutomorphism (... |
| C-294 | meta | Verified (math) | 2026-02-01 | Center of CD algebra: Z(A) = full algebra at dim=2 (C is commutative); Z(A) =... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHJCenter (4 tests). Full cent... |
| C-295 | meta | Verified (math) | 2026-02-01 | Cyclic associator sum: A(a,b,c) + A(b,c,a) + A(c,a,b) = 0 at dim<=4 (associat... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHKCyclicAssociatorSum (6 test... |
| C-296 | holography | Verified (math) | 2026-02-01 | Left-right multiplication intertwining: xa = conj(conj(a)conj(x)) holds at AL... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHLLRIntertwining (3 tests). B... |
| C-297 | holography | Verified (math) | 2026-02-01 | Product of conjugates: conj(a)conj(b) = conj(ba) holds at ALL CD dims 2-256 w... | src/scripts/analysis/cd_algebraic_experiments_v41.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v35.py::TestHMProductOfConjugates (3 test... |
| C-298 | holography, spectral | Verified (math) | 2026-02-01 | Trace of left-multiplication: Tr(L_a) = dim  Re(a) holds EXACTLY at ALL CD di... | src/scripts/analysis/cd_algebraic_experiments_v42.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v36.py::TestHNTraceLa (3 tests). Exact at... |
| C-299 | holography | Verified (math) | 2026-02-01 | Right Bol identity (ab)(ca) = a((bc)a) holds EXACTLY at dim<=8 and FAILS at d... | src/scripts/analysis/cd_algebraic_experiments_v42.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v36.py::TestHORightBol (4 tests). Bol iff... |
| C-300 | holography | Verified (math) | 2026-02-01 | Commutator-anticommutator decomposition: \|\|[a,b]\|\|^2 + \|\|{a,b}\|\|^2 =... | src/scripts/analysis/cd_algebraic_experiments_v42.py, data/csv/cd_algebraic_e... | tests/test_cd_algebraic_experiments_v36.py::TestHPCommAnticommDecomposition (... |
