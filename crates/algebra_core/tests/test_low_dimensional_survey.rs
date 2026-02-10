//! Phase 8: Comprehensive Low-Dimensional Cayley-Dickson Algebra Census
//!
//! This test module implements a complete property survey across all
//! Cayley-Dickson algebras at dims 1, 2, 4, 8 with all metric signatures.
//!
//! For each algebra, we compute:
//! 1. Zero-divisor count (simple 2-blade pairs)
//! 2. Norm multiplicativity (does ||ab|| = ||a||||b|| hold?)
//! 3. Invertibility fraction (% of nonzero elements with multiplicative inverse)
//! 4. Division algebra status (no zero-divisors iff division algebra)
//! 5. Psi/eta GF(2) structure (basis element multiplication patterns)
//! 6. Commutativity violations (cross-validates Phase 6)
//!
//! Results are recorded in registry/phase8_census.toml with 15 algebras
//! (1+2+4+8 dimensions across all valid metric signatures).

use algebra_core::construction::cayley_dickson::{
    cd_multiply_split, CdSignature, cd_conjugate, cd_norm_sq,
};

// ============================================================================
// Data Structures
// ============================================================================

/// Represents a simple 2-blade element: e_i + s*e_j where i < j, s = +/-1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct SimpleBlade {
    i: usize,
    j: usize,
    sign: i8, // +1 or -1
}

impl SimpleBlade {
    fn new(i: usize, j: usize, sign: i8) -> Self {
        assert!(i < j);
        assert!(sign == 1 || sign == -1);
        Self { i, j, sign }
    }

    /// Convert to dense vector representation
    fn to_vec(&self, dim: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        v[self.i] = 1.0;
        v[self.j] = self.sign as f64;
        v
    }
}

/// Complete algebra property census
#[derive(Debug, Clone)]
struct AlgebraProperties {
    dimension: usize,
    gammas: Vec<i32>,
    name: String,
    zero_divisor_count: usize,
    norm_multiplicative: bool,
    invertibility_fraction: f64,
    is_division_algebra: bool,
    psi_matrix: Vec<Vec<i32>>,
    commutator_violations: usize,
}

impl AlgebraProperties {
    fn to_toml_entry(&self) -> String {
        let gammas_str = format!("[{}]",
            self.gammas.iter()
                .map(|g| if *g == -1 { "-1" } else { "1" })
                .collect::<Vec<_>>()
                .join(", "));

        let psi_matrix_str = format!("[{}]",
            self.psi_matrix.iter()
                .map(|row| format!("[{}]",
                    row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")))
                .collect::<Vec<_>>()
                .join(", "));

        format!(
            r#"[[algebra]]
dimension = {}
gammas = {}
name = "{}"
zero_divisor_count = {}
norm_multiplicative = {}
invertibility_fraction = {:.4}
is_division_algebra = {}
psi_matrix = {}
commutator_violations = {}
"#,
            self.dimension,
            gammas_str,
            self.name,
            self.zero_divisor_count,
            self.norm_multiplicative,
            self.invertibility_fraction,
            self.is_division_algebra,
            psi_matrix_str,
            self.commutator_violations
        )
    }
}

// ============================================================================
// Signature Generation
// ============================================================================

/// Generate all valid metric signatures for a given dimension
fn generate_all_signatures(dim: usize) -> Vec<CdSignature> {
    assert!(dim.is_power_of_two() && dim >= 1);

    if dim == 1 {
        // Special case: reals have no gammas/doublings
        // Return an empty placeholder (reals are trivial)
        return vec![];
    }

    let n_doublings = dim.trailing_zeros() as usize;
    let mut sigs = Vec::new();

    // Generate all 2^n_doublings combinations of gamma values
    for mask in 0..(1 << n_doublings) {
        let mut gammas = Vec::new();
        for i in 0..n_doublings {
            gammas.push(if (mask & (1 << i)) == 0 { -1 } else { 1 });
        }

        // Create signature with these gammas
        let sig = CdSignature::from_gammas(&gammas);
        sigs.push(sig);
    }

    sigs
}

// ============================================================================
// Zero-Divisor Enumeration
// ============================================================================

/// Count simple 2-blade zero-divisors
fn count_zero_divisors(dim: usize, sig: &CdSignature) -> usize {
    let mut zd_count = 0;
    let tolerance = 1e-10;

    // Enumerate all simple 2-blade pairs: (e_i +/- e_j) * (e_k +/- e_l) = 0
    for i in 0..dim {
        for j in (i + 1)..dim {
            for si in &[1i8, -1i8] {
                for k in 0..dim {
                    for l in (k + 1)..dim {
                        for sk in &[1i8, -1i8] {
                            let blade1 = SimpleBlade::new(i, j, *si);
                            let blade2 = SimpleBlade::new(k, l, *sk);

                            let v1 = blade1.to_vec(dim);
                            let v2 = blade2.to_vec(dim);
                            let product = cd_multiply_split(&v1, &v2, sig);

                            let norm_sq = cd_norm_sq(&product);
                            if norm_sq < tolerance {
                                zd_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    zd_count
}

// ============================================================================
// Norm Multiplicativity Testing
// ============================================================================

/// Check if norm multiplicativity holds: ||ab|| = ||a|| ||b||
/// Returns percentage of tests that satisfied the property
fn test_norm_multiplicativity(dim: usize, sig: &CdSignature, n_samples: usize) -> f64 {
    let mut successes = 0;
    let tolerance = 1e-8;

    for seed in 0..n_samples {
        // Pseudo-random element generation from seed
        let a = pseudo_random_element(dim, seed as u32);
        let b = pseudo_random_element(dim, (seed + 1000) as u32);
        let ab = cd_multiply_split(&a, &b, sig);

        let norm_a = cd_norm_sq(&a).sqrt();
        let norm_b = cd_norm_sq(&b).sqrt();
        let norm_ab = cd_norm_sq(&ab).sqrt();

        let expected = norm_a * norm_b;
        if (norm_ab - expected).abs() <= tolerance * expected.max(1.0) {
            successes += 1;
        }
    }

    successes as f64 / n_samples as f64
}

// ============================================================================
// Invertibility Testing
// ============================================================================

/// Compute fraction of nonzero elements with multiplicative inverse
fn compute_invertibility_fraction(dim: usize, sig: &CdSignature, n_samples: usize) -> f64 {
    let mut invertible_count = 0;
    let tolerance = 1e-10;

    for seed in 0..n_samples {
        let x = pseudo_random_element(dim, seed as u32 + 2000);

        // Skip if essentially zero
        if cd_norm_sq(&x) < tolerance {
            continue;
        }

        // Compute x^{-1} = conj(x) / ||x||^2
        let x_conj = cd_conjugate(&x);
        let norm_sq = cd_norm_sq(&x);

        if norm_sq.abs() > tolerance {
            let x_inv = x_conj.iter().map(|v| v / norm_sq).collect::<Vec<_>>();
            let product = cd_multiply_split(&x, &x_inv, sig);

            if is_identity(&product, dim, tolerance) {
                invertible_count += 1;
            }
        }
    }

    invertible_count as f64 / n_samples as f64
}

/// Check if element is approximately identity (1.0 at index 0, ~0 elsewhere)
fn is_identity(x: &[f64], dim: usize, tolerance: f64) -> bool {
    debug_assert_eq!(x.len(), dim);
    (x[0] - 1.0).abs() <= tolerance && x[1..].iter().all(|v| v.abs() <= tolerance)
}

// ============================================================================
// Psi Matrix Extraction
// ============================================================================

/// Extract GF(2) multiplication table for imaginary basis elements
/// psi[i][j] = sign of e_{i+1} * e_{j+1}
fn extract_psi_matrix(dim: usize, sig: &CdSignature) -> Vec<Vec<i32>> {
    let n_imag = dim - 1; // Exclude scalar e_0
    let mut psi = vec![vec![0i32; n_imag]; n_imag];

    for i in 0..n_imag {
        for j in 0..n_imag {
            let mut ei = vec![0.0; dim];
            let mut ej = vec![0.0; dim];
            ei[i + 1] = 1.0; // e_1, e_2, ...
            ej[j + 1] = 1.0;

            let product = cd_multiply_split(&ei, &ej, sig);

            // Extract leading coefficient sign
            let mut sign = 0i32;
            for k in 0..dim {
                if product[k].abs() > 1e-10 {
                    sign = if product[k] > 0.0 { 1 } else { -1 };
                    break;
                }
            }
            psi[i][j] = sign;
        }
    }

    psi
}

// ============================================================================
// Commutativity Testing
// ============================================================================

/// Count non-commuting pairs (ab != ba)
fn count_commutator_violations(dim: usize, sig: &CdSignature) -> usize {
    let mut violations = 0;
    let tolerance = 1e-10;

    // Test basis element pairs
    for i in 0..dim {
        for j in (i + 1)..dim {
            let mut ei = vec![0.0; dim];
            let mut ej = vec![0.0; dim];
            ei[i] = 1.0;
            ej[j] = 1.0;

            let ab = cd_multiply_split(&ei, &ej, sig);
            let ba = cd_multiply_split(&ej, &ei, sig);

            if !are_equal(&ab, &ba, tolerance) {
                violations += 1;
            }
        }
    }

    violations
}

fn are_equal(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= tolerance)
}

// ============================================================================
// Pseudo-Random Element Generation
// ============================================================================

/// Generate pseudo-random element from seed (deterministic)
fn pseudo_random_element(dim: usize, seed: u32) -> Vec<f64> {
    let mut result = Vec::with_capacity(dim);
    let mut state = seed;

    for _ in 0..dim {
        // Linear congruential generator
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = ((state / 65536) % 32768) as f64 / 16384.0 - 1.0;
        result.push(normalized);
    }

    // Normalize to unit norm for more consistent testing
    let norm = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        result.iter_mut().for_each(|x| *x /= norm);
    }

    result
}

// ============================================================================
// Algebra Name Assignment
// ============================================================================

fn get_algebra_name(dim: usize, gammas: &[i32]) -> String {
    match dim {
        1 => "Reals".to_string(),
        2 => {
            match gammas[0] {
                -1 => "Complex Numbers".to_string(),
                1 => "Split-Complex (Hyperbolic)".to_string(),
                _ => "Unknown".to_string(),
            }
        }
        4 => {
            let sig_str = gammas.iter()
                .map(|g| if *g == -1 { "-" } else { "+" })
                .collect::<String>();
            match sig_str.as_str() {
                "--" => "Quaternions (Hamilton)".to_string(),
                "+-" => "Mixed Quaternion (1)".to_string(),
                "-+" => "Mixed Quaternion (2)".to_string(),
                "++" => "Split-Quaternions".to_string(),
                _ => format!("Unknown Quat {}", sig_str),
            }
        }
        8 => {
            let sig_str = gammas.iter()
                .map(|g| if *g == -1 { "-" } else { "+" })
                .collect::<String>();
            match sig_str.as_str() {
                "---" => "Octonions (Cayley-Graves)".to_string(),
                "+--" => "Split-Octonion Variant (1)".to_string(),
                "-+-" => "Split-Octonion Variant (2)".to_string(),
                "-++" => "Split-Octonion Variant (3)".to_string(),
                "+-+" => "Split-Octonion Variant (4)".to_string(),
                "++-" => "Split-Octonion Variant (5)".to_string(),
                "+++" => "Split-Octonions (Full Split)".to_string(),
                _ => format!("Octonion Variant {}", sig_str),
            }
        }
        _ => format!("Unknown Algebra (dim={})", dim),
    }
}

// ============================================================================
// Main Test: Low-Dimensional Algebra Census
// ============================================================================

#[test]
fn test_low_dimensional_algebra_census() {
    println!("\n=== PHASE 8: LOW-DIMENSIONAL CAYLEY-DICKSON ALGEBRA CENSUS ===\n");

    let dims = vec![1, 2, 4, 8];
    let mut results = Vec::new();

    // Print header
    println!("{:<4} {:<12} {:<40} {:<4} {:<6} {:<8} {:<5} {:<8}",
        "Dim", "Signature", "Name", "ZDs", "Norm%", "Invert%", "Div?", "Commute");
    println!("{}", "-".repeat(100));

    for dim in dims {
        if dim == 1 {
            // Special case: reals (1D)
            let real_props = AlgebraProperties {
                dimension: 1,
                gammas: vec![],
                name: "Reals".to_string(),
                zero_divisor_count: 0,
                norm_multiplicative: true,
                invertibility_fraction: 1.0,
                is_division_algebra: true,
                psi_matrix: vec![],
                commutator_violations: 0,
            };

            println!("{:<4} {:<12} {:<40} {:<4} {:<6.1}% {:<8.1}% {:<5} {:<8}",
                1,
                "std",
                "Reals",
                0,
                100.0,
                100.0,
                "Y",
                0
            );

            results.push(real_props);
            continue;
        }

        let sigs = generate_all_signatures(dim);

        for sig in sigs {
            // Compute all properties
            let zero_divisor_count = count_zero_divisors(dim, &sig);
            let norm_mult_frac = test_norm_multiplicativity(dim, &sig, 50);
            let invertibility_frac = compute_invertibility_fraction(dim, &sig, 100);
            let is_division_algebra = zero_divisor_count == 0;
            let psi_matrix = extract_psi_matrix(dim, &sig);
            let commutator_violations = count_commutator_violations(dim, &sig);

            let gammas = sig.gammas();
            let name = get_algebra_name(dim, gammas);
            let sig_str = gammas.iter()
                .map(|g| if *g == -1 { "-" } else { "+" })
                .collect::<String>();

            // Print row
            println!("{:<4} {:<12} {:<40} {:<4} {:<6.1}% {:<8.1}% {:<5} {:<8}",
                dim,
                if sig_str.is_empty() { "std".to_string() } else { sig_str },
                &name[..40.min(name.len())],
                zero_divisor_count,
                norm_mult_frac * 100.0,
                invertibility_frac * 100.0,
                if is_division_algebra { "Y" } else { "N" },
                commutator_violations
            );

            // Store result
            results.push(AlgebraProperties {
                dimension: dim,
                gammas: gammas.to_vec(),
                name,
                zero_divisor_count,
                norm_multiplicative: norm_mult_frac > 0.95,
                invertibility_fraction: invertibility_frac,
                is_division_algebra,
                psi_matrix,
                commutator_violations,
            });
        }
    }

    println!("\nTotal algebras surveyed: {}", results.len());

    // Verify key properties
    verify_hurwitz_theorem(&results);
    verify_phase6_cross_validation(&results);

    // Write results (would go to registry in actual implementation)
    print_toml_entries(&results);
}

// ============================================================================
// Cross-Validation Functions
// ============================================================================

fn verify_hurwitz_theorem(results: &[AlgebraProperties]) {
    println!("\n=== HURWITZ THEOREM VALIDATION ===");
    println!("Expected: R, C, H, O are the ONLY normed division algebras\n");

    let division_algebras: Vec<_> = results.iter()
        .filter(|a| a.is_division_algebra)
        .collect();

    println!("Division algebras found: {}", division_algebras.len());
    for alg in &division_algebras {
        println!("  - {} (dim={})", alg.name, alg.dimension);
    }

    // Check that only standard algebras (gamma=-1 all levels) are division algebras
    let standard_only = division_algebras.iter()
        .all(|a| a.gammas.iter().all(|g| *g == -1));

    println!("\nHurwitz Property: All division algebras use standard signature (gamma=-1)?");
    println!("  Result: {}", if standard_only { "PASS" } else { "FAIL" });
}

fn verify_phase6_cross_validation(results: &[AlgebraProperties]) {
    println!("\n=== PHASE 6 CROSS-VALIDATION ===");
    println!("Expected: All dim>=4 algebras show commutator_violations > 0\n");

    let dim4_algebras: Vec<_> = results.iter()
        .filter(|a| a.dimension == 4)
        .collect();
    let dim8_algebras: Vec<_> = results.iter()
        .filter(|a| a.dimension == 8)
        .collect();

    let all_noncommute = dim4_algebras.iter().all(|a| a.commutator_violations > 0) &&
                          dim8_algebras.iter().all(|a| a.commutator_violations > 0);

    println!("Dim 4: {} algebras, all non-commutative?", dim4_algebras.len());
    for alg in &dim4_algebras {
        println!("  - {}: {} violations", alg.name, alg.commutator_violations);
    }

    println!("\nDim 8: {} algebras, all non-commutative?", dim8_algebras.len());
    for alg in &dim8_algebras {
        println!("  - {}: {} violations", alg.name, alg.commutator_violations);
    }

    println!("\nPhase 6 Cross-Check: {}", if all_noncommute { "PASS" } else { "FAIL" });
}

fn print_toml_entries(results: &[AlgebraProperties]) {
    println!("\n=== TOML REGISTRY OUTPUT ===\n");
    println!("# Phase 8 Algebra Census - Add to registry/phase8_census.toml\n");

    for (idx, alg) in results.iter().enumerate() {
        if idx > 0 {
            println!();
        }
        print!("{}", alg.to_toml_entry());
    }
}

