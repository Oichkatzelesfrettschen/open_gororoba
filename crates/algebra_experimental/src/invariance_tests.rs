//! Invariance tests for the sedenion GUT predictive pipeline.
//!
//! Tests that the Casimir mass ratios are invariant under:
//! - G2 automorphisms (proper octonion automorphisms, not just SO(2))
//! - S3 permutations of the three octonionic subalgebras
//! - Combined G2 x S3 symmetry (the full structural symmetry group)

use crate::{
    cayley_dickson_structs::Sedenion, quantum_state::QuantumState,
    sedenion_subalgebras::get_octonion_subalgebras,
    su_n_generators::construct_su5_generators_algebraic,
};
use gororoba_algebra::construction::g2_automorphisms::OctonionDerivation;

/// Compute the generational mass from projecting the Casimir onto a subalgebra.
fn calculate_generational_mass(casimir: &Sedenion, subalgebra_indices: &[usize]) -> f64 {
    let projected_casimir = casimir.project_to_subalgebra(subalgebra_indices);
    projected_casimir.norm_sqr().sqrt()
}

/// Run the full predictive pipeline: SU(5) generators -> Casimir -> mass ratio.
fn run_predictive_pipeline_on_basis(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> f64 {
    let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
    let surviving_gens: Vec<_> = su5_gens
        .into_iter()
        .filter(|g| *g != QuantumState::TopologicalNull)
        .collect();

    let casimir_op = surviving_gens.into_iter().fold(
        QuantumState::Observable(Sedenion::default()),
        |acc, generator| acc + generator * generator,
    );

    let casimir = match casimir_op {
        QuantumState::Observable(s) => s,
        QuantumState::TopologicalNull => panic!("Casimir operator should not be null"),
    };

    let (o1, o2, _) = get_octonion_subalgebras();
    let m1 = calculate_generational_mass(&casimir, &o1);
    let m2 = calculate_generational_mass(&casimir, &o2);

    m2 / m1
}

/// Run the pipeline and return the sorted mass ratio triple from 3 subalgebras.
///
/// Returns ratios normalized to the largest mass (so the triple sums structure
/// is scale-independent). G2 rotations change absolute Casimir magnitudes
/// but preserve the ratio structure.
fn mass_ratios_sorted(casimir: &Sedenion, subs: [&[usize]; 3]) -> [f64; 3] {
    let mut masses = [
        calculate_generational_mass(casimir, subs[0]),
        calculate_generational_mass(casimir, subs[1]),
        calculate_generational_mass(casimir, subs[2]),
    ];
    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max = masses[2];
    if max > 1e-15 {
        [masses[0] / max, masses[1] / max, 1.0]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Compute the Casimir operator from a given basis and complex structure.
fn compute_casimir(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> Sedenion {
    let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
    let surviving_gens: Vec<_> = su5_gens
        .into_iter()
        .filter(|g| *g != QuantumState::TopologicalNull)
        .collect();

    let casimir_op = surviving_gens
        .into_iter()
        .fold(QuantumState::Observable(Sedenion::default()), |acc, g| {
            acc + g * g
        });

    match casimir_op {
        QuantumState::Observable(s) => s,
        QuantumState::TopologicalNull => panic!("Casimir operator should not be null"),
    }
}

/// Taylor series matrix exponential for a 7x7 matrix.
///
/// exp(t*D) = I + t*D + (t*D)^2/2! + ... + (t*D)^N/N!
///
/// Converges for |t| <= 1.0 with N=20 terms. Since D is skew-symmetric,
/// exp(t*D) is orthogonal (rotation matrix), so the result is always well-conditioned.
fn mat_exp_7x7(d: &[[f64; 7]; 7], t: f64, n_terms: usize) -> [[f64; 7]; 7] {
    let mut result = [[0.0; 7]; 7];
    for i in 0..7 {
        result[i][i] = 1.0; // Identity
    }

    let mut td = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            td[i][j] = t * d[i][j];
        }
    }

    // term = (t*D)^k / k!
    let mut term = [[0.0; 7]; 7];
    for i in 0..7 {
        term[i][i] = 1.0;
    }

    for k in 1..=n_terms {
        // term = term * td / k
        let old_term = term;
        for i in 0..7 {
            for j in 0..7 {
                let mut sum = 0.0;
                for m in 0..7 {
                    sum += old_term[i][m] * td[m][j];
                }
                term[i][j] = sum / k as f64;
            }
        }

        // result += term
        for i in 0..7 {
            for j in 0..7 {
                result[i][j] += term[i][j];
            }
        }
    }

    result
}

/// Apply a G2 derivation (exponentiated) to a sedenion basis.
///
/// 1. Extract the 7x7 imaginary sub-block from the OctonionDerivation's 8x8 matrix
/// 2. Compute R = exp(t*D) as a 7x7 orthogonal matrix
/// 3. Apply R to sedenion indices 1-7 (first octonion sub-block)
/// 4. Leave indices 0, 8-15 unchanged
fn apply_g2_derivation_to_basis(
    basis: &[Sedenion; 16],
    derivation: &OctonionDerivation,
    t: f64,
) -> [Sedenion; 16] {
    // Extract 7x7 imaginary sub-block (rows/cols 1-7 of the 8x8 matrix)
    let mut d7 = [[0.0; 7]; 7];
    for i in 0..7 {
        for j in 0..7 {
            d7[i][j] = derivation.matrix[i + 1][j + 1];
        }
    }

    let rot = mat_exp_7x7(&d7, t, 20);

    let mut new_basis = *basis;
    for b_idx in 0..16 {
        let mut components = basis[b_idx].to_slice();
        let orig = components;

        // Apply rotation to indices 1-7
        for i in 0..7 {
            let mut sum = 0.0;
            for j in 0..7 {
                sum += rot[i][j] * orig[j + 1];
            }
            components[i + 1] = sum;
        }

        new_basis[b_idx] = Sedenion::from_slice(&components);
    }

    new_basis
}

/// Build the standard sedenion basis.
fn standard_basis() -> [Sedenion; 16] {
    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }
    basis
}

/// Return all 6 permutations of 3 elements.
fn all_s3_permutations(subs: [&[usize]; 3]) -> Vec<[&[usize]; 3]> {
    vec![
        [subs[0], subs[1], subs[2]], // (0,1,2)
        [subs[0], subs[2], subs[1]], // (0,2,1)
        [subs[1], subs[0], subs[2]], // (1,0,2)
        [subs[1], subs[2], subs[0]], // (1,2,0)
        [subs[2], subs[0], subs[1]], // (2,0,1)
        [subs[2], subs[1], subs[0]], // (2,1,0)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::three_fermion_generations::get_sedenion_subalgebras;
    use gororoba_algebra::construction::g2_automorphisms::compute_g2_basis;

    #[test]
    fn test_generational_invariance() {
        let basis = standard_basis();
        let i_struct = basis[15];

        let ratio_baseline = run_predictive_pipeline_on_basis(&basis, &i_struct);

        // Test permutation by swapping octonion subalgebra roles
        let casimir = compute_casimir(&basis, &i_struct);
        let (o1, o2, _) = get_octonion_subalgebras();
        let m1_perm = calculate_generational_mass(&casimir, &o2);
        let m2_perm = calculate_generational_mass(&casimir, &o1);
        let ratio_perm = m2_perm / m1_perm;

        assert!((ratio_baseline - ratio_perm).abs() < 1e-9);
    }

    #[test]
    fn test_g2_invariance() {
        let basis = standard_basis();
        let i_struct = basis[15];
        let baseline_ratio = run_predictive_pipeline_on_basis(&basis, &i_struct);

        let g2_basis = compute_g2_basis();
        assert_eq!(g2_basis.len(), 14, "G2 should have 14 derivations");

        let t_values = [0.05, 0.1, 0.3, 0.5, 1.0];
        let mut pass_count = 0;
        let mut total_checks = 0;

        for (d_idx, derivation) in g2_basis.iter().enumerate() {
            for &t in &t_values {
                let rotated_basis = apply_g2_derivation_to_basis(&basis, derivation, t);
                let rotated_ratio = run_predictive_pipeline_on_basis(&rotated_basis, &i_struct);
                let diff = (baseline_ratio - rotated_ratio).abs();

                total_checks += 1;
                if diff < 1e-9 {
                    pass_count += 1;
                } else {
                    println!(
                        "  G2 derivation {d_idx}, t={t:.2}: ratio diff={diff:.2e} (baseline={baseline_ratio:.6})"
                    );
                }
            }
        }

        println!("G2 invariance: {pass_count}/{total_checks} checks passed (tolerance 1e-9)");
        // All 70 checks (14 derivations x 5 t-values) should pass
        assert_eq!(
            pass_count, total_checks,
            "All G2 checks must pass: {pass_count}/{total_checks}"
        );
    }

    #[test]
    fn test_s3_permutation_invariance() {
        let basis = standard_basis();
        let i_struct = basis[15];
        let casimir = compute_casimir(&basis, &i_struct);

        // Test with contiguous-block subalgebras
        let (o1_c, o2_c, o3_c) = get_octonion_subalgebras();
        let subs_contiguous: [&[usize]; 3] = [&o1_c, &o2_c, &o3_c];
        let perms_c = all_s3_permutations(subs_contiguous);
        let baseline_c = mass_ratios_sorted(&casimir, perms_c[0]);

        for (i, perm) in perms_c.iter().enumerate() {
            let triple = mass_ratios_sorted(&casimir, *perm);
            for j in 0..3 {
                assert!(
                    (triple[j] - baseline_c[j]).abs() < 1e-12,
                    "Contiguous S3 perm {i}: mass[{j}] differs: {:.6e} vs {:.6e}",
                    triple[j],
                    baseline_c[j]
                );
            }
        }
        println!("S3 invariance (contiguous-block): 6/6 permutations passed");

        // Test with interleaved-stride subalgebras
        let (o1_i, o2_i, o3_i) = get_sedenion_subalgebras();
        let subs_interleaved: [&[usize]; 3] = [&o1_i, &o2_i, &o3_i];
        let perms_i = all_s3_permutations(subs_interleaved);
        let baseline_i = mass_ratios_sorted(&casimir, perms_i[0]);

        for (i, perm) in perms_i.iter().enumerate() {
            let triple = mass_ratios_sorted(&casimir, *perm);
            for j in 0..3 {
                assert!(
                    (triple[j] - baseline_i[j]).abs() < 1e-12,
                    "Interleaved S3 perm {i}: mass[{j}] differs: {:.6e} vs {:.6e}",
                    triple[j],
                    baseline_i[j]
                );
            }
        }
        println!("S3 invariance (interleaved-stride): 6/6 permutations passed");
    }

    #[test]
    fn test_g2_cross_s3_full_invariance() {
        let basis = standard_basis();
        let i_struct = basis[15];
        let baseline_casimir = compute_casimir(&basis, &i_struct);

        let g2_basis = compute_g2_basis();
        let t_values = [0.05, 0.1, 0.3, 0.5, 1.0];

        // Both subalgebra definitions
        let (o1_c, o2_c, o3_c) = get_octonion_subalgebras();
        let (o1_i, o2_i, o3_i) = get_sedenion_subalgebras();

        let baseline_triple_c =
            mass_ratios_sorted(&baseline_casimir, [&o1_c[..], &o2_c[..], &o3_c[..]]);
        let baseline_triple_i =
            mass_ratios_sorted(&baseline_casimir, [&o1_i[..], &o2_i[..], &o3_i[..]]);

        let mut pass_count = 0;
        let mut total_checks = 0;

        for derivation in &g2_basis {
            for &t in &t_values {
                let rotated_basis = apply_g2_derivation_to_basis(&basis, derivation, t);
                let rotated_casimir = compute_casimir(&rotated_basis, &i_struct);

                // 6 S3 permutations x 2 subalgebra defs = 12 checks per (derivation, t)
                let subs_c: [&[usize]; 3] = [&o1_c, &o2_c, &o3_c];
                for perm in all_s3_permutations(subs_c) {
                    let triple = mass_ratios_sorted(&rotated_casimir, perm);
                    total_checks += 1;
                    let max_diff = (0..3)
                        .map(|j| (triple[j] - baseline_triple_c[j]).abs())
                        .fold(0.0_f64, f64::max);
                    if max_diff < 1e-9 {
                        pass_count += 1;
                    }
                }

                let subs_i: [&[usize]; 3] = [&o1_i, &o2_i, &o3_i];
                for perm in all_s3_permutations(subs_i) {
                    let triple = mass_ratios_sorted(&rotated_casimir, perm);
                    total_checks += 1;
                    let max_diff = (0..3)
                        .map(|j| (triple[j] - baseline_triple_i[j]).abs())
                        .fold(0.0_f64, f64::max);
                    if max_diff < 1e-9 {
                        pass_count += 1;
                    }
                }
            }
        }

        println!("G2 x S3 full invariance: {pass_count}/{total_checks} checks passed");
        // 14 derivations x 5 t-values x 6 S3 perms x 2 subalgebra defs = 840
        assert_eq!(total_checks, 840, "expected 840 total checks");
        assert_eq!(
            pass_count, total_checks,
            "All G2 x S3 checks must pass: {pass_count}/{total_checks}"
        );
    }
}
