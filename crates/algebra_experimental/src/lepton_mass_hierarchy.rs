//! Lepton Mass Hierarchy from Sedenion Subalgebra Braiding
//!
//! This module implements a computational experiment to test the hypothesis that
//! the lepton mass hierarchy emerges from the distinct topological friction of
//! Majorana braids within the three canonical octonionic subalgebras of the
//! Sedenions.
//!
//! # Physics
//!
//! - **Three Fermion Generations:** The Sedenion algebra contains three distinct
//!   octonionic subalgebras, which are hypothesized to correspond to the three
//!   fermion generations.
//! - **Topological Friction:** As demonstrated in `majorana_braiding.rs`, braiding
//!   operations in non-associative algebras accumulate a "topological friction"
//!   measured by the total associator norm.
//! - **Mass Hierarchy Hypothesis:** If the topological friction for a standard
//!   braid is different within each of the three octonionic subalgebras, this
//!   could provide a purely algebraic origin for the mass hierarchy of the
//!   electron, muon, and tau leptons.
//!
//! ## Simulation
//!
//! This executable will:
//! 1. Define the three canonical octonionic subalgebras ($O_1, O_2, O_3$).
//! 2. Define a standard Majorana braiding sequence.
//! 3. Execute the braid *within each subalgebra* by restricting the probe
//!    operators to that subalgebra.
//! 4. Compare the resulting topological friction values.
//!
//! # References
//! - Tang, Q., & Tang, J. (2023). Sedenion algebra for three lepton/quark
//!   generations and its relations to SU(5). arXiv:2308.14768.

use crate::{
    bell_inequality::{SignTableCache, rotate_sparse},
    majorana_braiding::{BraidResult, MajoranaMode},
};

pub fn cd_braid_in_subalgebra(
    mode_i: &MajoranaMode,
    mode_j: &MajoranaMode,
    subalgebra: &[usize],
    sign_table: &SignTableCache,
) -> BraidResult {
    let _dim = mode_i.cd_dim;
    let i = mode_i.cd_basis_index;
    let j = mode_j.cd_basis_index;

    // Construct sparse basis elements
    let a_sparse = vec![(i, 1.0)];

    // Braid = SO(2) rotation by pi/4 in the (e_i, e_j) subplane
    let theta = std::f64::consts::FRAC_PI_4;
    let a_rotated = rotate_sparse(&a_sparse, i, j, theta);

    // Measure associator [A_rotated, X, B] for all basis probes *within the subalgebra*
    let mut total_friction = 0.0;
    let mut max_norm = 0.0;
    let mut n_probes = 0;

    for &k in subalgebra {
        if k == 0 || k == i || k == j {
            // Exclude identity and the braid axes
            continue;
        }
        let x_sparse = [(k, 1.0)];
        let b_sparse = vec![(j, 1.0)]; // B is just e_j
        let assoc_sum = sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
        let norm = assoc_sum.abs();
        total_friction += norm;
        if norm > max_norm {
            max_norm = norm;
        }
        n_probes += 1;
    }

    BraidResult {
        parity_preserved: true,
        topological_friction: total_friction,
        fidelity: if total_friction < 1e-12 { 1.0 } else { 0.0 },
        max_associator_norm: max_norm,
        n_probes_tested: n_probes,
    }
}

/// Signed friction: sum of associator values WITHOUT taking abs().
/// This preserves orientation information that the norm-based friction discards.
pub fn cd_braid_signed_friction(
    mode_i: &MajoranaMode,
    mode_j: &MajoranaMode,
    subalgebra: &[usize],
    sign_table: &SignTableCache,
) -> f64 {
    let i = mode_i.cd_basis_index;
    let j = mode_j.cd_basis_index;
    let a_sparse = vec![(i, 1.0)];
    let theta = std::f64::consts::FRAC_PI_4;
    let a_rotated = rotate_sparse(&a_sparse, i, j, theta);
    let mut signed_sum = 0.0;
    for &k in subalgebra {
        if k == 0 || k == i || k == j {
            continue;
        }
        let x_sparse = [(k, 1.0)];
        let b_sparse = vec![(j, 1.0)];
        signed_sum += sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
    }
    signed_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        majorana_braiding::map_majoranas_to_cd, three_fermion_generations::get_sedenion_subalgebras,
    };

    #[test]
    fn test_lepton_mass_hierarchy_from_braiding_friction() {
        println!("--- TESTING LEPTON MASS HIERARCHY FROM BRAIDING FRICTION ---");

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let modes = map_majoranas_to_cd(4, 16);
        let sign_table = SignTableCache::new(16);

        // We perform the same braid (e1, e2) in each subalgebra
        let braid_i = 0;
        let braid_j = 1;

        let friction1 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o1, &sign_table)
            .topological_friction;
        let friction2 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o2, &sign_table)
            .topological_friction;
        let friction3 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o3, &sign_table)
            .topological_friction;

        println!("Topological Friction (Generation 1 - O1): {:.4}", friction1);
        println!("Topological Friction (Generation 2 - O2): {:.4}", friction2);
        println!("Topological Friction (Generation 3 - O3): {:.4}", friction3);

        // RESULT: Friction values are identical across all three subalgebras.
        // The braiding friction is symmetric under the octonion subalgebra permutation
        // used here, so this particular construction does NOT produce a mass hierarchy.
        // This is a valid null result -- the hypothesis is falsified for this basis choice.
        assert!(
            (friction1 - friction2).abs() < 1e-9 && (friction2 - friction3).abs() < 1e-9,
            "Expected equal friction (null result); got f1={friction1}, f2={friction2}, f3={friction3}"
        );
    }

    /// S3 orbit scan: test ALL braid-axis pairs (e_i, e_j) for i < j in 1..16
    /// and classify by whether the friction triple (f1, f2, f3) breaks S3.
    ///
    /// If ANY orbit gives f1 != f2 != f3, we have a candidate for mass hierarchy.
    #[test]
    fn test_s3_orbit_scan_all_braid_axes() {
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let _modes = map_majoranas_to_cd(4, 16); // kept for reference
        let sign_table = SignTableCache::new(16);

        let mut degenerate_count = 0;
        let mut breaking_count = 0;
        let mut best_spread = 0.0_f64;
        let mut best_pair = (0, 0);
        let mut best_frictions = (0.0, 0.0, 0.0);

        // Scan all C(15,2) = 105 braid-axis pairs using direct basis indices
        let all_modes: Vec<MajoranaMode> = (1..16_usize)
            .map(|idx| MajoranaMode {
                gamma_index: idx - 1,
                cd_basis_index: idx,
                cd_dim: 16,
            })
            .collect();

        for i in 0..all_modes.len() {
            for j in (i + 1)..all_modes.len() {
                let f1 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o1, &sign_table)
                    .topological_friction;
                let f2 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o2, &sign_table)
                    .topological_friction;
                let f3 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o3, &sign_table)
                    .topological_friction;

                let spread = (f1 - f2).abs().max((f2 - f3).abs()).max((f1 - f3).abs());
                if spread < 1e-9 {
                    degenerate_count += 1;
                } else {
                    breaking_count += 1;
                    // Check if this is a 1+1+1 split (all three different)
                    let is_full_split =
                        (f1 - f2).abs() > 1e-9 && (f2 - f3).abs() > 1e-9 && (f1 - f3).abs() > 1e-9;
                    if is_full_split {
                        println!(
                            "  1+1+1 SPLIT: (e_{}, e_{}) -> f1={:.4}, f2={:.4}, f3={:.4}",
                            all_modes[i].cd_basis_index, all_modes[j].cd_basis_index, f1, f2, f3
                        );
                    }
                    if spread > best_spread {
                        best_spread = spread;
                        best_pair = (all_modes[i].cd_basis_index, all_modes[j].cd_basis_index);
                        best_frictions = (f1, f2, f3);
                    }
                }
            }
        }

        println!("--- S3 ORBIT SCAN ---");
        println!("Total pairs tested: {}", degenerate_count + breaking_count);
        println!("S3-degenerate: {}", degenerate_count);
        println!("S3-BREAKING:   {}", breaking_count);
        if breaking_count > 0 {
            println!(
                "Best S3-breaking pair: (e_{}, e_{})",
                best_pair.0, best_pair.1
            );
            println!(
                "  Frictions: f1={:.6}, f2={:.6}, f3={:.6}",
                best_frictions.0, best_frictions.1, best_frictions.2
            );
            println!("  Max spread: {:.6}", best_spread);
        } else {
            println!("NO S3-breaking pairs found -- all braid axes are flavor-universal");
        }

        // This test documents the finding, not asserts a specific outcome
        println!(
            "Scan complete: {} degenerate + {} breaking = {} total",
            degenerate_count,
            breaking_count,
            degenerate_count + breaking_count
        );
    }

    /// Signed friction scan: does the orientation-sensitive observable break S2?
    #[test]
    fn test_signed_friction_scan_for_s2_breaking() {
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);
        let all_modes: Vec<MajoranaMode> = (1..16_usize)
            .map(|idx| MajoranaMode {
                gamma_index: idx - 1,
                cd_basis_index: idx,
                cd_dim: 16,
            })
            .collect();

        let mut full_split_count = 0;
        let mut best_ratio = 0.0_f64;
        let mut best_pair_signed = (0, 0);
        let mut best_signed = (0.0, 0.0, 0.0);

        for i in 0..all_modes.len() {
            for j in (i + 1)..all_modes.len() {
                let s1 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o1, &sign_table);
                let s2 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o2, &sign_table);
                let s3 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o3, &sign_table);

                // Check for 1+1+1 split (all three different)
                let is_full =
                    (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9;
                if is_full {
                    full_split_count += 1;
                    let spread = (s1 - s2).abs().max((s2 - s3).abs()).max((s1 - s3).abs());
                    println!(
                        "  1+1+1: (e_{}, e_{}) -> s1={:.4}, s2={:.4}, s3={:.4} spread={:.4}",
                        all_modes[i].cd_basis_index,
                        all_modes[j].cd_basis_index,
                        s1,
                        s2,
                        s3,
                        spread
                    );
                    if spread > best_ratio {
                        best_ratio = spread;
                        best_pair_signed =
                            (all_modes[i].cd_basis_index, all_modes[j].cd_basis_index);
                        best_signed = (s1, s2, s3);
                    }
                }
            }
        }

        println!("--- SIGNED FRICTION SCAN ---");
        println!("1+1+1 full splits (signed): {}", full_split_count);
        if full_split_count > 0 {
            println!(
                "Best 1+1+1 pair: (e_{}, e_{}) spread={:.4}",
                best_pair_signed.0, best_pair_signed.1, best_ratio
            );
            println!(
                "  Signed frictions: s1={:.6}, s2={:.6}, s3={:.6}",
                best_signed.0, best_signed.1, best_signed.2
            );
            println!(
                "  Signed frictions: s1={:.6}, s2={:.6}, s3={:.6}",
                best_signed.0, best_signed.1, best_signed.2
            );
            println!("  Max/min ratio: {:.4}", best_ratio);
        } else {
            println!("NO 1+1+1 splits found with signed friction either");
        }
    }

    /// Wire signed friction into a lepton mass matrix and compare to experiment.
    ///
    /// Experimental: m_e : m_mu : m_tau = 0.511 : 105.7 : 1776.9 MeV = 1 : 207 : 3477
    ///
    /// Hypothesis: mass ~ exp(|signed_friction|) gives exponential amplification
    /// of topological friction, converting small algebraic ratios (0:1:3) into
    /// the steep hierarchy observed in Nature.
    #[test]
    fn test_lepton_mass_from_signed_friction() {
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);

        // Use the best 1+1+1 pair: (e_1, e_4)
        let mode_1 = MajoranaMode {
            gamma_index: 0,
            cd_basis_index: 1,
            cd_dim: 16,
        };
        let mode_4 = MajoranaMode {
            gamma_index: 3,
            cd_basis_index: 4,
            cd_dim: 16,
        };

        let s1 = cd_braid_signed_friction(&mode_1, &mode_4, &o1, &sign_table);
        let s2 = cd_braid_signed_friction(&mode_1, &mode_4, &o2, &sign_table);
        let s3 = cd_braid_signed_friction(&mode_1, &mode_4, &o3, &sign_table);

        println!("--- LEPTON MASS FROM SIGNED FRICTION ---");
        println!("Pair: (e_1, e_4)");
        println!("Signed frictions: s1={:.4}, s2={:.4}, s3={:.4}", s1, s2, s3);

        // Sort by absolute value to assign to generations (lightest first)
        let mut frictions = [(s1.abs(), "O1"), (s2.abs(), "O2"), (s3.abs(), "O3")];
        frictions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!(
            "\nSorted |friction|: {:.4} ({}) < {:.4} ({}) < {:.4} ({})",
            frictions[0].0,
            frictions[0].1,
            frictions[1].0,
            frictions[1].1,
            frictions[2].0,
            frictions[2].1
        );

        // Model A: mass ~ |friction| (linear)
        let linear_ratios = if frictions[0].0 > 1e-15 {
            [
                1.0,
                frictions[1].0 / frictions[0].0,
                frictions[2].0 / frictions[0].0,
            ]
        } else {
            [0.0, frictions[1].0, frictions[2].0]
        };
        println!("\nModel A (linear): m ~ |friction|");
        println!(
            "  Ratios: {:.1} : {:.1} : {:.1}",
            linear_ratios[0], linear_ratios[1], linear_ratios[2]
        );
        println!("  PDG:    1.0 : 207 : 3477");

        // Model B: mass ~ exp(|friction|) (exponential amplification)
        let exp_masses: Vec<f64> = frictions.iter().map(|(f, _)| f.exp()).collect();
        let exp_min = exp_masses[0];
        let exp_ratios: Vec<f64> = exp_masses.iter().map(|m| m / exp_min).collect();
        println!("\nModel B (exponential): m ~ exp(|friction|)");
        println!(
            "  exp values: {:.2}, {:.2}, {:.2}",
            exp_masses[0], exp_masses[1], exp_masses[2]
        );
        println!(
            "  Ratios: {:.1} : {:.1} : {:.1}",
            exp_ratios[0], exp_ratios[1], exp_ratios[2]
        );
        println!("  PDG:    1.0 : 207 : 3477");

        // Model C: mass ~ exp(alpha * |friction|) with fitted alpha
        // Target: exp(alpha * f2) / exp(alpha * f1) = 207
        // If f1 = 0, f2 = 2.83: exp(alpha * 2.83) = 207 -> alpha = ln(207)/2.83 = 1.88
        let alpha = if frictions[1].0 > 1e-15 && frictions[0].0 < 1e-15 {
            (207.0_f64).ln() / frictions[1].0
        } else if frictions[0].0 > 1e-15 {
            (207.0_f64).ln() / (frictions[1].0 - frictions[0].0)
        } else {
            1.0
        };
        let fitted_masses: Vec<f64> = frictions.iter().map(|(f, _)| (alpha * f).exp()).collect();
        let fitted_min = fitted_masses[0];
        let fitted_ratios: Vec<f64> = fitted_masses.iter().map(|m| m / fitted_min).collect();
        let predicted_tau_ratio = fitted_ratios[2];
        println!(
            "\nModel C (fitted exponential): m ~ exp({:.4} * |friction|)",
            alpha
        );
        println!(
            "  Ratios: {:.1} : {:.1} : {:.1}",
            fitted_ratios[0], fitted_ratios[1], fitted_ratios[2]
        );
        println!("  PDG:    1.0 : 207 : 3477");
        println!(
            "  Tau prediction: {:.1} (PDG: 3477, error: {:.1}%)",
            predicted_tau_ratio,
            ((predicted_tau_ratio - 3477.0) / 3477.0 * 100.0).abs()
        );
    }

    /// Composite operator scan: find pairs of braid axes whose SUMMED signed
    /// frictions give a ratio close to 1:1.52 (the target for m_tau/m_e = 3477).
    ///
    /// The math: if exp(alpha * f_mu) = 207 and exp(alpha * f_tau) = 3477,
    /// then f_tau/f_mu = ln(3477)/ln(207) = 1.527.
    ///
    /// Single pairs give 0:1:3 ratio. Composite (sum of two pairs) may give
    /// intermediate ratios closer to 0:1:1.52.
    #[test]
    fn test_composite_operator_scan_for_lepton_ratio() {
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);
        let all_modes: Vec<MajoranaMode> = (1..16_usize)
            .map(|idx| MajoranaMode {
                gamma_index: idx - 1,
                cd_basis_index: idx,
                cd_dim: 16,
            })
            .collect();

        // Target ratio: f_heavy / f_mid = ln(3477) / ln(207) = 1.527
        let target_ratio = (3477.0_f64).ln() / (207.0_f64).ln();
        println!("Target f_heavy/f_mid ratio: {:.4}", target_ratio);

        // Collect all 1+1+1 splitting pairs with their signed friction triples
        struct SplitPair {
            i: usize,
            j: usize,
            frictions: [f64; 3], // sorted by absolute value ascending
        }
        let mut splits = Vec::new();
        for i in 0..all_modes.len() {
            for j in (i + 1)..all_modes.len() {
                let s1 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o1, &sign_table);
                let s2 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o2, &sign_table);
                let s3 = cd_braid_signed_friction(&all_modes[i], &all_modes[j], &o3, &sign_table);
                if (s1 - s2).abs() > 1e-9 && (s2 - s3).abs() > 1e-9 && (s1 - s3).abs() > 1e-9 {
                    let mut f = [s1.abs(), s2.abs(), s3.abs()];
                    f.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    splits.push(SplitPair {
                        i: all_modes[i].cd_basis_index,
                        j: all_modes[j].cd_basis_index,
                        frictions: f,
                    });
                }
            }
        }
        println!("Found {} 1+1+1 splitting pairs", splits.len());

        // Scan composites: sum of frictions from two different pairs
        let mut best_error = f64::INFINITY;
        let mut best_composite = String::new();
        let mut best_frictions_composite = [0.0; 3];

        for a in 0..splits.len() {
            for b in (a + 1)..splits.len() {
                // Composite friction: element-wise sum of absolute friction triples
                let mut cf = [0.0_f64; 3];
                for g in 0..3 {
                    cf[g] = splits[a].frictions[g] + splits[b].frictions[g];
                }
                cf.sort_by(|x, y| x.partial_cmp(y).unwrap());

                // Check ratio: cf[2] / cf[1] should be close to target_ratio
                if cf[1] > 1e-9 {
                    let ratio = cf[2] / cf[1];
                    let error = (ratio - target_ratio).abs();
                    if error < best_error {
                        best_error = error;
                        best_composite = format!(
                            "(e_{},e_{}) + (e_{},e_{})",
                            splits[a].i, splits[a].j, splits[b].i, splits[b].j
                        );
                        best_frictions_composite = cf;
                    }
                }
            }
        }

        // Also scan single pairs -- maybe some have better ratio than 1:3
        for sp in &splits {
            if sp.frictions[1] > 1e-9 {
                let ratio = sp.frictions[2] / sp.frictions[1];
                let error = (ratio - target_ratio).abs();
                if error < best_error {
                    best_error = error;
                    best_composite = format!("(e_{},e_{}) single", sp.i, sp.j);
                    best_frictions_composite = sp.frictions;
                }
            }
        }

        println!("\n--- COMPOSITE OPERATOR SCAN ---");
        println!("Best composite: {}", best_composite);
        println!(
            "  Frictions: {:.4} : {:.4} : {:.4}",
            best_frictions_composite[0], best_frictions_composite[1], best_frictions_composite[2]
        );
        let achieved_ratio = if best_frictions_composite[1] > 1e-9 {
            best_frictions_composite[2] / best_frictions_composite[1]
        } else {
            f64::INFINITY
        };
        println!(
            "  f_heavy/f_mid = {:.4} (target: {:.4}, error: {:.4})",
            achieved_ratio, target_ratio, best_error
        );

        // Compute the implied mass hierarchy
        if best_frictions_composite[1] > 1e-9 && best_frictions_composite[0] < 1e-9 {
            // Zero lightest mode: use exp model with alpha fitted to muon
            let alpha_fit = (207.0_f64).ln() / best_frictions_composite[1];
            let m_tau_pred = (alpha_fit * best_frictions_composite[2]).exp();
            println!(
                "  Implied masses: 1 : 207 : {:.1} (PDG: 3477, error: {:.1}%)",
                m_tau_pred,
                ((m_tau_pred - 3477.0) / 3477.0 * 100.0).abs()
            );
            // --- Geometric vector sum scan ---
            // Instead of |f1| + |f2|, compute the VECTOR sum of associator
            // values per generation and THEN take the norm. Orthogonal
            // components interfere, giving irrational ratios.
            let mut best_vec_error = f64::INFINITY;
            let mut best_vec_pair = String::new();
            let mut best_vec_frictions = [0.0; 3];

            for a in 0..splits.len() {
                for b in (a + 1)..splits.len() {
                    // For each generation, compute the vector sum of the two
                    // signed friction values (treated as orthogonal components)
                    let mut vf = [0.0_f64; 3];
                    for g in 0..3 {
                        let fa = splits[a].frictions[g];
                        let fb = splits[b].frictions[g];
                        // Geometric sum: treat as orthogonal components
                        vf[g] = (fa * fa + fb * fb).sqrt();
                    }
                    vf.sort_by(|x, y| x.partial_cmp(y).unwrap());

                    if vf[1] > 1e-9 {
                        let ratio = vf[2] / vf[1];
                        let error = (ratio - target_ratio).abs();
                        if error < best_vec_error {
                            best_vec_error = error;
                            best_vec_pair = format!(
                                "(e_{},e_{}) |+| (e_{},e_{})",
                                splits[a].i, splits[a].j, splits[b].i, splits[b].j
                            );
                            best_vec_frictions = vf;
                        }
                    }
                }
            }

            println!("\n--- GEOMETRIC VECTOR SUM SCAN ---");
            println!("Best vector composite: {}", best_vec_pair);
            println!(
                "  ||F||: {:.4} : {:.4} : {:.4}",
                best_vec_frictions[0], best_vec_frictions[1], best_vec_frictions[2]
            );
            let vec_ratio = if best_vec_frictions[1] > 1e-9 {
                best_vec_frictions[2] / best_vec_frictions[1]
            } else {
                f64::INFINITY
            };
            println!(
                "  f_heavy/f_mid = {:.4} (target: {:.4}, error: {:.4})",
                vec_ratio, target_ratio, best_vec_error
            );
            if best_vec_frictions[1] > 1e-9 && best_vec_frictions[0] < 1e-9 {
                let alpha_fit = (207.0_f64).ln() / best_vec_frictions[1];
                let m_tau_pred = (alpha_fit * best_vec_frictions[2]).exp();
                println!(
                    "  Implied masses: 1 : 207 : {:.1} (PDG: 3477, error: {:.1}%)",
                    m_tau_pred,
                    ((m_tau_pred - 3477.0) / 3477.0 * 100.0).abs()
                );
            }

            // --- 2-PARAMETER WEIGHTED FIT (DIFFERENCE-NORMALIZED) ---
            //
            // CRITICAL FIX: The physical observable is the mass RATIO, not absolute mass.
            // We fit DIFFERENCES of friction values:
            //   F_mu - F_e = ln(m_mu/m_e) = ln(207)
            //   F_tau - F_e = ln(m_tau/m_e) = ln(3477)
            //
            // where F_g = w1*sel1_g + w2*sel2_g for each generation g.
            //
            // This yields a 2x2 system in (w1, w2) without assuming F_e = 0.
            // Previous bug: fitting F_mu = ln(207) directly assumes F_e = 0,
            // which biases weights when the electron generation has nonzero friction.
            {
                let mode_1 = MajoranaMode {
                    gamma_index: 0,
                    cd_basis_index: 1,
                    cd_dim: 16,
                };
                let mode_2 = MajoranaMode {
                    gamma_index: 1,
                    cd_basis_index: 2,
                    cd_dim: 16,
                };
                let mode_4 = MajoranaMode {
                    gamma_index: 3,
                    cd_basis_index: 4,
                    cd_dim: 16,
                };
                let subs = [&o1, &o2, &o3];

                let sel1: Vec<f64> = subs
                    .iter()
                    .map(|s| cd_braid_signed_friction(&mode_1, &mode_4, s, &sign_table))
                    .collect();
                let sel2: Vec<f64> = subs
                    .iter()
                    .map(|s| cd_braid_signed_friction(&mode_2, &mode_4, s, &sign_table))
                    .collect();

                println!("\n--- 2-PARAMETER WEIGHTED FIT (difference-normalized) ---");
                println!(
                    "Selector 1 (e_1,e_4): [{:.4}, {:.4}, {:.4}]",
                    sel1[0], sel1[1], sel1[2]
                );
                println!(
                    "Selector 2 (e_2,e_4): [{:.4}, {:.4}, {:.4}]",
                    sel2[0], sel2[1], sel2[2]
                );

                // Target DIFFERENCES:
                let log_mu_e = (207.0_f64).ln(); // ln(m_mu / m_e)
                let log_tau_e = (3477.0_f64).ln(); // ln(m_tau / m_e)

                let perms: [(usize, usize, usize); 6] = [
                    (0, 1, 2),
                    (0, 2, 1),
                    (1, 0, 2),
                    (1, 2, 0),
                    (2, 0, 1),
                    (2, 1, 0),
                ];
                let mut best_perm_error = f64::INFINITY;
                let mut best_perm_result = String::new();
                let mut best_w = (0.0_f64, 0.0_f64);

                for (e_gen, mu_gen, tau_gen) in &perms {
                    // System (DIFFERENCES):
                    //   w1*(sel1[mu] - sel1[e]) + w2*(sel2[mu] - sel2[e]) = ln(207)
                    //   w1*(sel1[tau] - sel1[e]) + w2*(sel2[tau] - sel2[e]) = ln(3477)
                    let d_sel1_mu = sel1[*mu_gen] - sel1[*e_gen];
                    let d_sel2_mu = sel2[*mu_gen] - sel2[*e_gen];
                    let d_sel1_tau = sel1[*tau_gen] - sel1[*e_gen];
                    let d_sel2_tau = sel2[*tau_gen] - sel2[*e_gen];

                    let det = d_sel1_mu * d_sel2_tau - d_sel2_mu * d_sel1_tau;
                    if det.abs() < 1e-10 {
                        continue;
                    }

                    let w1 = (log_mu_e * d_sel2_tau - log_tau_e * d_sel2_mu) / det;
                    let w2 = (d_sel1_mu * log_tau_e - d_sel1_tau * log_mu_e) / det;

                    // Verify: compute predicted mass ratios
                    let f_e = w1 * sel1[*e_gen] + w2 * sel2[*e_gen];
                    let f_mu = w1 * sel1[*mu_gen] + w2 * sel2[*mu_gen];
                    let f_tau = w1 * sel1[*tau_gen] + w2 * sel2[*tau_gen];

                    let pred_mu_ratio = (f_mu - f_e).exp();
                    let pred_tau_ratio = (f_tau - f_e).exp();

                    // Error: deviation from exact mass ratios
                    let error = ((pred_mu_ratio - 207.0) / 207.0).abs()
                        + ((pred_tau_ratio - 3477.0) / 3477.0).abs();

                    if error < best_perm_error {
                        best_perm_error = error;
                        best_w = (w1, w2);
                        best_perm_result = format!(
                            "e=O{}, mu=O{}, tau=O{}: w1={:.6}, w2={:.6}\n\
                         \t  F_e={:.4}, F_mu={:.4}, F_tau={:.4}\n\
                         \t  Ratios: 1 : {:.1} : {:.1} (PDG: 1 : 207 : 3477)\n\
                         \t  Relative error: mu={:.2e}, tau={:.2e}",
                            e_gen + 1,
                            mu_gen + 1,
                            tau_gen + 1,
                            w1,
                            w2,
                            f_e,
                            f_mu,
                            f_tau,
                            pred_mu_ratio,
                            pred_tau_ratio,
                            ((pred_mu_ratio - 207.0) / 207.0).abs(),
                            ((pred_tau_ratio - 3477.0) / 3477.0).abs()
                        );
                    }
                }
                println!("Best permutation: {}", best_perm_result);
                println!("Total relative error: {:.6e}", best_perm_error);

                // --- Naturality analysis of weights ---
                let (w1, w2) = best_w;
                let w_sym = (w1 + w2) / 2.0;
                let w_asym = (w1 - w2) / 2.0;
                println!("\n--- WEIGHT NATURALITY ANALYSIS ---");
                println!("  w1 = {:.6}, w2 = {:.6}", w1, w2);
                println!("  w_sym  = (w1+w2)/2 = {:.6}", w_sym);
                println!("  w_asym = (w1-w2)/2 = {:.6}", w_asym);
                println!(
                    "  |w_asym/w_sym| = {:.4} (small => near-symmetric coupling)",
                    (w_asym / w_sym).abs()
                );

                // Test proximity to fundamental values
                let fundamentals = [
                    (1.0, "1"),
                    (1.0 / 2.0_f64.sqrt(), "1/sqrt(2)"),
                    (1.0 / 3.0_f64.sqrt(), "1/sqrt(3)"),
                    (2.0 / 3.0, "2/3"),
                    (std::f64::consts::FRAC_1_PI, "1/pi"),
                    (0.5, "1/2"),
                ];
                println!("  Proximity to fundamental values:");
                for (val, name) in &fundamentals {
                    println!(
                        "    |w_sym| vs {}: diff = {:.4}",
                        name,
                        (w_sym.abs() - val).abs()
                    );
                }

                // Stability: scan nearby selector pairs to check weight convergence
                println!("\n--- SELECTOR PAIR STABILITY SCAN ---");
                let mode_3 = MajoranaMode {
                    gamma_index: 2,
                    cd_basis_index: 3,
                    cd_dim: 16,
                };
                let alt_sel2: Vec<f64> = subs
                    .iter()
                    .map(|s| cd_braid_signed_friction(&mode_3, &mode_4, s, &sign_table))
                    .collect();
                println!(
                    "Alt Selector 2 (e_3,e_4): [{:.4}, {:.4}, {:.4}]",
                    alt_sel2[0], alt_sel2[1], alt_sel2[2]
                );

                // Solve same system with alt selector
                for (e_gen, mu_gen, tau_gen) in &perms {
                    let d1_mu = sel1[*mu_gen] - sel1[*e_gen];
                    let d2_mu = alt_sel2[*mu_gen] - alt_sel2[*e_gen];
                    let d1_tau = sel1[*tau_gen] - sel1[*e_gen];
                    let d2_tau = alt_sel2[*tau_gen] - alt_sel2[*e_gen];
                    let det = d1_mu * d2_tau - d2_mu * d1_tau;
                    if det.abs() < 1e-10 {
                        continue;
                    }
                    let alt_w1 = (log_mu_e * d2_tau - log_tau_e * d2_mu) / det;
                    let alt_w2 = (d1_mu * log_tau_e - d1_tau * log_mu_e) / det;
                    let f_e = alt_w1 * sel1[*e_gen] + alt_w2 * alt_sel2[*e_gen];
                    let f_mu = alt_w1 * sel1[*mu_gen] + alt_w2 * alt_sel2[*mu_gen];
                    let pred_mu = (f_mu - f_e).exp();
                    if (pred_mu - 207.0).abs() / 207.0 < 0.01 {
                        println!(
                            "  Alt(e_3,e_4) e=O{}: w1={:.6}, w2={:.6}, w_sym={:.6}",
                            e_gen + 1,
                            alt_w1,
                            alt_w2,
                            (alt_w1 + alt_w2) / 2.0
                        );
                    }
                }
            }
        } else if best_frictions_composite[0] > 1e-9 {
            let alpha_fit =
                (207.0_f64).ln() / (best_frictions_composite[1] - best_frictions_composite[0]);
            let m_e_pred = (alpha_fit * best_frictions_composite[0]).exp();
            let m_tau_pred = (alpha_fit * best_frictions_composite[2]).exp();
            let m_mu_pred = (alpha_fit * best_frictions_composite[1]).exp();
            println!(
                "  Implied masses: {:.1} : {:.1} : {:.1}",
                m_e_pred / m_e_pred,
                m_mu_pred / m_e_pred,
                m_tau_pred / m_e_pred
            );
        }
    }

    /// 3-blade zero divisor friction scan.
    ///
    /// Tests whether 3-blade (sum of 3 basis elements) zero divisors give
    /// friction quantized in sqrt(3) or sqrt(6), potentially producing the
    /// lepton mass ratio ln(3477)/ln(207) = 1.529 without free parameters.
    ///
    /// A 3-blade ZD is a = e_i + e_j + e_k such that a*b = 0 for some b != 0.
    /// This requires that the pairwise products cancel: the three terms in
    /// (e_i + e_j + e_k) * b must sum to zero.
    #[test]
    fn test_3_blade_zero_divisor_friction() {
        use rayon::prelude::*;

        let sign_table = SignTableCache::new(16);
        let (o1, o2, o3) = get_sedenion_subalgebras();

        println!("--- 3-BLADE ZERO DIVISOR FRICTION SCAN ---");

        // Target ratio for parameter-free mass hierarchy
        let target_ratio: f64 = (3477.0_f64).ln() / (207.0_f64).ln();
        println!("Target f_heavy/f_mid ratio: {:.6}", target_ratio);

        // Generate all 3-blade combinations: C(15,3) = 455 triples
        let mut triples: Vec<(usize, usize, usize)> = Vec::new();
        for i in 1..16_usize {
            for j in (i + 1)..16 {
                for k in (j + 1)..16 {
                    triples.push((i, j, k));
                }
            }
        }
        println!("Total 3-blade triples: {}", triples.len());

        // For each triple (e_i, e_j, e_k), compute braid friction in each
        // subalgebra using the SUM of the three pairwise braid frictions.
        // This is the 3-blade analogue of the 2-blade signed friction.
        let results: Vec<_> = triples
            .par_iter()
            .map(|&(i, j, k)| {
                let mi = MajoranaMode {
                    gamma_index: i - 1,
                    cd_basis_index: i,
                    cd_dim: 16,
                };
                let mj = MajoranaMode {
                    gamma_index: j - 1,
                    cd_basis_index: j,
                    cd_dim: 16,
                };
                let mk = MajoranaMode {
                    gamma_index: k - 1,
                    cd_basis_index: k,
                    cd_dim: 16,
                };

                let mut frictions = [0.0_f64; 3];
                for (g, sub) in [&o1, &o2, &o3].iter().enumerate() {
                    // 3-blade friction: sum of all three pairwise signed frictions
                    let f_ij = cd_braid_signed_friction(&mi, &mj, sub, &sign_table);
                    let f_ik = cd_braid_signed_friction(&mi, &mk, sub, &sign_table);
                    let f_jk = cd_braid_signed_friction(&mj, &mk, sub, &sign_table);
                    frictions[g] = f_ij + f_ik + f_jk;
                }

                // Check for 1+1+1 split
                let mut abs_f = [frictions[0].abs(), frictions[1].abs(), frictions[2].abs()];
                abs_f.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let is_split = (frictions[0] - frictions[1]).abs() > 1e-9
                    && (frictions[1] - frictions[2]).abs() > 1e-9
                    && (frictions[0] - frictions[2]).abs() > 1e-9;

                let ratio = if abs_f[1] > 1e-9 {
                    abs_f[2] / abs_f[1]
                } else {
                    f64::INFINITY
                };
                let error = (ratio - target_ratio).abs();

                ((i, j, k), frictions, abs_f, is_split, ratio, error)
            })
            .collect();

        // Count splits and find best ratio
        let split_count = results.iter().filter(|r| r.3).count();
        println!("3-blade 1+1+1 splits: {} / {}", split_count, triples.len());

        // Find best ratio match
        let best = results
            .iter()
            .filter(|r| r.3 && r.2[1] > 1e-9)
            .min_by(|a, b| a.5.partial_cmp(&b.5).unwrap());

        if let Some(((i, j, k), frictions, abs_f, _, ratio, error)) = best {
            println!("\nBest 3-blade triple: (e_{}, e_{}, e_{})", i, j, k);
            println!(
                "  Signed frictions: [{:.4}, {:.4}, {:.4}]",
                frictions[0], frictions[1], frictions[2]
            );
            println!(
                "  |Frictions|: [{:.4}, {:.4}, {:.4}]",
                abs_f[0], abs_f[1], abs_f[2]
            );
            println!(
                "  Ratio f_heavy/f_mid = {:.6} (target: {:.6})",
                ratio, target_ratio
            );
            println!("  Error: {:.6}", error);

            // Check for sqrt(3) or sqrt(6) quantization
            let sqrt2: f64 = 2.0_f64.sqrt();
            let sqrt3: f64 = 3.0_f64.sqrt();
            let sqrt6: f64 = 6.0_f64.sqrt();
            println!("\n  Friction value analysis:");
            for (g, f) in abs_f.iter().enumerate() {
                if *f > 1e-9 {
                    println!(
                        "    f[{}] = {:.6}, f/sqrt(2) = {:.4}, f/sqrt(3) = {:.4}, f/sqrt(6) = {:.4}",
                        g,
                        f,
                        f / sqrt2,
                        f / sqrt3,
                        f / sqrt6
                    );
                }
            }

            // Compute implied mass hierarchy
            if abs_f[0] < 1e-9 && abs_f[1] > 1e-9 {
                let alpha_fit = (207.0_f64).ln() / abs_f[1];
                let m_tau_pred = (alpha_fit * abs_f[2]).exp();
                println!(
                    "\n  3-blade mass hierarchy: 1 : 207 : {:.1} (PDG: 3477, error: {:.1}%)",
                    m_tau_pred,
                    ((m_tau_pred - 3477.0) / 3477.0 * 100.0).abs()
                );
            }
        } else {
            println!("No 3-blade triples with 1+1+1 split found.");
        }

        // Print top-5
        let mut sorted: Vec<_> = results.iter().filter(|r| r.3 && r.2[1] > 1e-9).collect();
        sorted.sort_by(|a, b| a.5.partial_cmp(&b.5).unwrap());
        println!("\n--- TOP-5 3-BLADE TRIPLES ---");
        for (rank, ((i, j, k), _, abs_f, _, ratio, _)) in sorted.iter().take(5).enumerate() {
            println!(
                "  #{}: (e_{},e_{},e_{}) | |f|=[{:.2},{:.2},{:.2}] | ratio={:.4}",
                rank + 1,
                i,
                j,
                k,
                abs_f[0],
                abs_f[1],
                abs_f[2],
                ratio
            );
        }

        // Unique friction values: check for sqrt quantization
        let mut unique_vals: Vec<f64> = results
            .iter()
            .flat_map(|r| r.2.iter().copied())
            .filter(|&v| v > 1e-9)
            .collect();
        unique_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        println!(
            "\n  Unique nonzero |friction| values ({}):",
            unique_vals.len()
        );
        for v in &unique_vals {
            println!(
                "    {:.6} = {:.4}*sqrt(2) = {:.4}*sqrt(3) = {:.4}*sqrt(6)",
                v,
                v / 2.0_f64.sqrt(),
                v / 3.0_f64.sqrt(),
                v / 6.0_f64.sqrt()
            );
        }
    }
}
