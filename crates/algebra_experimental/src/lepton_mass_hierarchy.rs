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

use crate::majorana_braiding::{BraidResult, MajoranaMode};
use crate::bell_inequality::{rotate_sparse, SignTableCache};

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
        if k == 0 || k == i || k == j { // Exclude identity and the braid axes
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
        if k == 0 || k == i || k == j { continue; }
        let x_sparse = [(k, 1.0)];
        let b_sparse = vec![(j, 1.0)];
        signed_sum += sign_table.sparse_associator_sum(&a_rotated, &x_sparse, &b_sparse);
    }
    signed_sum
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::majorana_braiding::map_majoranas_to_cd;
    use crate::three_fermion_generations::get_sedenion_subalgebras;

    #[test]
    fn test_lepton_mass_hierarchy_from_braiding_friction() {
        println!("--- TESTING LEPTON MASS HIERARCHY FROM BRAIDING FRICTION ---");

        let (o1, o2, o3) = get_sedenion_subalgebras();
        let modes = map_majoranas_to_cd(4, 16);
        let sign_table = SignTableCache::new(16);

        // We perform the same braid (e1, e2) in each subalgebra
        let braid_i = 0;
        let braid_j = 1;

        let friction1 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o1, &sign_table).topological_friction;
        let friction2 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o2, &sign_table).topological_friction;
        let friction3 = cd_braid_in_subalgebra(&modes[braid_i], &modes[braid_j], &o3, &sign_table).topological_friction;
        
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
        let all_modes: Vec<MajoranaMode> = (1..16_usize).map(|idx| MajoranaMode {
            gamma_index: idx - 1,
            cd_basis_index: idx,
            cd_dim: 16,
        }).collect();

        for i in 0..all_modes.len() {
            for j in (i + 1)..all_modes.len() {
                let f1 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o1, &sign_table).topological_friction;
                let f2 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o2, &sign_table).topological_friction;
                let f3 = cd_braid_in_subalgebra(&all_modes[i], &all_modes[j], &o3, &sign_table).topological_friction;

                let spread = (f1 - f2).abs().max((f2 - f3).abs()).max((f1 - f3).abs());
                if spread < 1e-9 {
                    degenerate_count += 1;
                } else {
                    breaking_count += 1;
                    // Check if this is a 1+1+1 split (all three different)
                    let is_full_split = (f1 - f2).abs() > 1e-9
                        && (f2 - f3).abs() > 1e-9
                        && (f1 - f3).abs() > 1e-9;
                    if is_full_split {
                        println!("  1+1+1 SPLIT: (e_{}, e_{}) -> f1={:.4}, f2={:.4}, f3={:.4}",
                            all_modes[i].cd_basis_index, all_modes[j].cd_basis_index, f1, f2, f3);
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
            println!("Best S3-breaking pair: (e_{}, e_{})", best_pair.0, best_pair.1);
            println!("  Frictions: f1={:.6}, f2={:.6}, f3={:.6}", best_frictions.0, best_frictions.1, best_frictions.2);
            println!("  Max spread: {:.6}", best_spread);
        } else {
            println!("NO S3-breaking pairs found -- all braid axes are flavor-universal");
        }

        // This test documents the finding, not asserts a specific outcome
        println!("Scan complete: {} degenerate + {} breaking = {} total",
            degenerate_count, breaking_count, degenerate_count + breaking_count);
    }

    /// Signed friction scan: does the orientation-sensitive observable break S2?
    #[test]
    fn test_signed_friction_scan_for_s2_breaking() {
        let (o1, o2, o3) = get_sedenion_subalgebras();
        let sign_table = SignTableCache::new(16);
        let all_modes: Vec<MajoranaMode> = (1..16_usize).map(|idx| MajoranaMode {
            gamma_index: idx - 1,
            cd_basis_index: idx,
            cd_dim: 16,
        }).collect();

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
                let is_full = (s1 - s2).abs() > 1e-9
                    && (s2 - s3).abs() > 1e-9
                    && (s1 - s3).abs() > 1e-9;
                if is_full {
                    full_split_count += 1;
                    let spread = (s1 - s2).abs().max((s2 - s3).abs()).max((s1 - s3).abs());
                    println!("  1+1+1: (e_{}, e_{}) -> s1={:.4}, s2={:.4}, s3={:.4} spread={:.4}",
                        all_modes[i].cd_basis_index, all_modes[j].cd_basis_index,
                        s1, s2, s3, spread);
                    if spread > best_ratio {
                        best_ratio = spread;
                        best_pair_signed = (all_modes[i].cd_basis_index, all_modes[j].cd_basis_index);
                        best_signed = (s1, s2, s3);
                    }
                }
            }
        }

        println!("--- SIGNED FRICTION SCAN ---");
        println!("1+1+1 full splits (signed): {}", full_split_count);
        if full_split_count > 0 {
            println!("Best 1+1+1 pair: (e_{}, e_{}) spread={:.4}",
                best_pair_signed.0, best_pair_signed.1, best_ratio);
            println!("  Signed frictions: s1={:.6}, s2={:.6}, s3={:.6}",
                best_signed.0, best_signed.1, best_signed.2);
            println!("  Signed frictions: s1={:.6}, s2={:.6}, s3={:.6}",
                best_signed.0, best_signed.1, best_signed.2);
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
        let mode_1 = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
        let mode_4 = MajoranaMode { gamma_index: 3, cd_basis_index: 4, cd_dim: 16 };

        let s1 = cd_braid_signed_friction(&mode_1, &mode_4, &o1, &sign_table);
        let s2 = cd_braid_signed_friction(&mode_1, &mode_4, &o2, &sign_table);
        let s3 = cd_braid_signed_friction(&mode_1, &mode_4, &o3, &sign_table);

        println!("--- LEPTON MASS FROM SIGNED FRICTION ---");
        println!("Pair: (e_1, e_4)");
        println!("Signed frictions: s1={:.4}, s2={:.4}, s3={:.4}", s1, s2, s3);

        // Sort by absolute value to assign to generations (lightest first)
        let mut frictions = [(s1.abs(), "O1"), (s2.abs(), "O2"), (s3.abs(), "O3")];
        frictions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        println!("\nSorted |friction|: {:.4} ({}) < {:.4} ({}) < {:.4} ({})",
            frictions[0].0, frictions[0].1,
            frictions[1].0, frictions[1].1,
            frictions[2].0, frictions[2].1);

        // Model A: mass ~ |friction| (linear)
        let linear_ratios = if frictions[0].0 > 1e-15 {
            [1.0, frictions[1].0 / frictions[0].0, frictions[2].0 / frictions[0].0]
        } else {
            [0.0, frictions[1].0, frictions[2].0]
        };
        println!("\nModel A (linear): m ~ |friction|");
        println!("  Ratios: {:.1} : {:.1} : {:.1}", linear_ratios[0], linear_ratios[1], linear_ratios[2]);
        println!("  PDG:    1.0 : 207 : 3477");

        // Model B: mass ~ exp(|friction|) (exponential amplification)
        let exp_masses: Vec<f64> = frictions.iter().map(|(f, _)| f.exp()).collect();
        let exp_min = exp_masses[0];
        let exp_ratios: Vec<f64> = exp_masses.iter().map(|m| m / exp_min).collect();
        println!("\nModel B (exponential): m ~ exp(|friction|)");
        println!("  exp values: {:.2}, {:.2}, {:.2}", exp_masses[0], exp_masses[1], exp_masses[2]);
        println!("  Ratios: {:.1} : {:.1} : {:.1}", exp_ratios[0], exp_ratios[1], exp_ratios[2]);
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
        println!("\nModel C (fitted exponential): m ~ exp({:.4} * |friction|)", alpha);
        println!("  Ratios: {:.1} : {:.1} : {:.1}", fitted_ratios[0], fitted_ratios[1], fitted_ratios[2]);
        println!("  PDG:    1.0 : 207 : 3477");
        println!("  Tau prediction: {:.1} (PDG: 3477, error: {:.1}%)",
            predicted_tau_ratio,
            ((predicted_tau_ratio - 3477.0) / 3477.0 * 100.0).abs());
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
        let all_modes: Vec<MajoranaMode> = (1..16_usize).map(|idx| MajoranaMode {
            gamma_index: idx - 1,
            cd_basis_index: idx,
            cd_dim: 16,
        }).collect();

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
        println!("  Frictions: {:.4} : {:.4} : {:.4}",
            best_frictions_composite[0], best_frictions_composite[1], best_frictions_composite[2]);
        let achieved_ratio = if best_frictions_composite[1] > 1e-9 {
            best_frictions_composite[2] / best_frictions_composite[1]
        } else { f64::INFINITY };
        println!("  f_heavy/f_mid = {:.4} (target: {:.4}, error: {:.4})",
            achieved_ratio, target_ratio, best_error);

        // Compute the implied mass hierarchy
        if best_frictions_composite[1] > 1e-9 && best_frictions_composite[0] < 1e-9 {
            // Zero lightest mode: use exp model with alpha fitted to muon
            let alpha_fit = (207.0_f64).ln() / best_frictions_composite[1];
            let m_tau_pred = (alpha_fit * best_frictions_composite[2]).exp();
            println!("  Implied masses: 1 : 207 : {:.1} (PDG: 3477, error: {:.1}%)",
                m_tau_pred, ((m_tau_pred - 3477.0) / 3477.0 * 100.0).abs());
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
        println!("  ||F||: {:.4} : {:.4} : {:.4}",
            best_vec_frictions[0], best_vec_frictions[1], best_vec_frictions[2]);
        let vec_ratio = if best_vec_frictions[1] > 1e-9 {
            best_vec_frictions[2] / best_vec_frictions[1]
        } else { f64::INFINITY };
        println!("  f_heavy/f_mid = {:.4} (target: {:.4}, error: {:.4})",
            vec_ratio, target_ratio, best_vec_error);
        if best_vec_frictions[1] > 1e-9 && best_vec_frictions[0] < 1e-9 {
            let alpha_fit = (207.0_f64).ln() / best_vec_frictions[1];
            let m_tau_pred = (alpha_fit * best_vec_frictions[2]).exp();
            println!("  Implied masses: 1 : 207 : {:.1} (PDG: 3477, error: {:.1}%)",
                m_tau_pred, ((m_tau_pred - 3477.0) / 3477.0 * 100.0).abs());
        }

        // --- 2-parameter weighted fit using RAW signed frictions ---
        // The sorted frictions are identical across all 21 pairs (all {0, 2.83, 8.49}).
        // The raw SIGNED values differ: pair (e_1,e_4) -> (0, 2.83, -8.49)
        //                               pair (e_2,e_4) -> (-8.49, 0, 2.83)
        // These are linearly independent in R^3 family space.
        // Use two pairs with different generation assignments as a basis.
        //
        // The fit objective: find w1, w2 such that
        //   exp(w1*s1_g + w2*s2_g) ~ m_g for each generation g.
        {
            // Use pair (e_1,e_4): raw signed = (0, 2.83, -8.49) for (O1, O2, O3)
            // and pair (e_2,e_4): raw signed = (-8.49, 0, 2.83)
            // These are linearly independent.
            let mode_1 = MajoranaMode { gamma_index: 0, cd_basis_index: 1, cd_dim: 16 };
            let mode_2 = MajoranaMode { gamma_index: 1, cd_basis_index: 2, cd_dim: 16 };
            let mode_4 = MajoranaMode { gamma_index: 3, cd_basis_index: 4, cd_dim: 16 };
            let subs = [&o1, &o2, &o3];

            let sel1: Vec<f64> = subs.iter().map(|s|
                cd_braid_signed_friction(&mode_1, &mode_4, s, &sign_table)).collect();
            let sel2: Vec<f64> = subs.iter().map(|s|
                cd_braid_signed_friction(&mode_2, &mode_4, s, &sign_table)).collect();

            println!("\n--- 2-PARAMETER WEIGHTED FIT (raw signed) ---");
            println!("Selector 1 (e_1,e_4): [{:.4}, {:.4}, {:.4}]", sel1[0], sel1[1], sel1[2]);
            println!("Selector 2 (e_2,e_4): [{:.4}, {:.4}, {:.4}]", sel2[0], sel2[1], sel2[2]);

            // Target: F_g = w1*sel1_g + w2*sel2_g
            // where exp(F_e) ~ 0.511, exp(F_mu) ~ 105.7, exp(F_tau) ~ 1776.9
            // Normalize: F_e = 0, F_mu = ln(207), F_tau = ln(3477)
            let log_mu = (207.0_f64).ln();
            let log_tau = (3477.0_f64).ln();

            // We need to assign generations: which O_i is electron, muon, tau?
            // Try all 6 permutations and find the best fit.
            let perms: [(usize, usize, usize); 6] = [
                (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
            ];
            let mut best_perm_error = f64::INFINITY;
            let mut best_perm_result = String::new();

            for (e_gen, mu_gen, tau_gen) in &perms {
                // F_electron = 0 (lightest), F_mu = log_mu, F_tau = log_tau
                // w1*sel1[mu_gen] + w2*sel2[mu_gen] = log_mu
                // w1*sel1[tau_gen] + w2*sel2[tau_gen] = log_tau
                let a11 = sel1[*mu_gen]; let a12 = sel2[*mu_gen];
                let a21 = sel1[*tau_gen]; let a22 = sel2[*tau_gen];
                let det = a11 * a22 - a12 * a21;
                if det.abs() < 1e-10 { continue; }

                let w1 = (log_mu * a22 - log_tau * a12) / det;
                let w2 = (a11 * log_tau - a21 * log_mu) / det;

                // Check electron constraint: w1*sel1[e_gen] + w2*sel2[e_gen] should be ~0
                let f_e = w1 * sel1[*e_gen] + w2 * sel2[*e_gen];
                let error = f_e.abs(); // closer to 0 = better

                if error < best_perm_error {
                    best_perm_error = error;
                    let f_mu = w1 * sel1[*mu_gen] + w2 * sel2[*mu_gen];
                    let f_tau = w1 * sel1[*tau_gen] + w2 * sel2[*tau_gen];
                    best_perm_result = format!(
                        "e=O{}, mu=O{}, tau=O{}: w1={:.4}, w2={:.4}\n\
                         \t  F_e={:.4} (target 0), F_mu={:.4} (target {:.4}), F_tau={:.4} (target {:.4})\n\
                         \t  Masses: {:.1} : {:.1} : {:.1} (PDG: 1 : 207 : 3477)",
                        e_gen+1, mu_gen+1, tau_gen+1, w1, w2,
                        f_e, f_mu, log_mu, f_tau, log_tau,
                        f_e.exp(), f_mu.exp(), f_tau.exp()
                    );
                }
            }
            println!("Best permutation: {}", best_perm_result);
            println!("Electron residual (should be 0): {:.6}", best_perm_error);
        }

        } else if best_frictions_composite[0] > 1e-9 {
            let alpha_fit = (207.0_f64).ln() / (best_frictions_composite[1] - best_frictions_composite[0]);
            let m_e_pred = (alpha_fit * best_frictions_composite[0]).exp();
            let m_tau_pred = (alpha_fit * best_frictions_composite[2]).exp();
            let m_mu_pred = (alpha_fit * best_frictions_composite[1]).exp();
            println!("  Implied masses: {:.1} : {:.1} : {:.1}",
                m_e_pred / m_e_pred, m_mu_pred / m_e_pred, m_tau_pred / m_e_pred);
        }
    }
}
