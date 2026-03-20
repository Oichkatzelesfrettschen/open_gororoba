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
            println!("Best 1+1+1 pair: (e_{}, e_{})", best_pair_signed.0, best_pair_signed.1);
            println!("  Signed frictions: s1={:.6}, s2={:.6}, s3={:.6}",
                best_signed.0, best_signed.1, best_signed.2);
            println!("  Max/min ratio: {:.4}", best_ratio);
        } else {
            println!("NO 1+1+1 splits found with signed friction either");
        }
    }
}
