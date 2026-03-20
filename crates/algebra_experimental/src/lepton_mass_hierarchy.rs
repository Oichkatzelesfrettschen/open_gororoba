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
}
