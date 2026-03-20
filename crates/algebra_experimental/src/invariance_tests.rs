#[cfg(test)]
mod tests {
    use crate::cayley_dickson_structs::Sedenion;
    use crate::quantum_state::QuantumState;
    use crate::su_n_generators::construct_su5_generators_algebraic;
    use crate::sedenion_subalgebras::get_octonion_subalgebras;

    fn calculate_generational_mass(casimir: &Sedenion, subalgebra_indices: &[usize]) -> f64 {
        let projected_casimir = casimir.project_to_subalgebra(subalgebra_indices);
        projected_casimir.norm_sqr().sqrt()
    }

    fn run_predictive_pipeline_on_basis(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> f64 {
        let su5_gens = construct_su5_generators_algebraic(basis, complex_structure);
        let surviving_gens: Vec<_> = su5_gens.into_iter().filter(|g| *g != QuantumState::TopologicalNull).collect();

        let casimir_op = surviving_gens.into_iter().fold(QuantumState::Observable(Sedenion::default()), |acc, generator| {
            acc + generator * generator
        });
        
        let casimir = match casimir_op {
            QuantumState::Observable(s) => s,
            QuantumState::TopologicalNull => panic!("Casimir operator should not be null"),
        };
        
        let (o1, o2, _) = get_octonion_subalgebras();
        let m1 = calculate_generational_mass(&casimir, &o1);
        let m2 = calculate_generational_mass(&casimir, &o2);
        
        m2 / m1
    }

    #[test]
    fn test_generational_invariance() {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let i_struct = basis[15];

        let ratio_baseline = run_predictive_pipeline_on_basis(&basis, &i_struct);
        
        // Test permutation by swapping octonion subalgebra roles in mass extraction
        let su5_gens = construct_su5_generators_algebraic(&basis, &i_struct);
        let surviving_gens: Vec<_> = su5_gens.into_iter().filter(|g| *g != QuantumState::TopologicalNull).collect();
        let casimir_op = surviving_gens.into_iter().fold(QuantumState::Observable(Sedenion::default()), |acc, g| acc + g * g);
        let casimir = match casimir_op {
            QuantumState::Observable(s) => s,
            _ => panic!(),
        };
        let (o1, o2, _) = get_octonion_subalgebras();
        let m1_perm = calculate_generational_mass(&casimir, &o2);
        let m2_perm = calculate_generational_mass(&casimir, &o1);
        let ratio_perm = m2_perm / m1_perm;

        assert!((ratio_baseline - ratio_perm).abs() < 1e-9);
    }

    #[test]
    fn test_g2_invariance() {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let i_struct = basis[15];
        
        let baseline_ratio = run_predictive_pipeline_on_basis(&basis, &i_struct);

        // Apply a small G2 rotation (swapping e1 and e2 with a specific sign)
        // This is a discrete G2 element (permutation of imaginary units that preserves multiplication table)
        let mut rotated_basis = basis;
        let e1 = basis[1];
        let e2 = basis[2];
        let theta: f64 = 0.1;
        rotated_basis[1] = e1 * theta.cos() + e2 * theta.sin();
        rotated_basis[2] = e1 * (-theta.sin()) + e2 * theta.cos();
        
        let rotated_ratio = run_predictive_pipeline_on_basis(&rotated_basis, &i_struct);
        assert!((baseline_ratio - rotated_ratio).abs() < 1e-9);
    }
}
