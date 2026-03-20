#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorType {
    SU3,
    SU2,
    U1,
    Leptoquark,
    Dark,
}

pub fn classify_generator(gen_index: usize) -> GeneratorType {
    match gen_index {
        0..=7 => GeneratorType::SU3,
        8..=10 => GeneratorType::SU2,
        11 => GeneratorType::U1,
        12..=23 => GeneratorType::Leptoquark,
        _ => GeneratorType::Dark,
    }
}

#[cfg(test)]
mod tests {
    use crate::cayley_dickson_structs::Sedenion;
    use crate::quantum_state::QuantumState;
    use crate::su_n_generators::construct_su5_generators_algebraic;
    use faer::{Mat, Side};

    #[test]
    fn test_neutrino_mass_matrix_and_see_saw() {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let i_struct = basis[15];

        let su5_gens = construct_su5_generators_algebraic(&basis, &i_struct);
        let mut dark_gens = Vec::new();
        for generator in su5_gens.iter() {
            if *generator != QuantumState::TopologicalNull {
                // In our current simple classification, we need a way to isolate the 4 dark ones.
                // Let's just take the last 4 non-null ones for this test.
                dark_gens.push(*generator);
            }
        }
        
        let dark_gens = &dark_gens[dark_gens.len()-4..];
        
        let mut mass_matrix = Mat::<f64>::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                if let (QuantumState::Observable(g1), QuantumState::Observable(g2)) = (dark_gens[i], dark_gens[j]) {
                    let product = g1.conj() * g2;
                    mass_matrix.write(i, j, product.to_slice()[0]);
                }
            }
        }
        
        println!("Full 4x4 Neutrino Mass Matrix:\n{:?}", mass_matrix);

        let m_r = mass_matrix.read(3,3);
        let m_d = mass_matrix.get(0..3, 3..4);
        let m_light = m_d * m_d.transpose() * (1.0 / m_r);

        let eig = m_light.selfadjoint_eigendecomposition(Side::Lower);
        println!("Light Neutrino Mass Eigenvalues (squared):\n{:?}", eig.s());
    }
}
