use crate::cayley_dickson_structs::Sedenion;
use crate::quantum_state::QuantumState;
use crate::su_n_generators::construct_su5_generators_algebraic;
use crate::neutrino_sector::{classify_generator, GeneratorType};

pub fn extract_dark_quantum_numbers() -> Vec<[f64; 4]> {
    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }
    let i_struct = basis[15];

    let su5_gens = construct_su5_generators_algebraic(&basis, &i_struct);
    
    let cartan_indices = [2, 7, 10, 23]; 
    let sm_cartans: Vec<QuantumState> = cartan_indices.iter().map(|&i| su5_gens[i]).collect();

    let mut dark_gens = Vec::new();
    for (i, generator) in su5_gens.iter().enumerate() {
        if *generator != QuantumState::TopologicalNull && classify_generator(i) == GeneratorType::Dark {
            dark_gens.push(*generator);
        }
    }

    let mut charges = Vec::new();
    for dark_generator in dark_gens {
        let mut dark_charges = [0.0; 4];
        for (i, cartan) in sm_cartans.iter().enumerate() {
            let commutator = *cartan * dark_generator - dark_generator * *cartan;
            if let (QuantumState::Observable(s), QuantumState::Observable(d)) = (commutator, dark_generator) {
                let norm_sq = d.norm_sqr();
                if norm_sq > 1e-9 {
                    let dot_product = (s * d.conj()).to_slice()[0];
                    dark_charges[i] = dot_product / norm_sq;
                }
            }
        }
        charges.push(dark_charges);
    }
    
    charges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dark_sector_sterility() {
        let charges = extract_dark_quantum_numbers();
        let all_sterile = charges.iter().all(|c| c.iter().all(|&val| val.abs() < 1e-9));
        assert!(all_sterile);
    }
}
