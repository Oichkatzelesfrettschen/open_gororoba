use crate::cayley_dickson_structs::Sedenion;
use crate::quantum_state::QuantumState;

pub fn construct_annihilation_operators(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> Vec<QuantumState> {
    let i_op = complex_structure;

    let a1 = (basis[1] * -1.0 + *i_op * basis[2]) * 0.5;
    let a2 = (basis[3] * -1.0 + *i_op * basis[4]) * 0.5;
    let a3 = (basis[5] * -1.0 + *i_op * basis[6]) * 0.5;
    let a4 = (basis[8] * -1.0 + *i_op * basis[9]) * 0.5;
    let a5 = (basis[10] * -1.0 + *i_op * basis[11]) * 0.5;
    
    vec![
        QuantumState::Observable(a1),
        QuantumState::Observable(a2),
        QuantumState::Observable(a3),
        QuantumState::Observable(a4),
        QuantumState::Observable(a5),
    ]
}

pub fn construct_su5_generators_algebraic(basis: &[Sedenion; 16], complex_structure: &Sedenion) -> Vec<QuantumState> {
    let a_ops = construct_annihilation_operators(basis, complex_structure);
    let ad_ops: Vec<QuantumState> = a_ops.iter().map(|a| match a {
        QuantumState::Observable(s) => QuantumState::Observable(s.conj()),
        QuantumState::TopologicalNull => QuantumState::TopologicalNull,
    }).collect();

    let mut generators = Vec::new();

    for i in 0..5 {
        for j in 0..5 {
            if i == j { continue; }
            generators.push(ad_ops[i] * a_ops[j]);
        }
    }

    let h1 = (ad_ops[0] * a_ops[0] - ad_ops[1] * a_ops[1]) * 0.5;
    let h2 = (ad_ops[0] * a_ops[0] + ad_ops[1] * a_ops[1] - ad_ops[2] * a_ops[2] * 2.0) * (1.0 / 12.0_f64.sqrt());
    let h3 = (ad_ops[0] * a_ops[0] + ad_ops[1] * a_ops[1] + ad_ops[2] * a_ops[2] - ad_ops[3] * a_ops[3] * 3.0) * (1.0 / 24.0_f64.sqrt());
    let h4 = (ad_ops[0] * a_ops[0] + ad_ops[1] * a_ops[1] + ad_ops[2] * a_ops[2] + ad_ops[3] * a_ops[3] - ad_ops[4] * a_ops[4] * 4.0) * (1.0 / 40.0_f64.sqrt());
    
    generators.push(h1);
    generators.push(h2);
    generators.push(h3);
    generators.push(h4);

    generators
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algebraic_su5_generator_construction() {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        let i_struct = basis[15];

        let su5_gens = construct_su5_generators_algebraic(&basis, &i_struct);
        assert_eq!(su5_gens.len(), 24);
        let null_count = su5_gens.iter().filter(|g| **g == QuantumState::TopologicalNull).count();
        println!("Number of null generators: {}", null_count);
        assert!(null_count > 0);
    }
}
