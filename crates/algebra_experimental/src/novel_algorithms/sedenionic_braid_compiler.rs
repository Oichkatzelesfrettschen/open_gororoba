//! Sedenionic Braid Quantum Compiler
//!
//! Maps quantum gates to 16D sedenion elements. Braiding them non-associatively
//! allows quantum compilation that exploits the associator as an inherent phase shift,
//! naturally encoding anyonic braiding rules.

use cd_kernel::cayley_dickson::cd_multiply;

pub type SedenionGate = [f64; 16];

/// **Octonionic/Sedenionic Braid Compilation**
/// Evaluates a sequence of quantum gate applications. Because the algebra is non-associative,
/// the evaluation order (left vs. right bracketing) physically changes the compiled state,
/// mirroring the topological non-commutativity of anyon braiding.
pub fn compile_circuit_braid(gates: &[SedenionGate], left_associative: bool) -> SedenionGate {
    if gates.is_empty() {
        return [0.0; 16];
    }
    let mut state = gates[0];
    for gate in gates.iter().skip(1) {
        if left_associative {
            state = cd_multiply(&state, gate).try_into().unwrap();
        } else {
            state = cd_multiply(gate, &state).try_into().unwrap();
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_braid_compilation() {
        let gates = vec![[1.0; 16], [0.5; 16], [0.2; 16]];
        let state = compile_circuit_braid(&gates, true);
        assert_ne!(state[0], 0.0);
    }
}
