//! Hypercomplex Cellular Automata
//!
//! Game of Life where the state of a cell is a sedenion.
//! Survival/death depends on the alternativity of its neighborhood.

use cd_kernel::cayley_dickson::cd_multiply;

pub type CaCell = [f64; 16];

/// **Hypercomplex Update Rule**
/// A cell survives if its interaction with its environment is alternative.
/// If the environment induces non-alternativity (a topological knot), the cell dies.
pub fn update_cell(state: &CaCell, environment: &CaCell) -> CaCell {
    let env_squared: [f64; 16] = cd_multiply(environment, environment).try_into().unwrap();
    
    let left: [f64; 16] = cd_multiply(state, &env_squared).try_into().unwrap();
    let st_env: [f64; 16] = cd_multiply(state, environment).try_into().unwrap();
    let right: [f64; 16] = cd_multiply(&st_env, environment).try_into().unwrap();
    
    let mut alternativity_failure = 0.0;
    for i in 0..16 {
        alternativity_failure += (left[i] - right[i]).powi(2);
    }
    
    if alternativity_failure > 1e-5 {
        [0.0; 16] // Cell dies (overpopulation via algebraic knotting)
    } else {
        *state // Cell survives
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ca_update() {
        let cell = [1.0; 16];
        let env = [0.5; 16];
        let next_gen = update_cell(&cell, &env);
        assert_eq!(next_gen, cell); // Simple elements are alternative, so it survives
    }
}
