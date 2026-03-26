//! Supersymmetry Dimension Theorem and String Theory Connections.
//!
//! This module provides rigorous algorithmic checks for the physics-algebra
//! correspondence defined by the Baez-Huerta theorem.
//!
//! Supersymmetric Yang-Mills theory and Superstring existence are intimately tied
//! to the dimensions of normed division algebras (k ∈ {1, 2, 4, 8}).

/// Normed division algebras dimensions: R (1), C (2), H (4), O (8)
pub const DIVISION_ALGEBRA_DIMS: [usize; 4] = [1, 2, 4, 8];

/// Calculates the allowed spacetime dimensions for Supersymmetric Yang-Mills (SYM)
/// with massless spinors based on the Baez-Huerta theorem:
/// d = k + 2
pub fn allowed_sym_dimensions() -> Vec<usize> {
    DIVISION_ALGEBRA_DIMS.iter().map(|&k| k + 2).collect()
}

/// Calculates the allowed spacetime dimensions for Super-2-Branes:
/// d = k + 3
pub fn allowed_super_2_brane_dimensions() -> Vec<usize> {
    DIVISION_ALGEBRA_DIMS.iter().map(|&k| k + 3).collect()
}

/// Checks if a given spacetime dimension is valid for SYM theories.
/// Valid dimensions are: 3, 4, 6, and 10 (which connects to superstring theory).
pub fn is_valid_sym_dimension(d: usize) -> bool {
    allowed_sym_dimensions().contains(&d)
}

/// Checks if a given spacetime dimension is valid for Super-2-Branes.
/// Valid dimensions are: 4, 5, 7, and 11 (which connects to M-theory and 11D supergravity).
pub fn is_valid_super_2_brane_dimension(d: usize) -> bool {
    allowed_super_2_brane_dimensions().contains(&d)
}

/// Validates the 3-\psi's rule (tri(\psi) = 0) proxy requirement for SYM.
/// The 3-\psi's rule holds if and only if a division algebra structure exists in dim = d - 2.
pub fn check_3_psi_rule_algebra_existence(d: usize) -> Result<&'static str, &'static str> {
    if d < 2 {
        return Err("Spacetime dimension too low.");
    }
    let k = d - 2;
    match k {
        1 => Ok("Real numbers (R) structure exists."),
        2 => Ok("Complex numbers (C) structure exists."),
        4 => Ok("Quaternions (H) structure exists."),
        8 => Ok("Octonions (O) structure exists."),
        16 => Err("Sedenions (S) contain zero divisors, SYM cannot be formulated."),
        32 => Err("Trigintaduonions (T) contain zero divisors, SYM cannot be formulated."),
        _ => Err("No division algebra structure exists in this dimension."),
    }
}

/// Lie 2-Supergroup Construction for String Theory.
///
/// Spacetime dimension d = k + 2 corresponds to a division algebra K of dim k.
/// This construction extends the Poincare supergroup to describe parallel
/// transport of strings.
pub struct Lie2Supergroup {
    pub spacetime_dim: usize,
    pub division_algebra_dim: usize,
}

impl Lie2Supergroup {
    pub fn new(d: usize) -> Result<Self, &'static str> {
        if !is_valid_sym_dimension(d) {
            return Err("Invalid spacetime dimension for Lie 2-Supergroup (Baez-Huerta).");
        }
        Ok(Self {
            spacetime_dim: d,
            division_algebra_dim: d - 2,
        })
    }

    /// The 10D case (octonions) relates directly to superstring theory.
    pub fn is_superstring_related(&self) -> bool {
        self.spacetime_dim == 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_sym_dimensions() {
        let valid = allowed_sym_dimensions();
        assert_eq!(valid, vec![3, 4, 6, 10]);
        assert!(is_valid_sym_dimension(10));
        assert!(!is_valid_sym_dimension(5));
    }

    #[test]
    fn test_valid_super_2_brane_dimensions() {
        let valid = allowed_super_2_brane_dimensions();
        assert_eq!(valid, vec![4, 5, 7, 11]);
        assert!(is_valid_super_2_brane_dimension(11));
        assert!(!is_valid_super_2_brane_dimension(10));
    }

    #[test]
    fn test_3_psi_rule() {
        assert!(check_3_psi_rule_algebra_existence(10).is_ok());
        assert!(check_3_psi_rule_algebra_existence(11).is_err());
    }
}
