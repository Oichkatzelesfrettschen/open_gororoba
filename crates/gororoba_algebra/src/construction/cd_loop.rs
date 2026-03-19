//! Cayley-Dickson Loop (Q_n) properties.
//!
//! Investigates the multiplicative closure of basis units { +/- e_0, ..., +/- e_{2^n-1} }.
//! Based on:
//! - Kirshtein (2011), "Automorphism groups of Cayley-Dickson loops".
//! - de Marrais (2002), "Flying Higher Than A Box-Kite".

use crate::construction::hypercomplex::AlgebraDim;

/// Properties of the Cayley-Dickson Loop Q_n.
pub struct CDLoop {
    pub n: usize,
    pub dim: usize,
}

impl CDLoop {
    pub fn new(dim_enum: AlgebraDim) -> Self {
        let dim = dim_enum.dim();
        let n = (dim as f64).log2() as usize;
        CDLoop { n, dim }
    }

    /// The order of the loop Q_n is 2 * 2^n = 2^{n+1}.
    pub fn order(&self) -> usize {
        1 << (self.n + 1)
    }

    /// The order of the automorphism group Aut(Q_n).
    /// From Kirshtein (2011):
    /// |Aut(Q_0)| = 1
    /// |Aut(Q_1)| = 2
    /// |Aut(Q_2)| = 24
    /// |Aut(Q_3)| = 1344
    /// |Aut(Q_n)| = 2^n * |GL(n, 2)| ? No, that's for octonions.
    /// Kirshtein gives a specific formula for n >= 3.
    pub fn automorphism_group_order(&self) -> Option<usize> {
        match self.n {
            0 => Some(1),
            1 => Some(2),
            2 => Some(24),
            3 => Some(1344), // |GL(3, 2)| * 2^3 = 168 * 8 = 1344.
            // For n > 3, it involves the structure of the G2 action and ZDs.
            _ => None,
        }
    }
    
    /// Returns true if the loop is Hamiltonian.
    /// Q_2 (quaternion loop) is the iconic Hamiltonian loop (quaternion group Q8).
    pub fn is_hamiltonian(&self) -> bool {
        self.n == 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_orders() {
        let q2 = CDLoop::new(AlgebraDim::Quaternion);
        assert_eq!(q2.order(), 8);
        assert_eq!(q2.automorphism_group_order(), Some(24));
        assert!(q2.is_hamiltonian());

        let q3 = CDLoop::new(AlgebraDim::Octonion);
        assert_eq!(q3.order(), 16);
        assert_eq!(q3.automorphism_group_order(), Some(1344));
        assert!(!q3.is_hamiltonian());
    }
}
