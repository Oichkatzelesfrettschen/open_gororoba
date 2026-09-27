// SPDX-License-Identifier: MIT

//! Explicit completeness declarations for finite Fourier inputs.

use crate::interval::{Rational, int};

/// Missing modes vanish only when the datum is a complete polynomial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailDeclaration {
    CompletePolynomial,
    Unknown,
}

impl TailDeclaration {
    /// None cannot be promoted to zero by the verifier.
    pub fn h3_upper(self) -> Option<Rational> {
        match self {
            Self::CompletePolynomial => Some(int(0)),
            Self::Unknown => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tail_never_becomes_zero() {
        assert_eq!(TailDeclaration::Unknown.h3_upper(), None);
        assert_eq!(TailDeclaration::CompletePolynomial.h3_upper(), Some(int(0)));
    }
}
