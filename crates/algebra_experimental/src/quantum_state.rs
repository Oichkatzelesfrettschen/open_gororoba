use crate::cayley_dickson_structs::Sedenion;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantumState {
    Observable(Sedenion),
    TopologicalNull,
}

impl Default for QuantumState {
    fn default() -> Self {
        QuantumState::Observable(Sedenion::default())
    }
}

impl Mul for QuantumState {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match (self, rhs) {
            (QuantumState::Observable(a), QuantumState::Observable(b)) => {
                if a.norm_sqr() > 1e-12 && b.norm_sqr() > 1e-12 {
                    let product = a * b;
                    if product.norm_sqr() < 1e-12 {
                        return QuantumState::TopologicalNull;
                    }
                    QuantumState::Observable(product)
                } else {
                    QuantumState::Observable(a * b)
                }
            }
            _ => QuantumState::TopologicalNull,
        }
    }
}

impl Mul<f64> for QuantumState {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        match self {
            QuantumState::Observable(s) => QuantumState::Observable(s * rhs),
            QuantumState::TopologicalNull => QuantumState::TopologicalNull,
        }
    }
}

impl Add for QuantumState {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        match (self, rhs) {
            (QuantumState::Observable(a), QuantumState::Observable(b)) => QuantumState::Observable(a + b),
            _ => QuantumState::TopologicalNull,
        }
    }
}

impl Sub for QuantumState {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        match (self, rhs) {
            (QuantumState::Observable(a), QuantumState::Observable(b)) => QuantumState::Observable(a - b),
            _ => QuantumState::TopologicalNull,
        }
    }
}

impl Neg for QuantumState {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            QuantumState::Observable(a) => QuantumState::Observable(-a),
            QuantumState::TopologicalNull => QuantumState::TopologicalNull,
        }
    }
}
