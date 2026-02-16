//! Generic Hypercomplex Traits.
//!
//! Provides a unified type system for Cayley-Dickson algebras (Hypercomplex numbers)
//! allowing for generic algorithms and runtime polymorphism.

/// A generalized hypercomplex number system.
pub trait Hypercomplex: Clone + Send + Sync + 'static {
    /// Dimension of the algebra (e.g., 4 for Quaternions, 8 for Octonions).
    fn dimension(&self) -> usize;

    /// Multiply two vectors in this algebra.
    fn multiply(&self, a: &[f64], b: &[f64]) -> Vec<f64>;

    /// Conjugate a vector.
    fn conjugate(&self, a: &[f64]) -> Vec<f64>;

    /// Squared norm of a vector.
    fn norm_sq(&self, a: &[f64]) -> f64;

    /// Retrieve the standard basis vector e_i.
    fn basis(&self, i: usize) -> Vec<f64>;

    /// Check if the algebra is associative.
    fn is_associative(&self) -> bool;

    /// Check if the algebra is alternative.
    fn is_alternative(&self) -> bool;
}
