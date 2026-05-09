//! Non-associative algebras: Malcev, Bol, Lie-admissible.
//!
//! This module hosts the `NonAssociativeAlgebra` trait used to test identities
//! (commutator anticommutativity, Jacobi, Malcev) on candidate algebras.
//!
//! # Consolidation note (2026-05-08)
//!
//! Earlier revisions of this file contained two stub types --
//! `FreudenthalTitsMagicSquare` (string-based) and `E8RootSystem` (trivial
//! placeholder) -- that collided with the canonical typed versions in
//! [`crate::lie::e8::magic_square`] and [`crate::lie::e8::root_system`]. Both
//! stubs had no external consumers and have been removed; use the canonical
//! `lie::e8::*` types instead.

use std::fmt;

/// Trait for non-associative algebras in the construction-method hierarchy.
///
/// Implementors satisfy:
/// - Level 1 = Mechanism (anticommutative bracket).
/// - Level 2 = Dimension.
/// - Level 3 = Composition parameters.
pub trait NonAssociativeAlgebra: Clone + fmt::Debug {
    /// Dimension of the algebra.
    fn dim(&self) -> usize;

    /// Mechanism-specific product.
    fn product(&self, a: &[f64], b: &[f64]) -> Vec<f64>;

    /// Commutator `[a, b] = ab - ba`.
    fn commutator(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let ab = self.product(a, b);
        let ba = self.product(b, a);
        ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
    }

    /// True iff `[a, b]` vanishes within numerical tolerance.
    fn commutes(&self, a: &[f64], b: &[f64]) -> bool {
        let comm = self.commutator(a, b);
        comm.iter().all(|x| x.abs() < 1e-10)
    }

    /// `||[a, b]||` -- non-commutativity magnitude.
    fn non_commutativity_violation(&self, a: &[f64], b: &[f64]) -> f64 {
        let comm = self.commutator(a, b);
        comm.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Jacobi-identity violation `|| [[a,b],c] + [[b,c],a] + [[c,a],b] ||`.
    fn jacobi_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.commutator(a, b);
        let bc = self.commutator(b, c);
        let ca = self.commutator(c, a);

        let ab_c = self.commutator(&ab, c);
        let bc_a = self.commutator(&bc, a);
        let ca_b = self.commutator(&ca, b);

        let mut result = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            result[i] = ab_c[i] + bc_a[i] + ca_b[i];
        }
        result.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Malcev-identity violation
    /// `|| (xy)(xz) - (((xy)z)x + ((yz)x)x + ((zx)x)y) ||`.
    fn malcev_violation(&self, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let xy = self.product(x, y);
        let xy_xz = self.product(&xy, x);
        let xy_xz_prod = self.product(&xy_xz, z);

        let xy_z = self.product(&xy, z);
        let xy_z_x = self.product(&xy_z, x);

        let yz = self.product(y, z);
        let yz_x = self.product(&yz, x);
        let yz_x_x = self.product(&yz_x, x);

        let zx = self.product(z, x);
        let zx_x = self.product(&zx, x);
        let zx_x_y = self.product(&zx_x, y);

        let mut rhs = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            rhs[i] = xy_z_x[i] + yz_x_x[i] + zx_x_y[i];
        }

        xy_xz_prod
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }

    /// Associativity violation `|| (ab)c - a(bc) ||`.
    fn associativity_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.product(a, b);
        let ab_c = self.product(&ab, c);

        let bc = self.product(b, c);
        let a_bc = self.product(a, &bc);

        ab_c.iter()
            .zip(a_bc.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }
}
