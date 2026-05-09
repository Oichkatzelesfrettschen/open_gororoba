//! Lie algebras and exceptional structure for `gororoba_algebra`.
//!
//! # Layout
//!
//! Each exceptional Lie algebra has its own subdirectory; classical and
//! Kac-Moody machinery has dedicated modules.
//!
//! ## Exceptional algebra summary
//!
//! All five exceptional simple Lie algebras live here. Numbers in the
//! "branch" column are 0-indexed Dynkin nodes in the convention used by the
//! corresponding `*_simple_roots()` and `*_cartan_matrix()` functions.
//!
//! | Module | Rank | Dim | #Roots | \|W\| | Branch (0-idx) | Simply-laced? | Casimir of fund. rep |
//! |--------|------|-----|--------|------|----------------|---------------|----------------------|
//! | [`g2`] | 2 | 14 | 12 | 12 | -- (rank 2) | no (triple bond) | -- |
//! | [`f4`] | 4 | 52 | 48 | 1152 | -- (chain) | no (double bond at 1-2) | C2(26) = 1/4 (physics norm) |
//! | [`e6`] | 6 | 78 | 72 | 51840 | 2 (= beta_3) | yes | C2(27) = 26/3 (physics norm) |
//! | [`e7`] | 7 | 133 | 126 | 2903040 | 3 (= beta_4) | yes | -- |
//! | [`e8`] | 8 | 248 | 240 | 696729600 | 4 (= alpha_4) | yes | -- |
//!
//! ## Magic-square placement
//!
//! [`e8::magic_square`] hosts the 4x4 Freudenthal-Tits matrix
//! `(R, C, H, O) x (R, C, H, O) -> (A_1, A_2, A_2 x A_2, A_5, C_3, D_6, F_4, E_6, E_7, E_8)`.
//! The octonionic row alone yields all four exceptionals in this module:
//! `O x R = F_4`, `O x C = E_6`, `O x H = E_7`, `O x O = E_8`.
//!
//! ## Infinite-dimensional E-series
//!
//! The Kac-Moody extensions `E_9 = E_8^{(1)}` (affine), `E_10` (hyperbolic),
//! `E_11` (extended hyperbolic) live in [`kac_moody::e_series`], and reuse
//! the branch-at-node-4 numbering of [`e8::root_system`]. Their generalized
//! Cartan matrices are produced by [`kac_moody::cartans`].
//!
//! ## Other contents
//!
//! - [`group_theory`] -- standalone group-theoretic helpers (`PSL(2, q)`
//!   orders, exceptional-group dimensions).
//! - [`lyndon_basis`] -- Chevalley-Tits extraspecial 2-cocycle for `E_7`,
//!   verified against the Jacobi identity.
//! - [`nilpotent_orbits`] -- Jordan-type classification of nilpotent
//!   matrices, used by `su5_gut`.
//! - [`su5_gut`] -- `SU(5)` Grand Unified Theory generators
//!   (gated on the `physics-sm` feature).
//! - [`three_fermion_generations`] -- sedenion subalgebra structure
//!   underlying the three-generation mystery.
//!
//! ## Rocq cross-checks
//!
//! For each of `E_6, E_7, E_8, F_4`, a Rocq proof in `proofs/theories/`
//! verifies that the Cartan matrix derives from the simple roots over exact
//! rationals (`vm_compute. reflexivity.`). See `E6CartanDerivation.v`,
//! `E7CartanDerivation.v`, `E8CartanDerivation.v`, `F4CartanDerivation.v`.

pub mod e6;
pub mod e7;
pub mod e8;
pub mod f4;
pub mod g2;
pub mod group_theory;
pub mod kac_moody;
pub mod lyndon_basis;
pub mod nilpotent_orbits;
// three_fermion_generations depends only on cd_kernel + rand -- no heavy deps.
// It must be always available because lepton_mass_hierarchy (algebra_experimental)
// imports get_sedenion_subalgebras from it regardless of feature state.
#[cfg(feature = "physics-sm")]
pub mod su5_gut;
pub mod three_fermion_generations;
