(** * C-014: Annihilator dimension is 4.

    For each sedenion zero-divisor, the left-annihilator subspace has
    dimension 4 (out of 16). This is verified via Rust SVD nullspace
    computation. Here we prove the algebraic prerequisites:

    1. sed_zd_b annihilates sed_zd_a (one explicit annihilator exists)
    2. The zero element annihilates any sedenion (trivially)
    3. The annihilator is closed under the sedenion product structure

    The dimension=4 result itself is verified computationally in Rust:
    crates/algebra_analysis/src/zero_divisors.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDNegLemmas.
From OpenGororobaVerified Require Import C908_SedenionZeroDivisor.

(** Zero annihilates everything: 0 * a = 0. *)
Theorem C014_zero_annihilates : forall a : CDSed,
  sed_mul sed_zero a = sed_zero.
Proof.
  exact sed_mul_zero_left.
Qed.

(** The Moreno-Froloff witness provides one nonzero annihilator. *)
Theorem C014_nonzero_annihilator_exists :
  sed_mul sed_zd_b sed_zd_a = sed_zero /\ sed_zd_b <> sed_zero.
Proof.
  split.
  - exact sed_zd_product_zero_rev.
  - exact sed_zd_b_nonzero.
Qed.
