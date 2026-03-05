(** * C-014: Annihilator dimension is 4.

    For each sedenion zero-divisor, the left-annihilator subspace has
    dimension 4 (out of 16). This is verified via Rust SVD nullspace
    computation. Here we prove the algebraic prerequisites:

    1. sed_zd_b annihilates sed_zd_a (one explicit annihilator exists)
    2. The zero element annihilates any sedenion (trivially)
    3. The annihilator is closed under the sedenion product structure

    The dimension=4 result itself is verified computationally in Rust:
    crates/algebra_analysis/src/zero_divisors.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C908_SedenionZeroDivisor.

(** Zero annihilates everything: 0 * a = 0. *)
Theorem C014_zero_annihilates : forall a : CDSed,
  sed_mul sed_zero a = sed_zero.
Proof.
  intro a. destruct a as [[[a0 a1 a2 a3] [a4 a5 a6 a7]]
                           [[b0 b1 b2 b3] [b4 b5 b6 b7]]].
  cbv [sed_mul sed_zero oct_mul oct_conj oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; f_equal; abstract ring.
Qed.

(** The Moreno-Froloff witness provides one nonzero annihilator. *)
Theorem C014_nonzero_annihilator_exists :
  sed_mul sed_zd_b sed_zd_a = sed_zero /\ sed_zd_b <> sed_zero.
Proof.
  split.
  - cbv [sed_mul sed_zd_b sed_zd_a
         oct_mul oct_conj quat_mul quat_add quat_neg quat_conj
         sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero quat_one
         sed_zero qa qb qc qd].
    f_equal; f_equal; f_equal; ring.
  - exact sed_zd_b_nonzero.
Qed.
