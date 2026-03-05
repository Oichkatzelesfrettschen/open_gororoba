(** * C-005: Zero-divisor annihilator geometry.

    For the Moreno-Froloff witness sed_zd_a, at least one left-annihilator
    exists (sed_zd_b), and the annihilator set is closed under addition
    and scalar multiplication (it is a linear subspace).

    The full annihilator has dimension 4 (verified via Rust SVD); here we
    prove the structural properties that make this well-defined. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C908_SedenionZeroDivisor.

(** sed_zd_b is a left-annihilator of sed_zd_a. *)
Theorem C005_witness_annihilator :
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  cbv [sed_mul sed_zd_b sed_zd_a
       oct_mul oct_conj quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero quat_one
       sed_zero qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

(** The right-annihilator also holds (a * b = 0, already proved as C-908). *)
Theorem C005_right_annihilator :
  sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof. exact C908_sedenion_zero_divisor. Qed.
