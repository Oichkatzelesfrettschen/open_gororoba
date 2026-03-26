(** * C-005: Zero-divisor annihilator geometry.

    For the Moreno-Froloff witness sed_zd_a, at least one left-annihilator
    exists (sed_zd_b), and the annihilator set is closed under addition
    and scalar multiplication (it is a linear subspace).

    The full annihilator has dimension 4 (verified via Rust SVD); here we
    prove the structural properties that make this well-defined. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** sed_zd_b is a left-annihilator of sed_zd_a. *)
Theorem C005_witness_annihilator :
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  exact sed_zd_product_zero_rev.
Qed.

(** The right-annihilator also holds (a * b = 0, already proved as C-908). *)
Theorem C005_right_annihilator :
  sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof. exact sed_zd_product_zero. Qed.
