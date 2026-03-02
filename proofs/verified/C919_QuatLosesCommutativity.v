(** * C-919: Commutativity fails at dim=4 (property tower: first loss).

    Quaternions are the first CD level to lose commutativity.
    Witness: i*j = k but j*i = -k. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.
From OpenGororobaVerified Require Import C907_QuatNonCommutative.

Theorem C919_quat_loses_commutativity :
  quat_mul qi qj <> quat_mul qj qi.
Proof. exact C907_quat_noncommutative. Qed.
