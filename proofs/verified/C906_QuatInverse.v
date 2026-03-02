(** * C-906: Quaternion two-sided inverse.

    q * conj(q)/|q|^2 = 1 and conj(q)/|q|^2 * q = 1
    for all nonzero quaternions. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra CDInverse.

Theorem C906_quat_inverse_right : forall q (Hnz : quat_norm_sq q <> 0),
  quat_mul q (quat_inv q Hnz) = quat_one.
Proof. exact quat_mul_inv_r. Qed.

Theorem C906_quat_inverse_left : forall q (Hnz : quat_norm_sq q <> 0),
  quat_mul (quat_inv q Hnz) q = quat_one.
Proof. exact quat_mul_inv_l. Qed.
