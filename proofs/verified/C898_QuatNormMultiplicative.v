(** * C-898: Quaternion norm is multiplicative (Hurwitz at dim=4).

    |p*q|^2 = |p|^2 * |q|^2 for all p, q in H. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C898_quat_norm_multiplicative : forall p q,
  quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q.
Proof. exact quat_norm_mul. Qed.
