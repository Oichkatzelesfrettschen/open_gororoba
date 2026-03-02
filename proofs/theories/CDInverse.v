(** * CDInverse: Two-sided inverse and division structure at dim=4.

    Quaternion inverse: q^{-1} = conj(q) / |q|^2.
    Requires norm_sq <> 0 (i.e. q <> 0).

    Mirrors: algebra_core/src/construction/cayley_dickson.rs (inverse). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

(** Quaternion inverse: conj(q) / |q|^2. *)
Definition quat_inv (q : CDQuat) (Hnz : quat_norm_sq q <> 0) : CDQuat :=
  quat_scale (/ quat_norm_sq q) (quat_conj q).

(** Right inverse: q * q^{-1} = 1. *)
Theorem quat_mul_inv_r : forall q (Hnz : quat_norm_sq q <> 0),
  quat_mul q (quat_inv q Hnz) = quat_one.
Proof.
  intros q Hnz. destruct q as [a b c d].
  unfold quat_inv, quat_mul, quat_scale, quat_conj, quat_one, quat_norm_sq in *.
  simpl.
  f_equal; field.
  all: simpl in Hnz; lra || assumption.
Qed.

(** Left inverse: q^{-1} * q = 1. *)
Theorem quat_mul_inv_l : forall q (Hnz : quat_norm_sq q <> 0),
  quat_mul (quat_inv q Hnz) q = quat_one.
Proof.
  intros q Hnz. destruct q as [a b c d].
  unfold quat_inv, quat_mul, quat_scale, quat_conj, quat_one, quat_norm_sq in *.
  simpl.
  f_equal; field.
  all: simpl in Hnz; lra || assumption.
Qed.
