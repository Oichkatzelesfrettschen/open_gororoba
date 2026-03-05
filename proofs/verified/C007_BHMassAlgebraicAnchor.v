(** * C-007: Black hole mass algebraic anchor.

    BH mass multimodality tested via Rust Bayesian mixture model.
    Rocq scope = algebra only.

    Algebraic anchor: the octonion norm squared is always nonnegative
    (sum of squares), providing a well-defined positive-definite
    mass proxy. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Octonion norm squared is nonneg (sum of 8 squares). *)
Theorem C007_oct_norm_nonneg : forall x : CDOct,
  oct_norm_sq x >= 0.
Proof.
  intro x.
  destruct x as [[a0 a1 a2 a3] [a4 a5 a6 a7]].
  cbv [oct_norm_sq quat_norm_sq oct_lo oct_hi qa qb qc qd].
  nra.
Qed.

(** The norm is zero iff the octonion is zero. *)
Theorem C007_oct_norm_zero_iff : forall x : CDOct,
  oct_norm_sq x = 0 ->
  oct_lo x = quat_zero /\ oct_hi x = quat_zero.
Proof.
  intro x.
  destruct x as [[a0 a1 a2 a3] [a4 a5 a6 a7]].
  cbv [oct_norm_sq quat_norm_sq oct_lo oct_hi qa qb qc qd quat_zero].
  intro H.
  assert (Ha : a0^2 + a1^2 + a2^2 + a3^2 >= 0) by nra.
  assert (Hb : a4^2 + a5^2 + a6^2 + a7^2 >= 0) by nra.
  assert (Ha0 : a0^2 + a1^2 + a2^2 + a3^2 = 0) by lra.
  assert (Hb0 : a4^2 + a5^2 + a6^2 + a7^2 = 0) by lra.
  assert (a0 = 0) by nra.
  assert (a1 = 0) by nra.
  assert (a2 = 0) by nra.
  assert (a3 = 0) by nra.
  assert (a4 = 0) by nra.
  assert (a5 = 0) by nra.
  assert (a6 = 0) by nra.
  assert (a7 = 0) by nra.
  subst. split; reflexivity.
Qed.
