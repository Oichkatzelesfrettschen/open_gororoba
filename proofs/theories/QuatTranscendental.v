(** * Quaternion transcendental tier: exp, log, slerp.

    The exponential / logarithm / spherical-linear-interpolation maps on the unit
    quaternion sphere, with their load-bearing identities.  These ground the
    QEXP / QLOG / SLERP virtual ops (the axis-angle <-> quaternion bridge and unit-
    quaternion interpolation), which need the transcendentals sin/cos the lower
    tiers avoid.  Built on CDQuat (CayleyDicksonAlgebra.v). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Open Scope R_scope.

(** Exponential of the pure quaternion theta * (ux,uy,uz):
    exp(theta u) = (cos theta, sin theta * u).  For a unit axis u this is the
    rotation quaternion of angle 2*theta about u. *)
Definition quat_exp_pure (theta ux uy uz : R) : CDQuat :=
  mkQuat (cos theta) (sin theta * ux) (sin theta * uy) (sin theta * uz).

(** The exponential of a unit-axis pure quaternion lands on the unit sphere:
    |exp(theta u)|^2 = cos^2 theta + sin^2 theta |u|^2 = 1.  This is the property
    QEXP relies on to produce valid rotation quaternions. *)
Theorem quat_exp_pure_unit : forall theta ux uy uz,
  ux*ux + uy*uy + uz*uz = 1 ->
  quat_norm_sq (quat_exp_pure theta ux uy uz) = 1.
Proof.
  intros theta ux uy uz Haxis.
  unfold quat_exp_pure, quat_norm_sq; simpl.
  pose proof (sin2_cos2 theta) as Hsc.
  unfold Rsqr in Hsc.
  nra.
Qed.

(** Zero angle gives the identity quaternion: exp(0) = 1. *)
Theorem quat_exp_pure_zero : forall ux uy uz,
  quat_exp_pure 0 ux uy uz = quat_one.
Proof.
  intros ux uy uz.
  unfold quat_exp_pure, quat_one.
  rewrite cos_0, sin_0.
  f_equal; ring.
Qed.

(** Logarithm of a quaternion: log(q) = acos(qa) * vec(q)/|vec(q)|, the pure
    quaternion whose exponential is q (on the unit sphere, for a nonzero angle). *)
Definition quat_log (q : CDQuat) : CDQuat :=
  let theta := acos (qa q) in
  mkQuat 0 (qb q / sin theta * theta)
           (qc q / sin theta * theta)
           (qd q / sin theta * theta).

(** Round-trip: log(exp(theta u)) recovers the pure quaternion theta u, for an
    angle in (0, PI) (where acos(cos theta) = theta and sin theta > 0). *)
Theorem quat_log_exp_roundtrip : forall theta ux uy uz,
  0 < theta < PI ->
  quat_log (quat_exp_pure theta ux uy uz) =
  mkQuat 0 (theta * ux) (theta * uy) (theta * uz).
Proof.
  intros theta ux uy uz [Hlo Hhi].
  unfold quat_log, quat_exp_pure; simpl.
  assert (Hac : acos (cos theta) = theta) by (apply acos_cos; lra).
  rewrite Hac.
  assert (Hs : sin theta <> 0) by (apply Rgt_not_eq, sin_gt_0; lra).
  f_equal; try reflexivity; field; exact Hs.
Qed.

(** Spherical linear interpolation of q0,q1 separated by angle omega:
    slerp(t) = (sin((1-t) omega) q0 + sin(t omega) q1) / sin omega. *)
Definition quat_slerp (q0 q1 : CDQuat) (omega t : R) : CDQuat :=
  quat_add (quat_scale (sin ((1 - t) * omega) / sin omega) q0)
           (quat_scale (sin (t * omega) / sin omega) q1).

(** Endpoint t = 0 returns q0. *)
Theorem quat_slerp_t0 : forall q0 q1 omega,
  sin omega <> 0 -> quat_slerp q0 q1 omega 0 = q0.
Proof.
  intros q0 q1 omega Hs.
  unfold quat_slerp.
  replace ((1 - 0) * omega) with omega by ring.
  replace (0 * omega) with 0 by ring.
  rewrite sin_0.
  destruct q0 as [a b c d]; destruct q1 as [a1 b1 c1 d1].
  unfold quat_add, quat_scale; simpl.
  f_equal; field; exact Hs.
Qed.

(** Endpoint t = 1 returns q1. *)
Theorem quat_slerp_t1 : forall q0 q1 omega,
  sin omega <> 0 -> quat_slerp q0 q1 omega 1 = q1.
Proof.
  intros q0 q1 omega Hs.
  unfold quat_slerp.
  replace ((1 - 1) * omega) with 0 by ring.
  replace (1 * omega) with omega by ring.
  rewrite sin_0.
  destruct q0 as [a b c d]; destruct q1 as [a1 b1 c1 d1].
  unfold quat_add, quat_scale; simpl.
  f_equal; field; exact Hs.
Qed.
