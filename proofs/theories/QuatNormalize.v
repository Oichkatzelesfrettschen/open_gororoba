(** * Quaternion normalization.

    quat_normalize q = q / |q| = q * (1 / sqrt |q|^2), the unit quaternion in the
    direction of a nonzero q.  The normalized quaternion has unit norm:

        |normalize(q)|^2 = 1   (for q with |q|^2 <> 0).

    This grounds the QNORMALIZE virtual op on the RS480 compute-as-raster
    substrate (q * RSQ(|q|^2): one DP4 for the squared norm, the US RSQ for the
    reciprocal square root, one vec4 scale). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Open Scope R_scope.

(** The squared norm of a scaled quaternion: |r * q|^2 = r^2 |q|^2. *)
Lemma quat_norm_sq_scale : forall (r : R) (q : CDQuat),
  quat_norm_sq (quat_scale r q) = r ^ 2 * quat_norm_sq q.
Proof.
  intros r [a b c d].
  unfold quat_norm_sq, quat_scale; simpl.
  ring.
Qed.

(** Normalization to the unit sphere: divide by the length sqrt(|q|^2). *)
Definition quat_normalize (q : CDQuat) : CDQuat :=
  quat_scale (/ sqrt (quat_norm_sq q)) q.

(** A normalized nonzero quaternion has unit norm. *)
Theorem quat_normalize_unit : forall q : CDQuat,
  quat_norm_sq q <> 0 ->
  quat_norm_sq (quat_normalize q) = 1.
Proof.
  intros q Hnz.
  assert (Hpos : 0 < quat_norm_sq q).
  { unfold quat_norm_sq in *. nra. }
  assert (Hss : sqrt (quat_norm_sq q) * sqrt (quat_norm_sq q) = quat_norm_sq q).
  { apply sqrt_sqrt. lra. }
  assert (Hsnz : sqrt (quat_norm_sq q) <> 0).
  { apply Rgt_not_eq, sqrt_lt_R0. exact Hpos. }
  unfold quat_normalize.
  rewrite quat_norm_sq_scale.
  replace ((/ sqrt (quat_norm_sq q)) ^ 2 * quat_norm_sq q)
    with (quat_norm_sq q / (sqrt (quat_norm_sq q) * sqrt (quat_norm_sq q)))
    by (field; exact Hsnz).
  rewrite Hss.
  field. lra.
Qed.
