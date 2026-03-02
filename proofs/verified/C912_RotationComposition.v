(** * C-912: Rotation composition equals product rotation.

    quat_rotate p (quat_rotate q v) = quat_rotate (p*q) v
    for ALL quaternions p, q (no unit constraint needed).

    Proof: direct polynomial identity in 11 variables. Each component
    is a degree-6 polynomial identity that ring can verify. *)

From OpenGororoba Require Import Prelude Quaternion.

Theorem C912_rotation_composition : forall p q v,
  quat_rotate p (quat_rotate q v) = quat_rotate (quat_mul p q) v.
Proof.
  intros p q v.
  destruct p as [pw px py pz].
  destruct q as [qw0 qx0 qy0 qz0].
  destruct v as [vx0 vy0 vz0].
  unfold quat_rotate, quat_mul, quat_conj, embed_vec, extract_vec.
  simpl.
  f_equal; ring.
Qed.
