(** * C-911: Unit quaternion rotation preserves vector norm.

    ||q*v*conj(q)||^2 = ||v||^2 for all unit q and all v.
    Already proved as C876_rotation_preserves_norm. This file
    re-exports with the canonical claim number. *)

From Stdlib Require Import RNsatz.
From OpenGororoba Require Import Prelude Quaternion.
From OpenGororobaVerified Require Import C876_QuaternionRotation.

Theorem C911_rotation_preserves_norm : forall q v,
  is_unit q -> vec3_norm_sq (quat_rotate q v) = vec3_norm_sq v.
Proof. exact C876_rotation_preserves_norm. Qed.
