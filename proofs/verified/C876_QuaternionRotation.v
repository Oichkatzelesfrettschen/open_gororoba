(** * C-876: Quaternion rotation equals matrix rotation.

    Formal proof that for any unit quaternion q and any 3D vector v:
      q * v * conj(q) = R(q) * v

    where R(q) is the standard 3x3 rotation matrix derived from q.

    This is the fundamental theorem connecting the quaternion sandwich
    product to SO(3) rotations. Both sides are degree-4 polynomial
    expressions in (w, x, y, z, vx, vy, vz) with the constraint
    w^2 + x^2 + y^2 + z^2 = 1.

    Proof strategy: unfold all definitions, use the unit constraint to
    replace 1 with w^2+x^2+y^2+z^2 in the matrix formula, then ring
    solves each component as an unconditional polynomial identity.

    Mirrors: quat_rotate_vector() and quat_to_rotation_matrix() in
    quat_rotation.rs. Test: test_rotation_composition_matches_matrix. *)

From Stdlib Require Import RNsatz.
From OpenGororoba Require Import Prelude Quaternion.

(** CLAIM C-876: Quaternion rotation equals matrix rotation for unit q.

    For all unit quaternions q and all vectors v:
      quat_rotate q v = matrix_rotate q v

    This means Im(q * (0,v) * conj(q)) produces the same result as
    multiplying v by the rotation matrix R(q). *)
(*<*c876rotation>*)
Theorem C876_quat_rotation_eq_matrix :
  forall (q : Quat) (v : Vec3),
    is_unit q ->
    quat_rotate q v = matrix_rotate q v.
Proof.
  intros q v Hunit.
  destruct q as [w x y z].
  destruct v as [vx0 vy0 vz0].
  unfold quat_rotate, matrix_rotate, quat_mul, quat_conj,
         embed_vec, extract_vec, is_unit in *.
  simpl in *.
  (* After unfolding, both sides are polynomial expressions.
     The matrix formula uses "1" which equals w^2+x^2+y^2+z^2
     by the unit constraint. Substituting makes the equation
     an unconditional polynomial identity that ring can verify. *)
  f_equal;
    replace 1 with (w * w + x * x + y * y + z * z) by lra;
    ring.
Qed.
(*</c876rotation>*)

(** Corollary: quaternion rotation preserves vector norm.
    ||q*v*conj(q)||^2 = ||v||^2 for unit q. *)
Theorem C876_rotation_preserves_norm :
  forall (q : Quat) (v : Vec3),
    is_unit q ->
    vec3_norm_sq (quat_rotate q v) = vec3_norm_sq v.
Proof.
  intros q v Hunit.
  rewrite C876_quat_rotation_eq_matrix by assumption.
  destruct q as [w x y z].
  destruct v as [vx0 vy0 vz0].
  unfold vec3_norm_sq, matrix_rotate, is_unit in *.
  simpl in *.
  (* Degree-8 polynomial identity modulo the unit constraint.
     nsatz handles this via Groebner bases. *)
  nsatz.
Qed.

(** Corollary: the identity quaternion produces no rotation. *)
Theorem C876_identity_rotation :
  forall (v : Vec3),
    quat_rotate (mkQuat 1 0 0 0) v = v.
Proof.
  intros v. destruct v as [vx0 vy0 vz0].
  unfold quat_rotate, quat_mul, quat_conj, embed_vec, extract_vec.
  simpl. f_equal; ring.
Qed.
