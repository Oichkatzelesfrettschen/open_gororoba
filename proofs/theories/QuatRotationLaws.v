(** * Load-bearing algebra law for the RS482 vertex-transform kernel.

    The GPU fragment ALU runs matrix_rotate: a 3x3 matrix built from a
    unit quaternion, applied to a vector by nine multiply-adds. The
    quaternion sandwich product quat_rotate q v = Im(q * (0,v) * q^-1)
    is the reference semantics for "rotate v by q". The kernel is only
    correct if the two agree on unit quaternions, and if the shared
    operation preserves vector norm (a rotation must not scale v).

    Both theorems below are proved from Quaternion.v's definitions
    with no admitted step. *)

From Stdlib Require Import RNsatz.
From OpenGororoba Require Import Prelude Quaternion.

(** Normalization identity: for a unit quaternion, 1 rewrites to the
    sum of squares w^2+x^2+y^2+z^2. matrix_rotate's diagonal terms are
    written as "1 - 2*(...)"; substituting the unit constraint turns
    every entry of R(q) into a homogeneous degree-2 polynomial in
    (w,x,y,z), which is what makes the two degree-4 formulas below
    (in w,x,y,z,vx,vy,vz) into one unconditional polynomial identity
    that ring can close. *)
Lemma is_unit_one_as_sum_sq (q : Quat) :
  is_unit q -> 1 = qw q * qw q + qx q * qx q + qy q * qy q + qz q * qz q.
Proof.
  unfold is_unit. lra.
Qed.

(** Law (1): the matrix kernel agrees with the quaternion conjugation
    form for every unit quaternion and every vector. *)
Theorem quat_rotate_eq_matrix_rotate :
  forall (q : Quat) (v : Vec3),
    is_unit q -> quat_rotate q v = matrix_rotate q v.
Proof.
  intros q v Hunit.
  destruct q as [w x y z].
  destruct v as [vx0 vy0 vz0].
  unfold quat_rotate, matrix_rotate, quat_mul, quat_conj,
         embed_vec, extract_vec, is_unit in *.
  simpl in *.
  f_equal;
    replace 1 with (w * w + x * x + y * y + z * z) by lra;
    ring.
Qed.

(** Law (2): the kernel preserves vector norm on unit quaternions.
    Routed through Law (1) so the matrix form inherits the norm
    property directly from the conjugation form; nsatz closes the
    resulting degree-8 polynomial identity modulo the unit
    constraint via a Groebner basis. *)
Theorem quat_rotate_preserves_norm :
  forall (q : Quat) (v : Vec3),
    is_unit q -> vec3_norm_sq (quat_rotate q v) = vec3_norm_sq v.
Proof.
  intros q v Hunit.
  rewrite quat_rotate_eq_matrix_rotate by assumption.
  destruct q as [w x y z].
  destruct v as [vx0 vy0 vz0].
  unfold vec3_norm_sq, matrix_rotate, is_unit in *.
  simpl in *.
  nsatz.
Qed.
