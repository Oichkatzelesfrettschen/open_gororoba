(** * Quaternion rotation sandwich = rotation matrix, on the CDQuat type.

    The substrate's quaternion ops (QMUL, QROTATE) are grounded on CDQuat
    (CayleyDicksonAlgebra.v, fields qa qb qc qd).  C876_QuaternionRotation.v proves
    quat_rotate = matrix_rotate on the separate Quat type; this file proves the
    same equivalence directly on CDQuat, so the QROTATE catalog grounding needs no
    cross-type bridge.

    For a unit quaternion q, the sandwich q * embed(v) * conj(q) equals the vector
    R(q) v, where R(q) is the standard quaternion rotation matrix.  The real part of
    the sandwich is identically zero (the sandwich of a pure quaternion is pure for
    ANY q, since conj(q p conj q) = q (conj p) conj q = -(q p conj q) when p is
    pure), so only the three vector components carry the unit constraint. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Open Scope R_scope.

(** Unit quaternion predicate on CDQuat. *)
Definition cd_is_unit (q : CDQuat) : Prop :=
  qa q * qa q + qb q * qb q + qc q * qc q + qd q * qd q = 1.

(** Embed a 3-vector as a pure quaternion (0, vx, vy, vz). *)
Definition cd_embed (vx vy vz : R) : CDQuat := mkQuat 0 vx vy vz.

(** Rotation by the sandwich product q * embed(v) * conj(q). *)
Definition cd_quat_rotate (q : CDQuat) (vx vy vz : R) : CDQuat :=
  quat_mul (quat_mul q (cd_embed vx vy vz)) (quat_conj q).

(** The standard rotation matrix R(q) applied to (vx,vy,vz), as a pure quaternion.
    R = [[1-2(y^2+z^2), 2(xy-wz),     2(xz+wy)    ],
         [2(xy+wz),     1-2(x^2+z^2), 2(yz-wx)    ],
         [2(xz-wy),     2(yz+wx),     1-2(x^2+y^2)]]  with (w,x,y,z) = (qa,qb,qc,qd). *)
Definition cd_matrix_rotate (q : CDQuat) (vx vy vz : R) : CDQuat :=
  let w := qa q in let x := qb q in let y := qc q in let z := qd q in
  mkQuat 0
    ((1 - 2*(y*y + z*z)) * vx + 2*(x*y - w*z) * vy + 2*(x*z + w*y) * vz)
    (2*(x*y + w*z) * vx + (1 - 2*(x*x + z*z)) * vy + 2*(y*z - w*x) * vz)
    (2*(x*z - w*y) * vx + 2*(y*z + w*x) * vy + (1 - 2*(x*x + y*y)) * vz).

(** The sandwich rotation equals the matrix rotation for any unit quaternion. *)
Theorem cd_quat_rotate_eq_matrix : forall q vx vy vz,
  cd_is_unit q -> cd_quat_rotate q vx vy vz = cd_matrix_rotate q vx vy vz.
Proof.
  intros q vx vy vz Hunit.
  destruct q as [w x y z].
  unfold cd_quat_rotate, cd_matrix_rotate, cd_embed, quat_mul, quat_conj,
         cd_is_unit in *.
  simpl in *.
  f_equal;
    first [ ring | (replace 1 with (w*w + x*x + y*y + z*z) by lra; ring) ].
Qed.

(** Identity quaternion is the identity rotation. *)
Theorem cd_identity_rotation : forall vx vy vz,
  cd_quat_rotate quat_one vx vy vz = cd_embed vx vy vz.
Proof.
  intros vx vy vz.
  unfold cd_quat_rotate, cd_embed, quat_one, quat_mul, quat_conj.
  simpl.
  f_equal; ring.
Qed.
