(** * Quaternion algebra and rotation operators.

    Mirrors quat_rotation.rs: Quaternion = [w, x, y, z],
    Hamilton product via Cayley-Dickson at dim=4, rotation via
    q * v * q^{-1}, and the direct rotation matrix formula.

    Convention: q = (w, x, y, z) where w is scalar.
    Unit quaternion: w^2 + x^2 + y^2 + z^2 = 1. *)

From OpenGororoba Require Import Prelude.

(** Quaternion record. *)
Record Quat : Set := mkQuat {
  qw : R;
  qx : R;
  qy : R;
  qz : R;
}.

(** 3D vector as a triple of reals. *)
Record Vec3 : Set := mkVec3 {
  vx : R;
  vy : R;
  vz : R;
}.

(** Unit quaternion predicate. *)
Definition is_unit (q : Quat) : Prop :=
  qw q * qw q + qx q * qx q + qy q * qy q + qz q * qz q = 1.

(** Hamilton product (Cayley-Dickson multiplication at dim=4).

    (a0 + a1*i + a2*j + a3*k)(b0 + b1*i + b2*j + b3*k) =
      (a0*b0 - a1*b1 - a2*b2 - a3*b3)
    + (a0*b1 + a1*b0 + a2*b3 - a3*b2)*i
    + (a0*b2 - a1*b3 + a2*b0 + a3*b1)*j
    + (a0*b3 + a1*b2 - a2*b1 + a3*b0)*k *)
Definition quat_mul (p q : Quat) : Quat :=
  mkQuat
    (qw p * qw q - qx p * qx q - qy p * qy q - qz p * qz q)
    (qw p * qx q + qx p * qw q + qy p * qz q - qz p * qy q)
    (qw p * qy q - qx p * qz q + qy p * qw q + qz p * qx q)
    (qw p * qz q + qx p * qy q - qy p * qx q + qz p * qw q).

(** Quaternion conjugate: q* = (w, -x, -y, -z). *)
Definition quat_conj (q : Quat) : Quat :=
  mkQuat (qw q) (- qx q) (- qy q) (- qz q).

(** Embed a 3D vector as a pure quaternion (0, vx, vy, vz). *)
Definition embed_vec (v : Vec3) : Quat :=
  mkQuat 0 (vx v) (vy v) (vz v).

(** Extract the vector part of a quaternion. *)
Definition extract_vec (q : Quat) : Vec3 :=
  mkVec3 (qx q) (qy q) (qz q).

(** Quaternion rotation via sandwich product:
    v' = Im(q * embed(v) * conj(q)). *)
Definition quat_rotate (q : Quat) (v : Vec3) : Vec3 :=
  extract_vec (quat_mul (quat_mul q (embed_vec v)) (quat_conj q)).

(** Direct rotation matrix from unit quaternion.
    R(q) as per quat_to_rotation_matrix() in quat_rotation.rs:69-86.

    R = [[1-2(y^2+z^2), 2(xy-wz),     2(xz+wy)    ],
         [2(xy+wz),     1-2(x^2+z^2),  2(yz-wx)    ],
         [2(xz-wy),     2(yz+wx),      1-2(x^2+y^2)]] *)
Definition matrix_rotate (q : Quat) (v : Vec3) : Vec3 :=
  let w := qw q in
  let x := qx q in
  let y := qy q in
  let z := qz q in
  mkVec3
    ((1 - 2*(y*y + z*z)) * vx v + 2*(x*y - w*z) * vy v + 2*(x*z + w*y) * vz v)
    (2*(x*y + w*z) * vx v + (1 - 2*(x*x + z*z)) * vy v + 2*(y*z - w*x) * vz v)
    (2*(x*z - w*y) * vx v + 2*(y*z + w*x) * vy v + (1 - 2*(x*x + y*y)) * vz v).

(** Squared norm of a Vec3. *)
Definition vec3_norm_sq (v : Vec3) : R :=
  vx v * vx v + vy v * vy v + vz v * vz v.
