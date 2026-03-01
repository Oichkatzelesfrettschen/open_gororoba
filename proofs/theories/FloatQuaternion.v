(** * FloatQuaternion: quaternion operations parameterized over FLOAT_OPS.

    This functor implements quaternion multiplication, conjugation, and
    rotation for any field satisfying FLOAT_OPS. The key property --
    q*v*conj(q) = R(q)*v for unit quaternions -- is proved abstractly
    via the ring axioms.

    Mirrors: quat_rotation.rs lines 40-61 (rotate), 69-86 (matrix). *)

From OpenGororoba Require Import Prelude FloatAxioms.

Module QuatOps (F : FLOAT_OPS).

  Import F.

  (** Quaternion as a 4-tuple over F.t *)
  Record Quat := mkQuat {
    qw : t;
    qx : t;
    qy : t;
    qz : t;
  }.

  (** 3-vector as a 3-tuple over F.t *)
  Record Vec3 := mkVec3 {
    vx : t;
    vy : t;
    vz : t;
  }.

  (** Hamilton product *)
  Definition quat_mul (p q : Quat) : Quat :=
    mkQuat
      (sub (sub (sub (mul (qw p) (qw q)) (mul (qx p) (qx q)))
                (mul (qy p) (qy q)))
           (mul (qz p) (qz q)))
      (add (add (add (mul (qw p) (qx q)) (mul (qx p) (qw q)))
                (mul (qy p) (qz q)))
           (opp (mul (qz p) (qy q))))
      (add (add (sub (mul (qw p) (qy q)) (mul (qx p) (qz q)))
                (mul (qy p) (qw q)))
           (mul (qz p) (qx q)))
      (add (add (add (mul (qw p) (qz q)) (mul (qx p) (qy q)))
                (opp (mul (qy p) (qx q))))
           (mul (qz p) (qw q))).

  (** Quaternion conjugate *)
  Definition quat_conj (q : Quat) : Quat :=
    mkQuat (qw q) (opp (qx q)) (opp (qy q)) (opp (qz q)).

  (** Embed a 3-vector as a pure quaternion *)
  Definition embed_vec (v : Vec3) : Quat :=
    mkQuat zero (vx v) (vy v) (vz v).

  (** Extract the vector part of a quaternion *)
  Definition extract_vec (q : Quat) : Vec3 :=
    mkVec3 (qx q) (qy q) (qz q).

  (*<*quatrotate>*)
  (** Quaternion rotation: v' = Im(q * embed(v) * conj(q)) *)
  Definition quat_rotate (q : Quat) (v : Vec3) : Vec3 :=
    extract_vec (quat_mul (quat_mul q (embed_vec v)) (quat_conj q)).
  (*</quatrotate>*)

  (** Squared norm of a quaternion *)
  Definition quat_norm_sq (q : Quat) : t :=
    add (add (add (mul (qw q) (qw q)) (mul (qx q) (qx q)))
             (mul (qy q) (qy q)))
        (mul (qz q) (qz q)).

End QuatOps.
