(** * ThreeBodyAngMom: Angular momentum decomposition for three-body embedding.

    Proves that the angular momentum of a spacecraft relative to body B
    decomposes linearly via cross-product distributivity:

      (r - r_B) x v = r x v - r_B x v

    This is the foundational identity used in the 3-body 64D Chingon drag
    (Sprint 71.2), where Earth, Moon, and Sun each get their own angular
    momentum block in the 64D state vector.

    Mirrors: gr_core/src/forces/chingon_bivector_drag.rs
    Claims: C-954 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** 3D vector as a record of three reals. *)
Record Vec3 := mkVec3 {
  vx : R;
  vy : R;
  vz : R;
}.

(** Vector subtraction. *)
Definition vsub (a b : Vec3) : Vec3 :=
  mkVec3 (vx a - vx b) (vy a - vy b) (vz a - vz b).

(** Cross product: a x b. *)
Definition cross (a b : Vec3) : Vec3 :=
  mkVec3
    (vy a * vz b - vz a * vy b)
    (vz a * vx b - vx a * vz b)
    (vx a * vy b - vy a * vx b).

(** THEOREM: Cross product distributes over vector subtraction.
    (r - r_B) x v = r x v - r_B x v *)
Theorem cross_sub_distributes :
  forall r r_B v : Vec3,
    cross (vsub r r_B) v = vsub (cross r v) (cross r_B v).
Proof.
  intros r r_B v.
  destruct r as [rx ry rz].
  destruct r_B as [bx by_ bz].
  destruct v as [vvx vvy vvz].
  unfold cross, vsub; simpl.
  f_equal; ring.
Qed.

(** COROLLARY: Three-body angular momentum decomposition.
    h_lunar = h_earth - r_moon x v, where
    h_earth = r x v and h_lunar = (r - r_moon) x v *)
Corollary three_body_ang_mom_decomposition :
  forall r r_moon v : Vec3,
    cross (vsub r r_moon) v =
    vsub (cross r v) (cross r_moon v).
Proof.
  exact cross_sub_distributes.
Qed.

(** THEOREM: Cross product is bilinear (right distributivity).
    a x (b + c) = a x b + a x c *)
Definition vadd (a b : Vec3) : Vec3 :=
  mkVec3 (vx a + vx b) (vy a + vy b) (vz a + vz b).

Theorem cross_right_distributes :
  forall a b c : Vec3,
    cross a (vadd b c) = vadd (cross a b) (cross a c).
Proof.
  intros a b c.
  destruct a, b, c.
  unfold cross, vadd; simpl.
  f_equal; ring.
Qed.

(** THEOREM: Cross product is anti-commutative.
    a x b = -(b x a) *)
Definition vneg (a : Vec3) : Vec3 :=
  mkVec3 (- vx a) (- vy a) (- vz a).

Theorem cross_anticommutative :
  forall a b : Vec3,
    cross a b = vneg (cross b a).
Proof.
  intros a b.
  destruct a, b.
  unfold cross, vneg; simpl.
  f_equal; ring.
Qed.
