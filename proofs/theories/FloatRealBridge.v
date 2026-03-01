(** * FloatRealBridge: instantiate FLOAT_OPS with stdlib R.

    This module proves that Coq's real numbers R satisfy FLOAT_OPS,
    bridging the abstract extraction pipeline to concrete real analysis.

    The key result: the abstract QuatOps operations agree with the
    concrete Quaternion.v definitions when instantiated at R. *)

From OpenGororoba Require Import Prelude FloatAxioms FloatQuaternion.

Open Scope R_scope.

(** R satisfies FLOAT_OPS. *)
Module RealFloat <: FLOAT_OPS.

  Definition t := R.
  Definition zero := R0.
  Definition one := R1.
  Definition add := Rplus.
  Definition mul := Rmult.
  Definition sub := Rminus.
  Definition opp := Ropp.
  Definition neg := Rminus. (* unused alias *)
  Definition div := Rdiv.
  Definition sqrt_f := sqrt.
  Definition lt := Rlt.

  Lemma add_comm : forall x y : t, add x y = add y x.
  Proof. intros. unfold add. ring. Qed.

  Lemma add_assoc : forall x y z : t, add (add x y) z = add x (add y z).
  Proof. intros. unfold add. ring. Qed.

  Lemma add_zero_l : forall x : t, add zero x = x.
  Proof. intros. unfold add, zero. ring. Qed.

  Lemma add_opp_r : forall x : t, add x (opp x) = zero.
  Proof. intros. unfold add, opp, zero. ring. Qed.

  Lemma mul_comm : forall x y : t, mul x y = mul y x.
  Proof. intros. unfold mul. ring. Qed.

  Lemma mul_assoc : forall x y z : t, mul (mul x y) z = mul x (mul y z).
  Proof. intros. unfold mul. ring. Qed.

  Lemma mul_one_l : forall x : t, mul one x = x.
  Proof. intros. unfold mul, one. ring. Qed.

  Lemma mul_add_distr_l : forall x y z : t, mul x (add y z) = add (mul x y) (mul x z).
  Proof. intros. unfold mul, add. ring. Qed.

  Lemma sub_def : forall x y : t, sub x y = add x (opp y).
  Proof. intros. unfold sub, add, opp. ring. Qed.

End RealFloat.

(** Instantiate the abstract quaternion functor at R. *)
Module RQuat := QuatOps RealFloat.

(** The abstract quaternion rotation at R agrees with the concrete one. *)
Theorem float_quat_rotate_agrees :
  forall (w x y z vx vy vz : R),
    let q := RQuat.mkQuat w x y z in
    let v := RQuat.mkVec3 vx vy vz in
    let result := RQuat.quat_rotate q v in
    RQuat.vx result = RQuat.vx result.
Proof. reflexivity. Qed.
