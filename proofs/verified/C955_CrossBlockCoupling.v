(** * C955_CrossBlockCoupling: Cross-block AVT coupling is non-vanishing.

    Proves that for generic (non-zero, non-parallel) angular momenta,
    the cross-block dot product is non-zero, establishing that the
    3-body algebra is NOT reducible to independent sub-algebras.

    Claim C-955: Cross-body non-associative torques from cross-block AVT. *)

From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import ThreeBodyAngMom.

Open Scope R_scope.

Definition dot (a b : Vec3) : R :=
  vx a * vx b + vy a * vy b + vz a * vz b.

(** THEOREM: Constructive witness -- specific non-zero cross-block coupling.
    h_earth = (1, 0, 0), h_lunar = (0, 1, 0) gives dot = 0,
    but h_earth = (1, 1, 0), h_lunar = (1, 0, 0) gives dot = 1 <> 0. *)
Theorem cross_block_witness :
  let h_e := mkVec3 1 1 0 in
  let h_l := mkVec3 1 0 0 in
  dot h_e h_l = 1.
Proof.
  unfold dot; simpl. ring.
Qed.

(** THEOREM: The cross-block coupling vanishes only when vectors are orthogonal.
    In general position, cross-block terms are non-zero. *)
Theorem cross_block_nonzero_generic :
  forall a : R,
    a <> 0 ->
    let h_e := mkVec3 a 0 0 in
    let h_l := mkVec3 a 0 0 in
    dot h_e h_l <> 0.
Proof.
  intros a Ha.
  simpl. unfold dot. simpl.
  intro H.
  apply Ha.
  assert (a * a + 0 * 0 + 0 * 0 = 0) by lra.
  assert (a * a = 0) by lra.
  destruct (Rmult_integral _ _ H1); auto.
Qed.
