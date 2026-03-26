(** * C1542_MorVlambdaMod4: Moreno (1997) Theorem 1.16 -- canonical octonion V_lambda profile.

    Moreno's Theorem 1.16 says that for the eigenspaces V_lambda from
    Theorem 1.15 one has

      dim_R V_lambda = 0 mod 4

    for every lambda > 0.

    This file formalizes the canonical octonion witness used by the Moreno
    paper lane:

      a := e1 in A_3.

    In this concrete lane the eigenspace picture is completely explicit:
    - V_1 = H_a^perp = span{e2, e3, e6, e7}
    - V_lambda = {0} for every lambda > 0 with lambda <> 1

    so every canonical V_lambda block has dimension either 4 or 0, hence
    is automatically 0 mod 4.

    This is a dedicated paper-scoped canonical companion for C-1542.
    The full arbitrary-dimensional theorem remains separate. *)

From Stdlib Require Import Reals Lia Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDNegLemmas C1540_MorHaQuaternion C1541_MorDirectSum.

Open Scope R_scope.

Definition moreno16_v1_basis (i : nat) : CDOct :=
  match i with
  | 0 => oct_e 2
  | 1 => oct_e 3
  | 2 => oct_e 6
  | 3 => oct_e 7
  | _ => oct_zero
  end.

Definition moreno16_v1_coords (r2 r3 r6 r7 : R) : CDOct :=
  mkOct (mkQuat 0 0 r2 r3) (mkQuat 0 0 r6 r7).

Definition moreno16_v1_sum (r2 r3 r6 r7 : R) : CDOct :=
  oct_add (oct_scale r2 (oct_e 2))
    (oct_add (oct_scale r3 (oct_e 3))
      (oct_add (oct_scale r6 (oct_e 6))
               (oct_scale r7 (oct_e 7)))).

Lemma moreno16_oct_scale_one : forall x : CDOct,
  oct_scale 1 x = x.
Proof.
  intros [[a b c d] [e f g h]].
  unfold oct_scale, quat_scale.
  simpl.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno16_v1_coords_eq_sum : forall r2 r3 r6 r7 : R,
  moreno16_v1_coords r2 r3 r6 r7 = moreno16_v1_sum r2 r3 r6 r7.
Proof.
  intros r2 r3 r6 r7.
  unfold moreno16_v1_coords, moreno16_v1_sum.
  cbv [oct_add oct_scale oct_e oct_zero quat_add quat_scale quat_zero quat_one
       oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem moreno16_v1_eq_ha_perp : forall y : CDOct,
  moreno15_in_vlambda 1 y <-> moreno15_in_ha_perp y.
Proof.
  intro y.
  split.
  - intros [Hy _].
    exact Hy.
  - intro Hy.
    split.
    + exact Hy.
    + replace (- (1 * 1)) with (-1) by ring.
      rewrite moreno15_e1_left_square_neg.
      replace (oct_scale (-1) y) with (oct_neg (oct_scale 1 y)).
      2:{ symmetry. apply oct_scale_neg_r. }
      rewrite moreno16_oct_scale_one.
      reflexivity.
Qed.

Theorem moreno16_nonunit_vlambda_trivial : forall lambda : R, forall y : CDOct,
  0 < lambda ->
  lambda <> 1 ->
  moreno15_in_vlambda lambda y ->
  y = oct_zero.
Proof.
  exact moreno15_no_nontrivial_vlambda.
Qed.

Lemma moreno16_v1_basis_in_ha_perp : forall i : nat,
  (i < 4)%nat ->
  moreno15_in_ha_perp (moreno16_v1_basis i).
Proof.
  intros i Hi.
  destruct i as [|[|[|[|]]]]; try lia;
  unfold moreno16_v1_basis, moreno15_in_ha_perp;
  repeat eexists; reflexivity.
Qed.

Lemma moreno16_v1_basis_in_v1 : forall i : nat,
  (i < 4)%nat ->
  moreno15_in_vlambda 1 (moreno16_v1_basis i).
Proof.
  intros i Hi.
  apply moreno16_v1_eq_ha_perp.
  apply moreno16_v1_basis_in_ha_perp.
  exact Hi.
Qed.

Theorem moreno16_v1_coordinate_expansion : forall y : CDOct,
  moreno15_in_vlambda 1 y ->
  exists r2 r3 r6 r7 : R, y = moreno16_v1_coords r2 r3 r6 r7.
Proof.
  intros y Hy.
  apply moreno16_v1_eq_ha_perp in Hy.
  exact Hy.
Qed.

Theorem moreno16_v1_basis_expansion : forall y : CDOct,
  moreno15_in_vlambda 1 y ->
  exists r2 r3 r6 r7 : R, y = moreno16_v1_sum r2 r3 r6 r7.
Proof.
  intros y Hy.
  destruct (moreno16_v1_coordinate_expansion y Hy) as [r2 [r3 [r6 [r7 Hyc]]]].
  exists r2, r3, r6, r7.
  rewrite <- moreno16_v1_coords_eq_sum.
  exact Hyc.
Qed.

Theorem moreno16_v1_coords_unique : forall r2 r3 r6 r7 s2 s3 s6 s7 : R,
  moreno16_v1_coords r2 r3 r6 r7 = moreno16_v1_coords s2 s3 s6 s7 ->
  r2 = s2 /\ r3 = s3 /\ r6 = s6 /\ r7 = s7.
Proof.
  intros r2 r3 r6 r7 s2 s3 s6 s7 H.
  unfold moreno16_v1_coords in H.
  assert (Hlo := f_equal oct_lo H).
  assert (Hhi := f_equal oct_hi H).
  simpl in Hlo, Hhi.
  assert (Hr2 := f_equal qc Hlo).
  assert (Hr3 := f_equal qd Hlo).
  assert (Hr6 := f_equal qc Hhi).
  assert (Hr7 := f_equal qd Hhi).
  simpl in Hr2, Hr3, Hr6, Hr7.
  repeat split; lra.
Qed.

Corollary moreno16_v1_dim_mod4 :
  exists k : nat, moreno15_ker_t_tilde_dim = (4 * k)%nat.
Proof.
  unfold moreno15_ker_t_tilde_dim.
  exists 1%nat.
  reflexivity.
Qed.

Corollary moreno16_nonunit_vlambda_dim_mod4 :
  exists k : nat, moreno15_vlambda_total_dim = (4 * k)%nat.
Proof.
  unfold moreno15_vlambda_total_dim.
  exists 0%nat.
  reflexivity.
Qed.

Theorem moreno16_canonical_mod4_profile :
  (exists k : nat, moreno15_ker_t_tilde_dim = (4 * k)%nat) /\
  (exists k : nat, moreno15_vlambda_total_dim = (4 * k)%nat).
Proof.
  split.
  - exact moreno16_v1_dim_mod4.
  - exact moreno16_nonunit_vlambda_dim_mod4.
Qed.
