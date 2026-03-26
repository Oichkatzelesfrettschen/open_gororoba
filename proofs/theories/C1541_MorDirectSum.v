(** * C1541_MorDirectSum: Moreno (1997) Theorem 1.15 -- canonical octonion decomposition.

    Moreno's Theorem 1.15 states that for a doubly-pure unit-norm element a
    in A_n there is a direct-sum decomposition

      A_n = H_a (+) Ker(T~_a) (+) Ker(L_a) (+) direct-sum(V_lambda)

    where
      T_a = -I - L_a^2,
      T~_a is the restriction of T_a to H_a^perp,
      Ker(T~_a) = V_1,
      V_lambda = {x in H_a^perp | a*(a*x) = -(lambda^2) x}.

    This file formalizes the canonical octonion witness used by the Rust
    Moreno lane:

      a := e1 in A_3.

    In this case:
    - H_a = span{e0, e1, e5, e4}
    - H_a^perp = span{e2, e3, e6, e7}
    - octonion alternativity gives a*(a*x) = -x for all x in A_3
    - therefore T_a = 0 on all of A_3, hence Ker(T~_a) = H_a^perp
    - Ker(L_a) = {0}
    - there are no nontrivial V_lambda blocks for lambda > 0, lambda <> 1

    So the canonical 8D Moreno decomposition collapses to

      A_3 = H_a (+) H_a^perp

    with dimension profile 4 + 4 + 0 + 0 = 8.

    Source: Moreno 1997, arXiv:q-alg/9710013v1, pp. 15-17. *)

From Stdlib Require Import Reals Lia Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDInnerProduct CDNegLemmas CDSignBridge C1540_MorHaQuaternion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Open Scope R_scope.

Definition moreno15_t (y : CDOct) : CDOct :=
  oct_add (oct_neg y) (oct_neg (oct_mul moreno_a (oct_mul moreno_a y))).

Definition moreno15_ha_part (y : CDOct) : CDOct :=
  mkOct
    (mkQuat (qa (oct_lo y)) (qb (oct_lo y)) 0 0)
    (mkQuat (qa (oct_hi y)) (qb (oct_hi y)) 0 0).

Definition moreno15_ha_perp_part (y : CDOct) : CDOct :=
  mkOct
    (mkQuat 0 0 (qc (oct_lo y)) (qd (oct_lo y)))
    (mkQuat 0 0 (qc (oct_hi y)) (qd (oct_hi y))).

Definition moreno15_in_ha (y : CDOct) : Prop :=
  exists r0 r1 r2 r3,
    y = mkOct (mkQuat r0 r1 0 0) (mkQuat r3 r2 0 0).

Definition moreno15_in_ha_perp (y : CDOct) : Prop :=
  exists r2 r3 r6 r7,
    y = mkOct (mkQuat 0 0 r2 r3) (mkQuat 0 0 r6 r7).

Definition moreno15_in_ker_t_tilde (y : CDOct) : Prop :=
  moreno15_in_ha_perp y /\ moreno15_t y = oct_zero.

Definition moreno15_in_ker_l (y : CDOct) : Prop :=
  oct_mul moreno_a y = oct_zero.

Definition moreno15_in_vlambda (lambda : R) (y : CDOct) : Prop :=
  moreno15_in_ha_perp y /\
  oct_mul moreno_a (oct_mul moreno_a y) = oct_scale (- (lambda * lambda)) y.

Definition moreno15_h_a_dim : nat := 4%nat.
Definition moreno15_ker_t_tilde_dim : nat := 4%nat.
Definition moreno15_ker_l_dim : nat := 0%nat.
Definition moreno15_vlambda_total_dim : nat := 0%nat.

Lemma moreno15_ha_part_in_ha : forall y : CDOct,
  moreno15_in_ha (moreno15_ha_part y).
Proof.
  intros [[a b c d] [e f g h]].
  unfold moreno15_ha_part, moreno15_in_ha.
  exists a.
  exists b.
  exists f.
  exists e.
  reflexivity.
Qed.

Lemma moreno15_ha_perp_part_in_ha_perp : forall y : CDOct,
  moreno15_in_ha_perp (moreno15_ha_perp_part y).
Proof.
  intros [[a b c d] [e f g h]].
  unfold moreno15_ha_perp_part, moreno15_in_ha_perp.
  exists c.
  exists d.
  exists g.
  exists h.
  reflexivity.
Qed.

Theorem moreno15_split : forall y : CDOct,
  y = oct_add (moreno15_ha_part y) (moreno15_ha_perp_part y).
Proof.
  intros [[a b c d] [e f g h]].
  unfold moreno15_ha_part, moreno15_ha_perp_part, oct_add, quat_add.
  simpl.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem moreno15_ha_perp_orthogonal : forall h p : CDOct,
  moreno15_in_ha h ->
  moreno15_in_ha_perp p ->
  oct_inner h p = 0.
Proof.
  intros h p [r0 [r1 [r2 [r3 Hh]]]] [s2 [s3 [s6 [s7 Hp]]]].
  subst h p.
  unfold oct_inner, quat_inner.
  simpl.
  ring.
Qed.

Theorem moreno15_ha_intersection_zero : forall y : CDOct,
  moreno15_in_ha y ->
  moreno15_in_ha_perp y ->
  y = oct_zero.
Proof.
  intros y [r0 [r1 [r2 [r3 Hy]]]] [s2 [s3 [s6 [s7 Hp]]]].
  subst y.
  assert (Hlo := f_equal oct_lo Hp).
  assert (Hhi := f_equal oct_hi Hp).
  simpl in Hlo, Hhi.
  injection Hlo as Hr0 Hr1 _ _.
  injection Hhi as Hr3 Hr2 _ _.
  subst.
  reflexivity.
Qed.

Lemma moreno15_oct_mul_e0_left : forall x : CDOct,
  oct_mul (oct_e 0) x = x.
Proof.
  intros [[a b c d] [e f g h]].
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno15_oct_mul_neg_e0_left : forall x : CDOct,
  oct_mul (oct_neg (oct_e 0)) x = oct_neg x.
Proof.
  intro x.
  rewrite oct_neg_mul_left.
  rewrite moreno15_oct_mul_e0_left.
  reflexivity.
Qed.

Lemma moreno15_e1_left_square_neg : forall x : CDOct,
  oct_mul moreno_a (oct_mul moreno_a x) = oct_neg x.
Proof.
  intro x.
  unfold moreno_a.
  rewrite C910_octonion_left_alt_e1.
  rewrite oct_mul_self_neg by lia.
  apply moreno15_oct_mul_neg_e0_left.
Qed.

Theorem moreno15_t_zero : forall y : CDOct,
  moreno15_t y = oct_zero.
Proof.
  intro y.
  unfold moreno15_t.
  rewrite moreno15_e1_left_square_neg.
  rewrite oct_neg_neg.
  rewrite oct_add_comm.
  apply oct_add_neg_cancel.
Qed.

Theorem moreno15_t_tilde_zero : forall y : CDOct,
  moreno15_in_ha_perp y ->
  moreno15_t y = oct_zero.
Proof.
  intros y _.
  apply moreno15_t_zero.
Qed.

Theorem moreno15_ker_t_tilde_is_ha_perp : forall y : CDOct,
  moreno15_in_ker_t_tilde y <-> moreno15_in_ha_perp y.
Proof.
  intro y.
  split.
  - intros [Hy _].
    exact Hy.
  - intro Hy.
    split.
    + exact Hy.
    + apply moreno15_t_tilde_zero.
      exact Hy.
Qed.

Theorem moreno15_ker_l_trivial : forall y : CDOct,
  moreno15_in_ker_l y ->
  y = oct_zero.
Proof.
  intros y Hy.
  unfold moreno15_in_ker_l, moreno_a in Hy.
  apply (f_equal (fun z => oct_mul (oct_e 1) z)) in Hy.
  rewrite oct_mul_zero_right in Hy.
  rewrite moreno15_e1_left_square_neg in Hy.
  apply (f_equal oct_neg) in Hy.
  rewrite oct_neg_neg in Hy.
  rewrite oct_neg_zero in Hy.
  exact Hy.
Qed.

Theorem moreno15_direct_sum_exists : forall y : CDOct,
  exists h v1 : CDOct,
    moreno15_in_ha h /\
    moreno15_in_ker_t_tilde v1 /\
    y = oct_add h v1.
Proof.
  intro y.
  exists (moreno15_ha_part y), (moreno15_ha_perp_part y).
  split.
  - apply moreno15_ha_part_in_ha.
  - split.
    + apply moreno15_ker_t_tilde_is_ha_perp.
      apply moreno15_ha_perp_part_in_ha_perp.
    + apply moreno15_split.
Qed.

Theorem moreno15_sum_components_unique : forall h1 h2 p1 p2 : CDOct,
  moreno15_in_ha h1 ->
  moreno15_in_ha h2 ->
  moreno15_in_ha_perp p1 ->
  moreno15_in_ha_perp p2 ->
  oct_add h1 p1 = oct_add h2 p2 ->
  h1 = h2 /\ p1 = p2.
Proof.
  intros h1 h2 p1 p2
         [a0 [a1 [a2 [a3 Hh1]]]]
         [b0 [b1 [b2 [b3 Hh2]]]]
         [c2 [c3 [c6 [c7 Hp1]]]]
         [d2 [d3 [d6 [d7 Hp2]]]]
         Hsum.
  subst h1 h2 p1 p2.
  unfold oct_add, quat_add in Hsum.
  simpl in Hsum.
  inversion Hsum; subst.
  split; apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

Theorem moreno15_direct_sum_unique : forall y h1 h2 v1 v2 : CDOct,
  moreno15_in_ha h1 ->
  moreno15_in_ha h2 ->
  moreno15_in_ker_t_tilde v1 ->
  moreno15_in_ker_t_tilde v2 ->
  y = oct_add h1 v1 ->
  y = oct_add h2 v2 ->
  h1 = h2 /\ v1 = v2.
Proof.
  intros y h1 h2 v1 v2 Hh1 Hh2 [Hv1 _] [Hv2 _] Hy1 Hy2.
  rewrite Hy1 in Hy2.
  eapply moreno15_sum_components_unique; eauto.
Qed.

Theorem moreno15_no_nontrivial_vlambda : forall lambda : R, forall y : CDOct,
  0 < lambda ->
  lambda <> 1 ->
  moreno15_in_vlambda lambda y ->
  y = oct_zero.
Proof.
  intros lambda y Hlam Hneq [_ Hy].
  destruct y as [[a b c d] [e f g h]].
  rewrite moreno15_e1_left_square_neg in Hy.
  unfold oct_neg, oct_scale in Hy.
  simpl in Hy.
  assert (Hnz : 1 - lambda * lambda <> 0) by nra.
  assert (Hlo := f_equal oct_lo Hy).
  assert (Hhi := f_equal oct_hi Hy).
  simpl in Hlo, Hhi.
  assert (Haq := f_equal qa Hlo).
  assert (Hbq := f_equal qb Hlo).
  assert (Hcq := f_equal qc Hlo).
  assert (Hdq := f_equal qd Hlo).
  assert (Heq := f_equal qa Hhi).
  assert (Hfq := f_equal qb Hhi).
  assert (Hgq := f_equal qc Hhi).
  assert (Hhq := f_equal qd Hhi).
  simpl in Haq, Hbq, Hcq, Hdq, Heq, Hfq, Hgq, Hhq.
  assert (Ha : a = 0) by nra.
  assert (Hb : b = 0) by nra.
  assert (Hc : c = 0) by nra.
  assert (Hd : d = 0) by nra.
  assert (He : e = 0) by nra.
  assert (Hf : f = 0) by nra.
  assert (Hg : g = 0) by nra.
  assert (Hh : h = 0) by nra.
  subst.
  reflexivity.
Qed.

Corollary moreno16_canonical_vlambda_trivial : forall lambda : R,
  0 < lambda ->
  lambda <> 1 ->
  forall y : CDOct, moreno15_in_vlambda lambda y -> y = oct_zero.
Proof.
  intros lambda Hlam Hneq y Hy.
  apply (moreno15_no_nontrivial_vlambda lambda y Hlam Hneq Hy).
Qed.

Corollary moreno15_canonical_dimension_profile :
  (moreno15_h_a_dim + moreno15_ker_t_tilde_dim +
   moreno15_ker_l_dim + moreno15_vlambda_total_dim = 8)%nat.
Proof.
  unfold moreno15_h_a_dim, moreno15_ker_t_tilde_dim,
         moreno15_ker_l_dim, moreno15_vlambda_total_dim.
  reflexivity.
Qed.

Corollary moreno15_canonical_ker_l_mod4 :
  (moreno15_ker_l_dim mod 4 = 0)%nat.
Proof.
  unfold moreno15_ker_l_dim.
  reflexivity.
Qed.
