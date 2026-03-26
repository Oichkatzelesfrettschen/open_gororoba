(** * C1546_MorEigenZD: Moreno (1997) Theorem 2.9 -- canonical eigenvalue witness lane.

    Moreno's Theorem 2.9 characterizes zero divisors in A_(n+1) by the
    appearance of the eigenvalue -2 in the spectrum of L_(a+b)^2 on A_n.

    This file formalizes the repo's canonical concrete companions:

    Positive lane in A_3 -> A_4:
    - a := e1, b := e2, c := a + b
    - c^2 = -2 e0, hence c itself is a nonzero witness for
      L_c^2(y) = -2 y
    - the lifted sedenion pair (a,b) = e1 + e10 has the explicit partner
      e4 - e15 and multiplies to zero

    Negative contrast lane in A_3:
    - parallel pair a = b = e1, so c = 2 e1
    - L_c^2 = -4 I, hence -2 is not an eigenvalue

    This gives a paper-scoped concrete Rocq companion for C-1546 while the
    full abstract iff theorem remains a separate structural task. *)

From Stdlib Require Import Reals Lia Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDLinearLemmas CDNegLemmas CDSignBridge.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Open Scope R_scope.

Definition moreno29_a : CDOct := oct_e 1.
Definition moreno29_b : CDOct := oct_e 2.
Definition moreno29_c : CDOct := oct_add moreno29_a moreno29_b.

Definition moreno29_pair : CDSed := mkSed moreno29_a moreno29_b.
Definition moreno29_partner : CDSed := mkSed (oct_e 4) (oct_neg (oct_e 7)).

Definition moreno29_parallel_c : CDOct := oct_scale 2 (oct_e 1).

Lemma moreno29_c_nonzero : moreno29_c <> oct_zero.
Proof.
  unfold moreno29_c, moreno29_a, moreno29_b, oct_add, oct_zero.
  intro H.
  assert (Hlo := f_equal oct_lo H).
  simpl in Hlo.
  assert (Hb := f_equal qb Hlo).
  simpl in Hb.
  lra.
Qed.

Lemma moreno29_pair_nonzero : moreno29_pair <> sed_zero.
Proof.
  unfold moreno29_pair, moreno29_a, moreno29_b, sed_zero.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  simpl in Hlo.
  assert (Hq := f_equal oct_lo Hlo).
  simpl in Hq.
  assert (Hb := f_equal qb Hq).
  simpl in Hb.
  lra.
Qed.

Lemma moreno29_partner_nonzero : moreno29_partner <> sed_zero.
Proof.
  unfold moreno29_partner, sed_zero.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  simpl in Hlo.
  assert (Hhi := f_equal oct_hi Hlo).
  simpl in Hhi.
  assert (Ha := f_equal qa Hhi).
  simpl in Ha.
  lra.
Qed.

Lemma moreno29_c_square :
  oct_mul moreno29_c moreno29_c = oct_scale (-2) (oct_e 0).
Proof.
  unfold moreno29_c, moreno29_a, moreno29_b.
  cbv [oct_add oct_mul oct_scale oct_e oct_conj quat_mul quat_add
       quat_neg quat_conj quat_scale quat_zero quat_one oct_lo oct_hi
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem moreno29_c_has_eigen_minus_two :
  exists y : CDOct,
    y <> oct_zero /\
    oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y.
Proof.
  exists moreno29_c.
  split.
  - exact moreno29_c_nonzero.
  - rewrite moreno29_c_square.
    change
      (oct_mul moreno29_c (oct_scale (-2) (mkOct quat_one quat_zero)) =
       oct_scale (-2) moreno29_c).
    rewrite oct_mul_scale_right.
    rewrite oct_mul_one_right.
    reflexivity.
Qed.

Theorem moreno29_pair_is_zero_divisor :
  moreno29_pair <> sed_zero /\
  moreno29_partner <> sed_zero /\
  sed_mul moreno29_pair moreno29_partner = sed_zero.
Proof.
  split.
  - exact moreno29_pair_nonzero.
  - split.
    + exact moreno29_partner_nonzero.
    + unfold moreno29_pair, moreno29_partner, moreno29_a, moreno29_b, sed_zero.
      cbv [sed_mul oct_mul oct_neg oct_conj oct_e oct_zero
           quat_mul quat_add quat_neg quat_conj quat_zero quat_one
           sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
      apply (f_equal2 mkSed).
      * apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
      * apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno29_oct_scale_scale : forall r s : R, forall x : CDOct,
  oct_scale r (oct_scale s x) = oct_scale (r * s) x.
Proof.
  intros r s [[a b c d] [e f g h]].
  unfold oct_scale, quat_scale.
  simpl.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno29_oct_scale_neg : forall r : R, forall x : CDOct,
  oct_scale r (oct_neg x) = oct_neg (oct_scale r x).
Proof.
  intros r [[a b c d] [e f g h]].
  unfold oct_scale, oct_neg, quat_scale, quat_neg.
  simpl.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno29_oct_mul_e0_left : forall x : CDOct,
  oct_mul (oct_e 0) x = x.
Proof.
  intros [[a b c d] [e f g h]].
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno29_oct_mul_neg_e0_left : forall x : CDOct,
  oct_mul (oct_neg (oct_e 0)) x = oct_neg x.
Proof.
  intro x.
  rewrite oct_neg_mul_left.
  rewrite moreno29_oct_mul_e0_left.
  reflexivity.
Qed.

Lemma moreno29_e1_left_square_neg : forall x : CDOct,
  oct_mul (oct_e 1) (oct_mul (oct_e 1) x) = oct_neg x.
Proof.
  intro x.
  rewrite C910_octonion_left_alt_e1.
  rewrite oct_mul_self_neg by lia.
  apply moreno29_oct_mul_neg_e0_left.
Qed.

Theorem moreno29_parallel_left_square : forall x : CDOct,
  oct_mul moreno29_parallel_c (oct_mul moreno29_parallel_c x) =
  oct_scale (-4) x.
Proof.
  intro x.
  unfold moreno29_parallel_c.
  rewrite oct_mul_scale_left.
  rewrite oct_mul_scale_left.
  rewrite oct_mul_scale_right.
  rewrite moreno29_e1_left_square_neg.
  rewrite moreno29_oct_scale_scale.
  rewrite moreno29_oct_scale_neg.
  rewrite <- oct_scale_neg_r.
  reflexivity.
Qed.

Theorem moreno29_parallel_not_eigen_minus_two :
  ~ exists y : CDOct,
      y <> oct_zero /\
      oct_mul moreno29_parallel_c (oct_mul moreno29_parallel_c y) =
      oct_scale (-2) y.
Proof.
  intros [y [Hy_nonzero Hy]].
  rewrite moreno29_parallel_left_square in Hy.
  destruct y as [[a b c d] [e f g h]].
  unfold oct_scale in Hy.
  simpl in Hy.
  assert (Hlo := f_equal oct_lo Hy).
  assert (Hhi := f_equal oct_hi Hy).
  simpl in Hlo, Hhi.
  assert (Ha : a = 0).
  { apply (f_equal qa) in Hlo. simpl in Hlo. lra. }
  assert (Hb : b = 0).
  { apply (f_equal qb) in Hlo. simpl in Hlo. lra. }
  assert (Hc : c = 0).
  { apply (f_equal qc) in Hlo. simpl in Hlo. lra. }
  assert (Hd : d = 0).
  { apply (f_equal qd) in Hlo. simpl in Hlo. lra. }
  assert (He : e = 0).
  { apply (f_equal qa) in Hhi. simpl in Hhi. lra. }
  assert (Hf : f = 0).
  { apply (f_equal qb) in Hhi. simpl in Hhi. lra. }
  assert (Hg : g = 0).
  { apply (f_equal qc) in Hhi. simpl in Hhi. lra. }
  assert (Hh : h = 0).
  { apply (f_equal qd) in Hhi. simpl in Hhi. lra. }
  apply Hy_nonzero.
  subst.
  reflexivity.
Qed.
