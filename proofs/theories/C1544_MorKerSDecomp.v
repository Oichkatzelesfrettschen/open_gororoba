(** * C1544_MorKerSDecomp: Moreno (1997) Theorem 2.3 + Corollary 2.4 -- canonical special-couple lane.

    Moreno's Theorem 2.3 studies, for a special couple {a, b}, the operator

      S(y) = (a*y)*b - a*(y*b)

    restricted to V(a;b)^perp, and identifies its kernel with

      Ker(L_(a+b)) + Ker(L_(a-b)).

    Corollary 2.4 then gives the mod-8 constraint on the restricted kernel.

    SCOPE -- degenerate (alternative) regime.  The witnesses a, b are
    octonion (A_3) basis elements, so S restricted to V(a;b)^perp has trivial
    kernel and Ker(L_(a+b)) = Ker(L_(a-b)) = {0}; the decomposition is an
    equality of zero subspaces and the Corollary-2.4 dimension is 0 (trivially
    0 mod 8).  Moreno's Theorem 2.3 / Corollary 2.4 carry content only for the
    non-alternative zero-divisor regime at n >= 4 (where Ker(L_(a+-b)) is
    nonzero); cf. MorenoKerLaNonAlternative.v for nonzero kernels at n = 4.
    This file verifies the alternative instance only.

    This file formalizes the canonical octonion witness used throughout the
    Rust Moreno lane:

      a := e1,  b := e2

    In this case:
    - V(a;b) = span{e0, e1, e2, e3}
    - V(a;b)^perp = span{e4, e5, e6, e7}
    - S acts on V(a;b)^perp by an explicit quarter-turn matrix
    - Ker(S|_(V(a;b)^perp)) is trivial
    - Ker(L_(a+b)) and Ker(L_(a-b)) are both trivial

    Therefore the canonical 8D companion for Theorem 2.3 is the equality
    of three zero subspaces, and the canonical Corollary 2.4 dimension is 0,
    hence 0 mod 8. *)

From Stdlib Require Import Reals Lia Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDNegLemmas.

Open Scope R_scope.

Definition moreno23_a : CDOct := oct_e 1.
Definition moreno23_b : CDOct := oct_e 2.

Definition moreno23_cplus : CDOct := oct_add moreno23_a moreno23_b.
Definition moreno23_cminus : CDOct := oct_sub moreno23_a moreno23_b.

Definition moreno23_perp (p q r s : R) : CDOct :=
  mkOct quat_zero (mkQuat p q r s).

Definition moreno23_in_v_perp (y : CDOct) : Prop :=
  exists p q r s, y = moreno23_perp p q r s.

Definition moreno23_S (y : CDOct) : CDOct :=
  oct_sub (oct_mul (oct_mul moreno23_a y) moreno23_b)
          (oct_mul moreno23_a (oct_mul y moreno23_b)).

Definition moreno23_in_sum_kernels (y : CDOct) : Prop :=
  exists u v : CDOct,
    y = oct_add u v /\
    oct_mul moreno23_cplus u = oct_zero /\
    oct_mul moreno23_cminus v = oct_zero.

Definition moreno24_canonical_lsum_dim : nat := 0%nat.
Definition moreno24_canonical_restricted_dim : nat := 0%nat.

Theorem moreno23_s_on_perp : forall p q r s : R,
  moreno23_S (moreno23_perp p q r s) =
  oct_scale 2 (moreno23_perp s (- r) q (- p)).
Proof.
  intros p q r s.
  unfold moreno23_S, moreno23_perp, moreno23_a, moreno23_b.
  cbv [oct_sub oct_add oct_neg oct_mul oct_scale oct_conj oct_e oct_zero
       quat_scale quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem moreno23_s_restricted_kernel_trivial : forall y : CDOct,
  moreno23_in_v_perp y ->
  moreno23_S y = oct_zero ->
  y = oct_zero.
Proof.
  intros y [p [q [r [s Hy]]]] HS.
  subst y.
  rewrite moreno23_s_on_perp in HS.
  cbv [oct_scale moreno23_perp oct_zero quat_scale quat_zero] in HS.
  inversion HS; clear HS; subst.
  assert (Hs0 : s = 0) by lra.
  assert (Hr0 : r = 0) by lra.
  assert (Hq0 : q = 0) by lra.
  assert (Hp0 : p = 0) by lra.
  subst.
  reflexivity.
Qed.

Lemma moreno23_oct_norm_zero : forall y : CDOct,
  oct_norm_sq y = 0 ->
  y = oct_zero.
Proof.
  intros [[a b c d] [e f g h]] H.
  cbv [oct_norm_sq quat_norm_sq oct_zero quat_zero oct_lo oct_hi qa qb qc qd] in H.
  assert (Ha : a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 >= 0) by nra.
  assert (Hb : e ^ 2 + f ^ 2 + g ^ 2 + h ^ 2 >= 0) by nra.
  assert (Ha0 : a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 = 0) by lra.
  assert (Hb0 : e ^ 2 + f ^ 2 + g ^ 2 + h ^ 2 = 0) by lra.
  assert (a = 0) by nra.
  assert (b = 0) by nra.
  assert (c = 0) by nra.
  assert (d = 0) by nra.
  assert (e = 0) by nra.
  assert (f = 0) by nra.
  assert (g = 0) by nra.
  assert (h = 0) by nra.
  subst.
  reflexivity.
Qed.

Theorem moreno23_cplus_square :
  oct_mul moreno23_cplus moreno23_cplus = oct_scale (-2) (oct_e 0).
Proof.
  unfold moreno23_cplus, moreno23_a, moreno23_b.
  cbv [oct_add oct_mul oct_scale oct_e oct_conj quat_mul quat_add
       quat_neg quat_conj quat_scale quat_zero quat_one oct_lo oct_hi
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem moreno23_cminus_square :
  oct_mul moreno23_cminus moreno23_cminus = oct_scale (-2) (oct_e 0).
Proof.
  unfold moreno23_cminus, moreno23_a, moreno23_b, oct_sub.
  cbv [oct_add oct_neg oct_mul oct_scale oct_e oct_conj quat_mul quat_add
       quat_neg quat_conj quat_scale quat_zero quat_one oct_lo oct_hi
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno23_cplus_norm :
  oct_norm_sq moreno23_cplus = 2.
Proof.
  unfold moreno23_cplus, moreno23_a, moreno23_b.
  cbv [oct_norm_sq oct_add oct_e quat_norm_sq quat_add
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Lemma moreno23_cminus_norm :
  oct_norm_sq moreno23_cminus = 2.
Proof.
  unfold moreno23_cminus, moreno23_a, moreno23_b, oct_sub.
  cbv [oct_norm_sq oct_add oct_neg oct_e quat_norm_sq quat_add quat_neg
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Theorem moreno23_cplus_kernel_trivial : forall y : CDOct,
  oct_mul moreno23_cplus y = oct_zero ->
  y = oct_zero.
Proof.
  intros y Hy.
  assert (Hnorm : oct_norm_sq (oct_mul moreno23_cplus y) = 0).
  { rewrite Hy.
    cbv [oct_norm_sq oct_zero quat_norm_sq quat_zero oct_lo oct_hi qa qb qc qd].
    ring. }
  rewrite oct_norm_mul in Hnorm.
  rewrite moreno23_cplus_norm in Hnorm.
  apply moreno23_oct_norm_zero.
  lra.
Qed.

Theorem moreno23_cminus_kernel_trivial : forall y : CDOct,
  oct_mul moreno23_cminus y = oct_zero ->
  y = oct_zero.
Proof.
  intros y Hy.
  assert (Hnorm : oct_norm_sq (oct_mul moreno23_cminus y) = 0).
  { rewrite Hy.
    cbv [oct_norm_sq oct_zero quat_norm_sq quat_zero oct_lo oct_hi qa qb qc qd].
    ring. }
  rewrite oct_norm_mul in Hnorm.
  rewrite moreno23_cminus_norm in Hnorm.
  apply moreno23_oct_norm_zero.
  lra.
Qed.

Lemma moreno23_s_zero : moreno23_S oct_zero = oct_zero.
Proof.
  unfold moreno23_S, moreno23_a, moreno23_b.
  cbv [oct_sub oct_add oct_neg oct_mul oct_conj oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma moreno23_in_sum_kernels_zero : moreno23_in_sum_kernels oct_zero.
Proof.
  exists oct_zero, oct_zero.
  split.
  - rewrite oct_add_zero_left.
    reflexivity.
  - split.
    + apply oct_mul_zero_right.
    + apply oct_mul_zero_right.
Qed.

Theorem moreno23_kernel_decomposition_canonical : forall y : CDOct,
  moreno23_in_v_perp y ->
  (moreno23_S y = oct_zero <-> moreno23_in_sum_kernels y).
Proof.
  intros y Hy.
  split.
  - intro HS.
    assert (Hz : y = oct_zero).
    { apply moreno23_s_restricted_kernel_trivial; assumption. }
    subst y.
    exact moreno23_in_sum_kernels_zero.
  - intros [u [v [Hyv [Hu Hv]]]].
    assert (Hu0 : u = oct_zero).
    { apply moreno23_cplus_kernel_trivial; exact Hu. }
    assert (Hv0 : v = oct_zero).
    { apply moreno23_cminus_kernel_trivial; exact Hv. }
    subst u v.
    rewrite oct_add_zero_left in Hyv.
    subst y.
    exact moreno23_s_zero.
Qed.

Corollary moreno24_canonical_mod8 :
  moreno24_canonical_restricted_dim = (2 * moreno24_canonical_lsum_dim)%nat /\
  (moreno24_canonical_restricted_dim mod 8 = 0)%nat.
Proof.
  unfold moreno24_canonical_restricted_dim, moreno24_canonical_lsum_dim.
  split; reflexivity.
Qed.
