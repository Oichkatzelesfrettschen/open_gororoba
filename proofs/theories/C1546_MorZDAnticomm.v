(** * C1546_MorZDAnticomm: Anti-commutation from orthogonality (oct_free path).

    KEY RESULT: For pure octonions x, y with <x,y> = 0, we have yx = -xy.

    SOURCE: Moreno (1997), arXiv:q-alg/9710013v1, page 5:
      "elements of zero trace are orthogonal if and only if they anti-commute"

    This file provides the oct_free bridge that discharges all four
    anti-commutation hypotheses of moreno29_arbitrary_iff
    (C1546_MorArbitraryAlternative.v) from inner-product-zero conditions,
    closing the L-1546 lacuna without Gram matrices or linear algebra.

    PROOF METHOD: Direct computation at dim 8 (14 free real variables after
    purity constraints). Each component is a degree-2 polynomial identity
    dispatched by cbv + ring/lra.

    Supports claim C-1546. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDLinearLemmas CDNegLemmas CDConjAntimorph CDInnerProduct
  C1542_MorHaStabilityArb C1546_MorEigenIFF C1546_MorArbitraryAlternative.

Open Scope R_scope.

(** ================================================================== *)
(** * Helper: extract zero real part from purity.                      *)
(** ================================================================== *)

Lemma oct_pure_real_zero : forall x : CDOct,
  oct_conj x = oct_neg x ->
  qa (oct_lo x) = 0.
Proof.
  intros x Hpure.
  destruct x as [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  simpl.
  assert (Hlo := f_equal oct_lo Hpure).
  cbv [oct_conj oct_neg quat_conj quat_neg oct_lo] in Hlo.
  assert (Ha := f_equal qa Hlo). simpl in Ha. lra.
Qed.

(** ================================================================== *)
(** * Inner product linearity helpers.                                 *)
(** ================================================================== *)

Lemma oct_inner_neg_right : forall x y : CDOct,
  oct_inner x (oct_neg y) = - oct_inner x y.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [oct_inner oct_neg quat_inner quat_neg oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Lemma oct_inner_neg_left : forall x y : CDOct,
  oct_inner (oct_neg x) y = - oct_inner x y.
Proof.
  intros x y.
  rewrite oct_inner_symm.
  rewrite oct_inner_neg_right.
  rewrite oct_inner_symm.
  ring.
Qed.

(** ================================================================== *)
(** * Core: orthogonal pure octonions anti-commute.                    *)
(** ================================================================== *)

(** Moreno 1997 page 5: for pure x, y in A_n,
      x perp y  <==>  xy = -yx.

    Proof strategy at dim 8:
    - Extract qa(oct_lo x) = 0 and qa(oct_lo y) = 0 from purity.
    - The 7 imaginary components of (xy + yx) vanish identically for
      pure elements (ring identity in 14 variables).
    - The 1 real component of (xy + yx) equals -2 * oct_inner x y,
      which vanishes by the orthogonality hypothesis.
    - Together: yx = -xy. *)

Theorem oct_anticomm_of_pure_orthogonal : forall x y : CDOct,
  oct_conj x = oct_neg x ->
  oct_conj y = oct_neg y ->
  oct_inner x y = 0 ->
  oct_mul y x = oct_neg (oct_mul x y).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]]
         Hx_pure Hy_pure Horth.
  assert (Ha1 : a1 = 0).
  { exact (oct_pure_real_zero _ Hx_pure). }
  assert (Hb1 : b1 = 0).
  { exact (oct_pure_real_zero _ Hy_pure). }
  subst a1 b1.
  cbv [oct_inner quat_inner oct_lo oct_hi qa qb qc qd] in Horth.
  cbv [oct_mul oct_neg oct_conj quat_mul quat_add quat_neg quat_conj
       oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

(** ================================================================== *)
(** * Product of orthogonal pure elements is pure.                     *)
(** ================================================================== *)

(** If x, y are pure and orthogonal then xy is also pure.
    Proof: trace(xy) = xy + conj(xy) = xy + conj(y)*conj(x) = xy + yx
    (since conj = neg for pure elements). By anti-commutation, = xy - xy = 0. *)

Lemma oct_mul_pure_orthogonal_is_pure : forall x y : CDOct,
  oct_conj x = oct_neg x ->
  oct_conj y = oct_neg y ->
  oct_inner x y = 0 ->
  oct_conj (oct_mul x y) = oct_neg (oct_mul x y).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]]
         Hx Hy Horth.
  assert (Ha1 : a1 = 0) by exact (oct_pure_real_zero _ Hx).
  assert (Hb1 : b1 = 0) by exact (oct_pure_real_zero _ Hy).
  subst a1 b1.
  cbv [oct_inner quat_inner oct_lo oct_hi qa qb qc qd] in Horth.
  cbv [oct_conj oct_mul oct_neg quat_conj quat_mul quat_add quat_neg
       oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

(** ================================================================== *)
(** * Inner product rotation under left multiplication by pure element *)
(** ================================================================== *)

(** <a, b*y> = -<y, b*a> for pure b.

    Derivation:
      <a, by> = <by, a>               (symmetry)
              = <y, conj(b)*a>         (left adjoint: <xz, w> = <z, conj(x)*w>)
              = <y, (-b)*a>            (b pure: conj(b) = -b)
              = -<y, b*a>              (inner product linear in neg) *)

Lemma oct_inner_rotation_pure : forall a b y : CDOct,
  oct_conj b = oct_neg b ->
  oct_inner a (oct_mul b y) = - oct_inner y (oct_mul b a).
Proof.
  intros a b y Hb_pure.
  rewrite oct_inner_symm.
  rewrite oct_inner_adj_left.
  rewrite Hb_pure.
  rewrite oct_neg_mul_left.
  rewrite oct_inner_neg_right.
  ring.
Qed.

(** ================================================================== *)
(** * Configuration: three orthogonality conditions discharge all four *)
(*    anti-commutation hypotheses of moreno29_arbitrary_iff.           *)
(** ================================================================== *)

(** The four hypotheses required by moreno29_arbitrary_iff are:
    H1: oct_mul y a = oct_neg (oct_mul a y)      -- y anticommutes with a
    H2: oct_mul y b = oct_neg (oct_mul b y)      -- y anticommutes with b
    H3: oct_conj (oct_mul a (oct_mul b y)) = ...  -- a*(b*y) is pure
    H4: oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y))
                                                  -- (b*y) anticommutes with a

    These follow from three inner-product-zero conditions:
    C1: <a, y> = 0
    C2: <b, y> = 0
    C3: <b*a, y> = 0  (equivalently <a, b*y> = 0 by rotation lemma) *)

Theorem moreno29_anticomm_from_orthogonality :
  forall a b y : CDOct,
    oct_norm_sq a = 1 ->
    oct_norm_sq b = 1 ->
    oct_conj a = oct_neg a ->
    oct_conj b = oct_neg b ->
    oct_conj y = oct_neg y ->
    oct_inner a y = 0 ->
    oct_inner b y = 0 ->
    oct_inner (oct_mul b a) y = 0 ->
    (oct_mul y a = oct_neg (oct_mul a y) /\
     oct_mul y b = oct_neg (oct_mul b y) /\
     oct_conj (oct_mul a (oct_mul b y)) = oct_neg (oct_mul a (oct_mul b y)) /\
     oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y))).
Proof.
  intros a b y Ha_unit Hb_unit Ha_pure Hb_pure Hy_pure
         Hay Hby Hbay.
  (* H1: y anticommutes with a -- from <a,y> = 0 and both pure. *)
  assert (H1 : oct_mul y a = oct_neg (oct_mul a y)).
  { apply oct_anticomm_of_pure_orthogonal; assumption. }
  (* H2: y anticommutes with b -- from <b,y> = 0 and both pure. *)
  assert (H2 : oct_mul y b = oct_neg (oct_mul b y)).
  { apply oct_anticomm_of_pure_orthogonal; assumption. }
  (* Intermediate: b*y is pure (b and y are orthogonal pure). *)
  assert (Hby_pure : oct_conj (oct_mul b y) = oct_neg (oct_mul b y)).
  { apply oct_mul_pure_orthogonal_is_pure; assumption. }
  (* Intermediate: <a, b*y> = 0 from rotation + Hbay. *)
  assert (Haby_orth : oct_inner a (oct_mul b y) = 0).
  { rewrite oct_inner_rotation_pure by exact Hb_pure.
    rewrite (oct_inner_symm y (oct_mul b a)).
    rewrite Hbay. lra. }
  (* H4: (b*y) anticommutes with a -- from <a, b*y> = 0 and both pure. *)
  assert (H4 : oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y))).
  { apply oct_anticomm_of_pure_orthogonal; assumption. }
  (* H3: a*(b*y) is pure -- a and b*y are orthogonal pure. *)
  assert (H3 : oct_conj (oct_mul a (oct_mul b y)) = oct_neg (oct_mul a (oct_mul b y))).
  { apply oct_mul_pure_orthogonal_is_pure; assumption. }
  exact (conj H1 (conj H2 (conj H3 H4))).
Qed.

(** ================================================================== *)
(** * Final theorem: Moreno 2.9 with orthogonality hypotheses.         *)
(** ================================================================== *)

(** This is the definitive form of Theorem 2.9 for formal verification:
    the Moreno zero-divisor witness equation system is equivalent to y
    being a -2-eigenvector of L_{a+b}^2, under purely inner-product-based
    side conditions (no explicit anti-commutation hypotheses). *)

Theorem moreno29_orthogonal_iff :
  forall a b : CDOct,
    oct_norm_sq a = 1 ->
    oct_norm_sq b = 1 ->
    oct_conj a = oct_neg a ->
    oct_conj b = oct_neg b ->
    forall y : CDOct,
      oct_conj y = oct_neg y ->
      oct_inner a y = 0 ->
      oct_inner b y = 0 ->
      oct_inner (oct_mul b a) y = 0 ->
      (moreno29_witness_eqns a b (moreno29_partner a b y) y <->
       y <> oct_zero /\
       oct_mul (moreno29_c a b) (oct_mul (moreno29_c a b) y) = oct_scale (-2) y).
Proof.
  intros a b Ha_unit Hb_unit Ha_pure Hb_pure y Hy_pure Hay Hby Hbay.
  destruct (moreno29_anticomm_from_orthogonality
              a b y Ha_unit Hb_unit Ha_pure Hb_pure Hy_pure Hay Hby Hbay)
    as [H1 [H2 [H3 H4]]].
  exact (moreno29_arbitrary_iff a b Ha_unit Ha_pure Hb_unit Hb_pure
           y Hy_pure H1 H2 H4).
Qed.
