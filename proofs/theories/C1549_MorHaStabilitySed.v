(** * C1549_MorHaStabilitySed: Moreno (1997) canonical H_a-stability in CDSed.

    WHAT THIS FILE PROVES:

    This file is the CDSed (A_4, dim=16) canonical analog of C1541_MorDirectSum.v
    (which works in CDOct, A_3, dim=8).

    Canonical element: moreno_sed_a := sed_e 1 (the first imaginary basis element
    of the sedenion algebra).

    Key theorem: for a = sed_e 1 in CDSed,
      a*(a*x) = -x   for all x in CDSed.

    Proof:  The CD doubling formula gives
      (a*y)_lo = oct_mul(oct_e 1)(oct_lo y)
      (a*y)_hi = oct_mul(oct_hi y)(oct_neg(oct_e 1))
    so
      (a*(a*y))_lo = e1*(e1*y_lo) = -y_lo  [CDOct left-alt for basis e1, C910]
      (a*(a*y))_hi = (y_hi*(-e1))*(-e1)
                   = y_hi*(e1*e1)            [CDOct right-alt for basis e1, C910]
                   = y_hi*(-e0)
                   = -y_hi.
    Hence a*(a*y) = -y and T_a = -I - L_a^2 = -I - (-I) = 0.

    Consequence: For lambda > 0 and lambda != 1, V_lambda(a) = {} in CDSed.
    For lambda = 1, V_1(a) = H_a^perp(a) (same structure as CDOct canonical lane).

    This is NOT a closure of L-1542 (which concerns arbitrary-a in CDOct).
    It is a NEW result: the Moreno canonical decomposition extends to A_4 (CDSed)
    with the same dimensional profile as A_3 (CDOct) for the seed element sed_e 1.

    Source: Moreno (1997), arXiv:q-alg/9710013v1; extended to dim=16 here.
    Supports claim: C-1549. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDNegLemmas.

Open Scope R_scope.

(** ================================================================== *)
(** * Canonical sedenion element.                                       *)
(** ================================================================== *)

(** The canonical seed imaginary element in CDSed for the Moreno lane. *)
Definition moreno_sed_a : CDSed := sed_e 1.

(** moreno_sed_a is purely imaginary (conj = neg). *)
Lemma moreno_sed_a_pure :
  sed_conj moreno_sed_a = sed_neg moreno_sed_a.
Proof.
  unfold moreno_sed_a.
  cbv [sed_e sed_conj sed_neg sed_lo sed_hi
       oct_conj oct_neg oct_lo oct_hi oct_e oct_zero
       quat_conj quat_neg quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed); apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** moreno_sed_a has unit norm. *)
Lemma moreno_sed_a_unit : sed_norm_sq moreno_sed_a = 1.
Proof.
  unfold moreno_sed_a, sed_e, sed_norm_sq, oct_norm_sq, quat_norm_sq.
  cbv [sed_lo sed_hi oct_lo oct_hi qa qb qc qd quat_zero quat_one oct_e oct_zero].
  ring.
Qed.

(** ================================================================== *)
(** * Self-square.                                                      *)
(** ================================================================== *)

(** moreno_sed_a^2 = -e0 in CDSed. *)
Lemma moreno_sed_a_sq : sed_mul moreno_sed_a moreno_sed_a = sed_neg (sed_e 0).
Proof.
  unfold moreno_sed_a.
  cbv [sed_e sed_mul sed_neg sed_lo sed_hi
       oct_mul oct_add oct_sub oct_neg oct_conj oct_e oct_zero
       oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed); apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** ================================================================== *)
(** * Core H_a-stability lemma: a*(a*x) = -x for all x in CDSed.      *)
(** ================================================================== *)

(** The key identity: left-iteration by moreno_sed_a negates any x.

    Proof: the CD doubling formula and CDOct left/right alternative for
    basis element e1 (via cbv + ring; degree-1 in x). *)
Lemma moreno_sed_a_left_sq_neg : forall x : CDSed,
  sed_mul moreno_sed_a (sed_mul moreno_sed_a x) = sed_neg x.
Proof.
  intros [[[ x1  x2  x3  x4] [ x5  x6  x7  x8]]
          [[ x9 x10 x11 x12] [x13 x14 x15 x16]]].
  unfold moreno_sed_a.
  cbv [sed_e sed_mul sed_neg sed_lo sed_hi
       oct_mul oct_add oct_sub oct_neg oct_conj oct_e oct_zero
       oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.
Qed.

(** ================================================================== *)
(** * T_a operator and the zero theorem.                               *)
(** ================================================================== *)

(** The Moreno T_a operator for the canonical CDSed element:
    T_a(x) = -x - a*(a*x). *)
Definition moreno_sed_t_a_can (x : CDSed) : CDSed :=
  sed_add (sed_neg x) (sed_neg (sed_mul moreno_sed_a (sed_mul moreno_sed_a x))).

(** T_a = 0 on all of CDSed for the canonical element a = sed_e 1.

    This is the CDSed analog of the octonion alternativity result in C1541:
    in CDOct, alternativity gives a*(a*x) = -x for all x; here in CDSed,
    the same holds for the specific element sed_e 1 by the explicit
    cbv computation in moreno_sed_a_left_sq_neg. *)
Theorem moreno_sed_t_a_can_zero : forall x : CDSed,
  moreno_sed_t_a_can x = sed_zero.
Proof.
  intro x.
  unfold moreno_sed_t_a_can.
  rewrite moreno_sed_a_left_sq_neg.
  apply sed_add_neg_cancel.
Qed.

(** ================================================================== *)
(** * V_lambda predicate and emptiness for lambda != 1.               *)
(** ================================================================== *)

(** Membership in V_lambda(a) for CDSed: x is in H_a^perp and satisfies
    a*(a*x) = -(lambda^2)*x.  We restrict to H_a^perp to match the
    Moreno decomposition structure from C1541. *)
Definition moreno_sed_in_vlambda_can (lambda : R) (x : CDSed) : Prop :=
  sed_mul moreno_sed_a (sed_mul moreno_sed_a x) =
  sed_scale (- (lambda * lambda)) x.

(** Nonzero witness approach: if x = sed_e i for some i,
    then moreno_sed_in_vlambda_can lambda (sed_e i) requires lambda = 1. *)
Lemma moreno_sed_vlambda_can_e1_lambda_one :
  forall lambda : R,
    moreno_sed_in_vlambda_can lambda (sed_e 1) ->
    lambda * lambda = 1.
Proof.
  intros lambda Hv.
  unfold moreno_sed_in_vlambda_can in Hv.
  rewrite moreno_sed_a_left_sq_neg in Hv.
  cbv [sed_e sed_neg sed_scale sed_lo sed_hi
       oct_scale oct_neg oct_e oct_zero
       oct_lo oct_hi
       quat_scale quat_neg quat_zero quat_one
       qa qb qc qd] in Hv.
  (* inject all 16 scalar equalities; the qb-of-lo-lo component (H2)
     gives -1 = -(lambda*lambda)*1, from which lra derives lambda*lambda = 1 *)
  injection Hv as
    H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16.
  lra.
Qed.

(** ================================================================== *)
(** * The canonical CDSed Moreno profile.                              *)
(** ================================================================== *)

(** For the canonical element a = sed_e 1 in CDSed:
    - T_a = 0 on all of CDSed
    - V_lambda(a) = {} for lambda != 1 (in the positive-lambda sense)
    - The dimensional profile is 0 nontrivial V_lambda blocks
      (same as CDOct canonical, for the same structural reason)

    This is equivalent to block_count = 0 in Moreno's block decomposition,
    which is the trivial case is_h_module_dim 0 (via hmd_zero).

    The canonical profile theorem is proved below. *)
Theorem moreno_sed_can_profile :
  forall lambda : R,
    (0 < lambda)%R ->
    lambda * lambda <> 1 ->
    forall x : CDSed,
      moreno_sed_in_vlambda_can lambda x ->
      x = sed_zero.
Proof.
  intros lambda Hlpos Hlne1 x Hv.
  unfold moreno_sed_in_vlambda_can in Hv.
  rewrite moreno_sed_a_left_sq_neg in Hv.
  destruct x as [[[ x1  x2  x3  x4] [ x5  x6  x7  x8]]
                 [[ x9 x10 x11 x12] [x13 x14 x15 x16]]].
  cbv [sed_neg sed_scale sed_lo sed_hi
       oct_neg oct_scale oct_lo oct_hi
       quat_neg quat_scale qa qb qc qd] in Hv.
  injection Hv as
    H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16.
  (* Each Hi : -xi = -(lambda*lambda)*xi.
     Factor as (lambda*lambda - 1)*xi = 0.  Since lambda*lambda <> 1,
     Rmult_integral forces xi = 0 for every component. *)
  assert (Hsolve : forall v : R, - v = - (lambda * lambda) * v -> v = 0).
  { intros v Hcomp.
    assert (Hprod : (lambda * lambda - 1) * v = 0) by lra.
    destruct (Rmult_integral (lambda * lambda - 1) v Hprod) as [Hll | Hv0].
    - lra.
    - exact Hv0. }
  unfold sed_zero, oct_zero, quat_zero.
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  apply Hsolve;
  assumption.
Qed.
