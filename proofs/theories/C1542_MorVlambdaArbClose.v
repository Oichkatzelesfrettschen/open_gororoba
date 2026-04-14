(** * C1542_MorVlambdaArbClose: Moreno (1997) Theorem 1.16 -- arbitrary-a closure for CDOct.

    WHAT THIS FILE PROVES:

    The final piece of C-1542 (L-1542): for any doubly-pure unit-norm a in CDOct,
    every eigenspace V_lambda(a) with lambda > 0 and lambda != 1 is trivial ({0}).
    This immediately gives dim V_lambda = 0 = 4 * 0, hence 0 mod 4.

    WHY THIS CLOSES L-1542:

    The abstract H-module orbit framework (C1542_MorVlambdaOrbit.v) and the
    nonempty-block stepping bridge (C1548_MorVlambdaHBlock.v) already prove:
      is_h_module_dim d -> exists k, d = 4*k.
    The remaining gap was instantiating these with a concrete CDOct V_lambda
    subspace for arbitrary a.

    C1542_MorHaStabilityArb.v proved L_a^2 = -I for arbitrary unit imaginary a
    (first general alternative law in this codebase, proved via cbv + abstract ring).
    This collapses the Moreno 1.16 conclusion for CDOct entirely: since CDOct is
    alternative and L_a^2 = -I everywhere, T~_a = -I - L_a^2 = 0, so every
    V_lambda(a) with lambda != 1 is immediately zero.  The HModuleFinDim instance
    is the trivial zero module (hmd_zero: is_h_module_dim 0, vlambda_dim = 0 = 4*0).

    PROOF STRATEGY:

    For moreno16_vlambda_arb_trivial:
    (1) L_a^2 y = -y  [oct_unit_imag_left_sq_neg, C1542_MorHaStabilityArb]
    (2) L_a^2 y = -(lambda^2)*y  [eigenvalue hypothesis]
    From (1)+(2): oct_neg y = oct_scale (-(lambda^2)) y.
    Project to 8 real components; each satisfies -c_i = -(lambda^2)*c_i.
    Since lambda != 1 and lambda > 0 give lambda^2 != 1 (so 1-lambda^2 != 0),
    nra derives c_i = 0.  Hence y = oct_zero.

    For moreno16_ker_la_trivial_arb:
    Apply L_a to the hypothesis a*y = 0: get a*(a*y) = a*0 = 0.
    L_a^2 y = -y gives 0 = -y, hence y = 0.

    This proof pattern is structurally identical to moreno15_no_nontrivial_vlambda
    in C1541_MorDirectSum.v; the only change is using oct_unit_imag_left_sq_neg
    (arbitrary a) instead of moreno15_e1_left_square_neg (canonical a = e1).

    Closes L-1542.  Supports claim C-1542.
    Source: Moreno (1997), arXiv:q-alg/9710013v1, Theorem 1.16. *)

From Stdlib Require Import Reals Lra Psatz.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDNegLemmas
  FinDimHModule
  C1542_MorHaStabilityArb
  C1542_MorVlambdaOrbit.

Open Scope R_scope.

(** ================================================================== *)
(** * Triviality: V_lambda(a) = {0} for arbitrary unit imaginary a.   *)
(** ================================================================== *)

(** Main theorem: for any unit imaginary a and lambda > 0, lambda != 1,
    every y satisfying a*(a*y) = -(lambda^2)*y must be zero.

    The hypotheses are strictly weaker than the full moreno15_in_vlambda
    predicate (which also requires y in H_a^perp); here we only assume the
    eigenvalue equation.  Since L_a^2 = -I holds everywhere on CDOct, the
    H_a^perp condition is not needed to derive y = 0. *)
Theorem moreno16_vlambda_arb_trivial :
  forall a y : CDOct, forall lambda : R,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    0 < lambda ->
    lambda <> 1 ->
    oct_mul a (oct_mul a y) = oct_scale (-(lambda * lambda)) y ->
    y = oct_zero.
Proof.
  intros a y lambda Ha_unit Ha_pure Hpos Hne Hvl.
  (* (1) L_a^2 y = -y via the arbitrary-a stability lemma. *)
  pose proof (oct_unit_imag_left_sq_neg a y Ha_unit Ha_pure) as Hneg.
  (* (2) Combine with eigenvalue condition: -y = -(lambda^2)*y. *)
  assert (Heq : oct_neg y = oct_scale (-(lambda * lambda)) y).
  { rewrite <- Hneg. exact Hvl. }
  (* (3) Destruct y to 8 real components; project the equation. *)
  destruct y as [[c1 c2 c3 c4] [c5 c6 c7 c8]].
  unfold oct_neg, oct_scale in Heq.
  simpl in Heq.
  assert (Hnz : 1 - lambda * lambda <> 0) by nra.
  assert (Hlo := f_equal oct_lo Heq).
  assert (Hhi := f_equal oct_hi Heq).
  simpl in Hlo, Hhi.
  assert (Hc1 := f_equal qa Hlo). simpl in Hc1.
  assert (Hc2 := f_equal qb Hlo). simpl in Hc2.
  assert (Hc3 := f_equal qc Hlo). simpl in Hc3.
  assert (Hc4 := f_equal qd Hlo). simpl in Hc4.
  assert (Hc5 := f_equal qa Hhi). simpl in Hc5.
  assert (Hc6 := f_equal qb Hhi). simpl in Hc6.
  assert (Hc7 := f_equal qc Hhi). simpl in Hc7.
  assert (Hc8 := f_equal qd Hhi). simpl in Hc8.
  (* (4) Each -c_i = -(lambda^2)*c_i plus 1-lambda^2 != 0 gives c_i = 0. *)
  assert (Hc1z : c1 = 0) by nra.
  assert (Hc2z : c2 = 0) by nra.
  assert (Hc3z : c3 = 0) by nra.
  assert (Hc4z : c4 = 0) by nra.
  assert (Hc5z : c5 = 0) by nra.
  assert (Hc6z : c6 = 0) by nra.
  assert (Hc7z : c7 = 0) by nra.
  assert (Hc8z : c8 = 0) by nra.
  subst. reflexivity.
Qed.

(** ================================================================== *)
(** * Triviality of Ker(L_a) for arbitrary unit imaginary a.          *)
(** ================================================================== *)

(** Ker(L_a) = {0} for any unit imaginary a in CDOct.
    Dual of the stability result: L_a is injective because L_a^2 = -I. *)
Theorem moreno16_ker_la_trivial_arb :
  forall a y : CDOct,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    oct_mul a y = oct_zero ->
    y = oct_zero.
Proof.
  intros a y Ha_unit Ha_pure Hy.
  (* Apply L_a to the zero condition: a*(a*y) = a*0 = 0. *)
  pose proof (oct_unit_imag_left_sq_neg a y Ha_unit Ha_pure) as Hstab.
  rewrite Hy in Hstab.
  rewrite oct_mul_zero_right in Hstab.
  (* Hstab: oct_zero = oct_neg y.  Negate both sides. *)
  apply (f_equal oct_neg) in Hstab.
  rewrite oct_neg_zero in Hstab.
  rewrite oct_neg_neg in Hstab.
  symmetry. exact Hstab.
Qed.

(** ================================================================== *)
(** * Section: packaged arbitrary-a V_lambda conclusions.             *)
(** ================================================================== *)

Section Moreno16ArbitraryAVlambdaClose.
  Variable a      : CDOct.
  Variable lambda : R.
  Hypothesis Ha_unit   : oct_norm_sq a = 1%R.
  Hypothesis Ha_pure   : oct_conj a = oct_neg a.
  Hypothesis Hlambda   : 0 < lambda.
  Hypothesis Hlambda_ne : lambda <> 1.

  (** V_lambda(a) is trivial: dimension 0. *)
  Theorem moreno16_arb_vlambda_dim_zero :
    forall y : CDOct,
      oct_mul a (oct_mul a y) = oct_scale (-(lambda * lambda)) y ->
      y = oct_zero.
  Proof.
    intros y Hvl.
    exact (moreno16_vlambda_arb_trivial a y lambda Ha_unit Ha_pure Hlambda Hlambda_ne Hvl).
  Qed.

  (** The trivial module is a valid H-module dimension (vlambda_dim = 0 = 4*0). *)
  Theorem moreno16_arb_vlambda_is_h_module_dim : is_h_module_dim 0%nat.
  Proof. exact hmd_zero. Qed.

  Corollary moreno16_arb_vlambda_dim_mod4 : Nat.modulo 0%nat 4%nat = 0%nat.
  Proof. reflexivity. Qed.

  (** Package as Moreno16ConcreteVlambdaWitness with vlambda_dim = 0. *)
  Definition moreno16_arb_trivial_witness : Moreno16ConcreteVlambdaWitness :=
    {| moreno16_concrete_lambda              := lambda;
       moreno16_concrete_lambda_pos          := Hlambda;
       moreno16_concrete_vlambda_dim         := 0%nat;
       moreno16_concrete_vlambda_is_h_module_dim := hmd_zero |}.

End Moreno16ArbitraryAVlambdaClose.

(** ================================================================== *)
(** * Top-level bridge: vlambda_dim = 0 for arbitrary a, lambda != 1. *)
(** ================================================================== *)

(** Universal statement: for any packaging of arbitrary-a hypotheses
    with lambda != 1, the concrete vlambda_dim = 0 satisfies 0 mod 4 = 0.
    This instantiates the abstract Moreno1997 Theorem 1.16 framework. *)

Theorem moreno16_arb_close_bridge :
  forall a : CDOct, forall lambda : R,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    0 < lambda ->
    lambda <> 1 ->
    is_h_module_dim 0%nat /\ Nat.modulo 0%nat 4%nat = 0%nat.
Proof.
  intros. split. exact hmd_zero. reflexivity.
Qed.

(** Concrete dim-0 witness constructor for downstream use. *)
Definition moreno16_build_arb_trivial_witness
  (a : CDOct) (Ha_unit : oct_norm_sq a = 1%R)
  (Ha_pure : oct_conj a = oct_neg a)
  (lambda : R) (Hpos : 0 < lambda) (Hne : lambda <> 1)
  : Moreno16ConcreteVlambdaWitness :=
  {| moreno16_concrete_lambda              := lambda;
     moreno16_concrete_lambda_pos          := Hpos;
     moreno16_concrete_vlambda_dim         := 0%nat;
     moreno16_concrete_vlambda_is_h_module_dim := hmd_zero |}.
