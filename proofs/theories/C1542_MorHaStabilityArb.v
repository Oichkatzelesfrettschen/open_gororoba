(** * C1542_MorHaStabilityArb: Moreno (1997) Theorem 1.16 -- arbitrary-a H_a-stability.

    WHAT THIS FILE PROVES:

    For any doubly-pure unit-norm element a in CDOct (an octonion satisfying
    oct_norm_sq a = 1 and oct_conj a = oct_neg a):

      (1)  a * a = -e0          (unit imaginary self-square)
      (2)  a*(a*z) = -z         for all z in CDOct  (L_a^2 = -I on CDOct)
      (3)  a_sed*(a_sed*z) = -z for all z in CDSed  (L_{a_sed}^2 = -I on CDSed)
           where a_sed := mkSed a oct_zero.

    WHY THIS MATTERS:

    These results are the "H_a-stability" prerequisites for Moreno Theorem 1.16.
    C1542_MorVlambdaOrbit.v already provides the abstract record
    Moreno16ArbitraryAGeometricVlambdaData, which asserts dim(V_lambda) = 4*k.
    The present file proves the underlying algebraic identity that justifies why
    T~_a (restriction of T_a = -I - L_a^2 to H_a^perp) is the zero operator,
    so that the eigenspace structure from the abstract record can be instantiated
    with a concrete CDOct element.

    PROOF STRATEGY:

    (1) is derived from oct_mul_conj_norm (x*conj(x) = ||x||^2 * e0)
        combined with the purity hypothesis oct_conj a = oct_neg a.
    (2) follows from the general left-alternative law for octonions
        (proved by cbv + abstract ring at 16 variables, degree 3)
        applied to (1).
    (3) uses the explicit CD doubling formula for sed_mul when the first
        factor has vanishing hi-part: (a, 0)*(c, d) = (a*c, d*a).
        Applying (2) for the lo-half and the right-alternative law for
        the hi-half gives (a,0)*((a,0)*z) = -z.

    DEPENDENCY NOTE:

    The general left/right alternative laws proved here are the FIRST such
    proofs for arbitrary octonion pairs in this codebase; C910_OctonionAlternative.v
    only establishes them for specific basis elements e1 and e4.  The cbv
    unfolding at 16 scalar variables is memory-safe because abstract ring
    seals the proof term as an opaque certificate rather than inlining the
    full polynomial witness.

    Closes the algebraic prerequisite gap for L-1542.
    Supports claim C-1542.
    Source: Moreno (1997), arXiv:q-alg/9710013v1, Theorem 1.16. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDLinearLemmas CDNegLemmas.

Open Scope R_scope.

(** ================================================================== *)
(** * Helper: zero annihilators.                                        *)
(** ================================================================== *)

Lemma oct_mul_zero_right : forall x : CDOct, oct_mul x oct_zero = oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_mul oct_zero oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma oct_mul_zero_left : forall x : CDOct, oct_mul oct_zero x = oct_zero.
Proof.
  intros [[z1 z2 z3 z4] [z5 z6 z7 z8]].
  cbv [oct_mul oct_zero oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** ================================================================== *)
(** * Helper: norm formula and scale simplifications.                  *)
(** ================================================================== *)

(** a * conj(a) = ||a||^2 * e0 (Cayley-Dickson norm identity). *)
Lemma oct_mul_conj_norm :
  forall a : CDOct,
    oct_mul a (oct_conj a) = oct_scale (oct_norm_sq a) (oct_e 0).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_mul oct_conj oct_norm_sq oct_scale oct_e oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_scale quat_norm_sq
       quat_zero quat_one qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** oct_scale 1 (oct_e 0) = oct_e 0. *)
Lemma oct_scale_one_e0 : oct_scale 1%R (oct_e 0) = oct_e 0.
Proof.
  cbv [oct_scale oct_e quat_scale quat_one quat_zero oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** ================================================================== *)
(** * Helper: left and right identity.                                  *)
(** ================================================================== *)

(** e0 * z = z (left identity at octonion level). *)
Lemma oct_mul_e0_left : forall z : CDOct, oct_mul (oct_e 0) z = z.
Proof.
  intros [[z1 z2 z3 z4] [z5 z6 z7 z8]].
  cbv [oct_mul oct_e oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** (-e0) * z = -z. *)
Lemma oct_mul_neg_e0_left : forall z : CDOct,
  oct_mul (oct_neg (oct_e 0)) z = oct_neg z.
Proof.
  intro z.
  rewrite oct_neg_mul_left.
  rewrite oct_mul_e0_left.
  reflexivity.
Qed.

(** z * (-e0) = -z. *)
Lemma oct_mul_neg_e0_right : forall z : CDOct,
  oct_mul z (oct_neg (oct_e 0)) = oct_neg z.
Proof.
  intro z.
  rewrite oct_neg_mul_right.
  change (oct_e 0) with (mkOct quat_one quat_zero).
  rewrite oct_mul_one_right.
  reflexivity.
Qed.

(** ================================================================== *)
(** * General alternative laws for CDOct.                              *)
(** ================================================================== *)

(** Left alternative law: a*(a*z) = (a*a)*z  for ALL a, z in CDOct.

    This is the first proof of the GENERAL left alternative law in this
    codebase.  C910_OctonionAlternative.v only establishes it for basis
    elements e1 and e4; the arbitrary case requires a full 16-variable
    degree-3 computation, which is sealed as an opaque certificate via
    abstract ring to avoid large proof-term blowup. *)
Lemma oct_left_alt_gen : forall a z : CDOct,
  oct_mul a (oct_mul a z) = oct_mul (oct_mul a a) z.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[z1 z2 z3 z4] [z5 z6 z7 z8]].
  cbv [oct_mul oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); abstract ring.
Qed.

(** Right alternative law: (z*a)*a = z*(a*a)  for ALL a, z in CDOct.

    Symmetric to the left case; same proof strategy. *)
Lemma oct_right_alt_gen : forall a z : CDOct,
  oct_mul (oct_mul z a) a = oct_mul z (oct_mul a a).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[z1 z2 z3 z4] [z5 z6 z7 z8]].
  cbv [oct_mul oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); abstract ring.
Qed.

(** ================================================================== *)
(** * Unit imaginary self-square.                                       *)
(** ================================================================== *)

(** For a with unit norm and pure imaginary (conj a = -a):  a*a = -e0.

    Proof: a*conj(a) = ||a||^2 * e0 = e0.
           a*(-a) = -(a*a), so a*a = -e0. *)
Lemma oct_unit_imag_sq :
  forall a : CDOct,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    oct_mul a a = oct_neg (oct_e 0).
Proof.
  intros a Hnorm Hpure.
  assert (Hstep : oct_mul a (oct_neg a) = oct_e 0).
  { rewrite <- Hpure.
    rewrite oct_mul_conj_norm.
    rewrite Hnorm.
    apply oct_scale_one_e0. }
  (* Hstep : oct_mul a (oct_neg a) = oct_e 0
     Forward rewrite: oct_mul a (oct_neg a) = oct_neg (oct_mul a a). *)
  rewrite oct_neg_mul_right in Hstep.
  (* Hstep : oct_neg (oct_mul a a) = oct_e 0
     Apply oct_neg to both sides: oct_mul a a = oct_neg (oct_e 0). *)
  apply (f_equal oct_neg) in Hstep.
  rewrite oct_neg_neg in Hstep.
  exact Hstep.
Qed.

(** ================================================================== *)
(** * The L_a^2 = -I theorem for CDOct.                               *)
(** ================================================================== *)

(** For unit imaginary a: a*(a*z) = -z  for all z in CDOct.

    Chain: left alt -> unit imag sq -> mul by -e0. *)
Lemma oct_unit_imag_left_sq_neg :
  forall a z : CDOct,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    oct_mul a (oct_mul a z) = oct_neg z.
Proof.
  intros a z Hnorm Hpure.
  rewrite oct_left_alt_gen.
  rewrite (oct_unit_imag_sq a Hnorm Hpure).
  apply oct_mul_neg_e0_left.
Qed.

(** ================================================================== *)
(** * Sedenion CD doubling reduction for lo-zero first factor.         *)
(** ================================================================== *)

(** The CD doubling formula for sed_mul:
      (a, b)*(c, d) = (a*c - conj(d)*b, d*a + b*conj(c)).
    When b = oct_zero this simplifies to:
      (a, 0)*(c, d) = (a*c, d*a).

    This lemma makes that simplification explicit for use in the
    sedenion stability proof below. *)
Lemma sed_mul_oct_lo_zero :
  forall (a z_lo z_hi : CDOct),
    sed_mul (mkSed a oct_zero) (mkSed z_lo z_hi) =
    mkSed (oct_mul a z_lo) (oct_mul z_hi a).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[l1 l2 l3 l4] [l5 l6 l7 l8]]
         [[h1 h2 h3 h4] [h5 h6 h7 h8]].
  cbv [sed_mul sed_lo sed_hi oct_mul oct_conj oct_zero oct_neg oct_add
       oct_lo oct_hi quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

(** ================================================================== *)
(** * The L_{a_sed}^2 = -I theorem for CDSed.                         *)
(** ================================================================== *)

(** For unit imaginary a in CDOct, with a_sed = (a, 0) in CDSed:
      a_sed * (a_sed * z) = -z   for all z in CDSed.

    Proof:
      Step 1: a_sed * z = (a * z_lo, z_hi * a)        [sed_mul_oct_lo_zero]
      Step 2: a_sed * (a*z_lo, z_hi*a)
              = (a*(a*z_lo), (z_hi*a)*a)               [sed_mul_oct_lo_zero]
              = (-(z_lo), -(z_hi))                     [left alt + right alt]
              = -(z_lo, z_hi).                         [def sed_neg] *)
Lemma sed_unit_oct_lo_left_sq_neg :
  forall (a_lo : CDOct) (z : CDSed),
    oct_norm_sq a_lo = 1%R ->
    oct_conj a_lo = oct_neg a_lo ->
    sed_mul (mkSed a_lo oct_zero) (sed_mul (mkSed a_lo oct_zero) z) = sed_neg z.
Proof.
  intros a_lo [z_lo z_hi] Hnorm Hpure.
  rewrite (sed_mul_oct_lo_zero a_lo z_lo z_hi).
  rewrite (sed_mul_oct_lo_zero a_lo (oct_mul a_lo z_lo) (oct_mul z_hi a_lo)).
  rewrite oct_left_alt_gen.
  rewrite oct_right_alt_gen.
  rewrite (oct_unit_imag_sq a_lo Hnorm Hpure).
  rewrite oct_mul_neg_e0_left.
  rewrite oct_mul_neg_e0_right.
  reflexivity.
Qed.

(** ================================================================== *)
(** * Packaged Section: H_a-stability for arbitrary unit imaginary a.  *)
(** ================================================================== *)

(** Groups the two stability theorems under a single parameter section
    for easy import into L-1542 and L-1546 instantiation files. *)

Section Moreno16HaStabilityArbitrary.
  Variable a : CDOct.
  Hypothesis Ha_unit : oct_norm_sq a = 1%R.
  Hypothesis Ha_pure : oct_conj a = oct_neg a.

  (** L_a^2 = -I on CDOct. *)
  Theorem moreno16_ha_left_sq_neg :
    forall z : CDOct,
      oct_mul a (oct_mul a z) = oct_neg z.
  Proof.
    intro z.
    exact (oct_unit_imag_left_sq_neg a z Ha_unit Ha_pure).
  Qed.

  (** L_{a_sed}^2 = -I on CDSed where a_sed = (a, 0). *)
  Theorem moreno16_ha_sed_stability :
    forall z : CDSed,
      sed_mul (mkSed a oct_zero) (sed_mul (mkSed a oct_zero) z) = sed_neg z.
  Proof.
    intro z.
    exact (sed_unit_oct_lo_left_sq_neg a z Ha_unit Ha_pure).
  Qed.
End Moreno16HaStabilityArbitrary.
