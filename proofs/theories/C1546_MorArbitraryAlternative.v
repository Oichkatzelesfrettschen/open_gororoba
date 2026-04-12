(** * C1546_MorArbitraryAlternative: Moreno Theorem 2.9 for arbitrary unit imaginary pairs.

    WHAT THIS FILE PROVES:

    Instantiates the abstract eigenvalue-iff (moreno29_abstract_iff_anticomm from
    C1546_MorEigenIFF.v) for arbitrary unit imaginary pairs a, b in CDOct, by
    discharging the two standing hypotheses Ha_square_all and Hb_square_all using
    the general L_a^2 = -I result from C1542_MorHaStabilityArb.v.

    CONTEXT:

    C1546_MorEigenIFF.v establishes (inside Section Moreno29Abstract) that the
    abstract eigenvalue iff holds under two black-box hypotheses:
      Ha_square_all : forall z, a*(a*z) = -z
      Hb_square_all : forall z, b*(b*z) = -z

    Before C1542_MorHaStabilityArb.v was available, those hypotheses could only be
    discharged for specific basis elements (e1, e4) via C910_OctonionAlternative.v.
    Now that oct_unit_imag_left_sq_neg proves the general case for any unit imaginary
    element in CDOct, the abstract iff has a clean instantiation for arbitrary pairs.

    RESULT:

    For any two doubly-pure unit-norm octonions a, b satisfying the anti-commutation
    hypotheses required by Moreno, and any pure y anti-commuting with both a and b,
    the Moreno zero-divisor witness equation system is equivalent to y being a
    -2-eigenvector of L_{a+b}^2.

    This closes the "discharge Ha_square_all" gap documented in L-1546.

    Supports claim C-1546.
    Source: Moreno (1997), arXiv:q-alg/9710013v1, Theorem 2.9. *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDLinearLemmas CDNegLemmas
  C1542_MorHaStabilityArb C1546_MorEigenIFF.

Open Scope R_scope.

(** ================================================================== *)
(** * Arbitrary unit imaginary pair section.                            *)
(** ================================================================== *)

Section Moreno29ArbitraryUnitImag.
  Variables a b : CDOct.
  Hypothesis Ha_unit : oct_norm_sq a = 1%R.
  Hypothesis Ha_pure : oct_conj a = oct_neg a.
  Hypothesis Hb_unit : oct_norm_sq b = 1%R.
  Hypothesis Hb_pure : oct_conj b = oct_neg b.

  (** ================================================================ *)
  (** * Discharge Ha_square_all and Hb_square_all.                     *)
  (** ================================================================ *)

  (** The section hypothesis Ha_square_all from C1546_MorEigenIFF
      (forall z, a*(a*z) = -z) holds for arbitrary unit imaginary a. *)
  Lemma moreno29_a_square_all : forall z : CDOct,
    oct_mul a (oct_mul a z) = oct_neg z.
  Proof.
    intro z.
    exact (oct_unit_imag_left_sq_neg a z Ha_unit Ha_pure).
  Qed.

  (** The section hypothesis Hb_square_all from C1546_MorEigenIFF
      (forall z, b*(b*z) = -z) holds for arbitrary unit imaginary b. *)
  Lemma moreno29_b_square_all : forall z : CDOct,
    oct_mul b (oct_mul b z) = oct_neg z.
  Proof.
    intro z.
    exact (oct_unit_imag_left_sq_neg b z Hb_unit Hb_pure).
  Qed.

  (** ================================================================ *)
  (** * Instantiated eigenvalue iff.                                    *)
  (** ================================================================ *)

  (** The full Moreno 2.9 abstract iff, now with discharged hypotheses.

      For any pure y anticommuting with both a and b (and with b*y
      anti-commuting under right-multiplication by a), the Moreno
      zero-divisor witness equation system is equivalent to y being
      a -2-eigenvector of L_{a+b}^2.

      The proof is a single application of moreno29_abstract_iff_anticomm
      with the two discharged square-negative lemmas as witnesses for the
      formerly-open hypotheses. *)
  Theorem moreno29_arbitrary_iff :
    forall y : CDOct,
      oct_conj y = oct_neg y ->
      oct_mul y a = oct_neg (oct_mul a y) ->
      oct_mul y b = oct_neg (oct_mul b y) ->
      oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y)) ->
      (moreno29_witness_eqns a b (moreno29_partner a b y) y <->
       y <> oct_zero /\
       oct_mul (moreno29_c a b) (oct_mul (moreno29_c a b) y) = oct_scale (-2) y).
  Proof.
    intros y Hy_pure Hya Hyb Hbya.
    exact (moreno29_abstract_iff_anticomm
             a b
             moreno29_a_square_all
             moreno29_b_square_all
             y Ha_pure Hb_pure Hy_pure Hya Hyb Hbya).
  Qed.

End Moreno29ArbitraryUnitImag.
