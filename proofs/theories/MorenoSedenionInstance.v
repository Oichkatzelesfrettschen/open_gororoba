(** * MorenoSedenionInstance: a CONCRETE sedenion model of the Moreno
      inner-product interface, instantiating C1539's abstract MorSkewSymm
      functor.

    The review found that the abstract Moreno functors (MorSkewSymm over
    CDAlgMoreno, MorZDSymmetry over CDAlgInnerTrace) were never instantiated
    with the actual Cayley-Dickson algebra, so their interface axioms were
    never discharged -- the abstract Proposition 1.7 was a conditional whose
    hypotheses about the concrete algebra were never shown to hold.

    This file discharges the CDAlgMoreno interface for the dim-16 sedenions:
    the load-bearing adjoint identities are the already-verified
    CDInnerProduct.sed_inner_adj_left / _adj_right (proved by the tower split
    quat_inner -> oct_inner -> sed_inner, avoiding the 48-variable ring
    blowup), the negation/zero laws come from CDNegLemmas, and the remaining
    inner-product laws are direct component computations.  Instantiating
    MorSkewSymm on this model yields Moreno's Proposition 1.7 (skew-symmetry
    of L_x and R_x) for the REAL sedenions, not merely for an abstract
    structure -- closing the never-instantiated gap for the skew-symmetry
    lane.

    (The kernel-equality MorSkewSymm.ker_lx_eq_ker_rx still rests on the
    functor-internal im_lx_eq_im_rx axioms encoding Moreno Thm 1.15; this
    instantiation does not discharge those -- they are a separate, honestly
    labeled gap -- but the skew-symmetry results it exposes do not use them,
    as Print Assumptions confirms.) *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import CDInnerProduct CDNegLemmas C1539_MorSkewSymm.
Open Scope R_scope.

(** The concrete sedenion model of CDAlgMoreno. *)
Module SedMoreno <: CDAlgMoreno.
  Definition A     := CDSed.
  Definition mul   := sed_mul.
  Definition conj  := sed_conj.
  Definition neg   := sed_neg.
  Definition inner := sed_inner.
  Definition zero  := sed_zero.

  (** Adjoint identities: the verified tower-split lemmas. *)
  Lemma inner_adj_left : forall x y z,
    inner (mul x y) z = inner y (mul (conj x) z).
  Proof. exact sed_inner_adj_left. Qed.

  Lemma inner_adj_right : forall x y z,
    inner (mul x y) z = inner x (mul z (conj y)).
  Proof. exact sed_inner_adj_right. Qed.

  Lemma inner_symm : forall x y, inner x y = inner y x.
  Proof. exact sed_inner_symm. Qed.

  Lemma inner_neg_left : forall x y, inner (neg x) y = - inner x y.
  Proof.
    intros x y.
    cbv [inner neg sed_inner sed_neg oct_inner oct_neg quat_inner quat_neg
         sed_lo sed_hi oct_lo oct_hi qa qb qc qd]; ring.
  Qed.

  Lemma inner_neg_right : forall x y, inner x (neg y) = - inner x y.
  Proof.
    intros x y.
    cbv [inner neg sed_inner sed_neg oct_inner oct_neg quat_inner quat_neg
         sed_lo sed_hi oct_lo oct_hi qa qb qc qd]; ring.
  Qed.

  Lemma inner_zero_left : forall y, inner zero y = 0.
  Proof.
    intros y.
    cbv [inner zero sed_inner sed_zero oct_inner oct_zero quat_inner quat_zero
         sed_lo sed_hi oct_lo oct_hi qa qb qc qd]; ring.
  Qed.

  (** Negation / zero multiplication laws from CDNegLemmas. *)
  Lemma neg_mul_left : forall x y, mul (neg x) y = neg (mul x y).
  Proof. exact sed_neg_mul_left. Qed.

  Lemma neg_mul_right : forall x y, mul x (neg y) = neg (mul x y).
  Proof. exact sed_neg_mul_right. Qed.

  Lemma neg_zero : neg zero = zero.
  Proof. exact sed_neg_zero. Qed.

  (** Non-degeneracy: <x,y> = 0 for all y forces x = 0.  Take y = x; then
      <x,x> is the sum of the 16 squared real components, which vanishes
      only when every component is zero. *)
  Lemma inner_nondeg : forall x, (forall y, inner x y = 0) -> x = zero.
  Proof.
    intros x H. specialize (H x).
    destruct x as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                   [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
    cbv [inner sed_inner oct_inner quat_inner
         sed_lo sed_hi oct_lo oct_hi qa qb qc qd] in H.
    cbv [zero sed_zero oct_zero quat_zero].
    repeat f_equal; nra.
  Qed.
End SedMoreno.

(** Instantiate the abstract skew-symmetry functor on the concrete model. *)
Module SedMorSkew := MorSkewSymm SedMoreno.

(** Moreno Proposition 1.7 for the ACTUAL sedenions: L_x is skew-symmetric
    for purely imaginary x. *)
Theorem sed_l_x_skew_symm : forall x y z : CDSed,
  sed_conj x = sed_neg x ->
  sed_inner (sed_mul x y) z = - sed_inner y (sed_mul x z).
Proof. exact SedMorSkew.l_x_skew_symm. Qed.

(** R_x is skew-symmetric for purely imaginary x. *)
Theorem sed_r_x_skew_symm : forall x y z : CDSed,
  sed_conj x = sed_neg x ->
  sed_inner (sed_mul y x) z = - sed_inner y (sed_mul z x).
Proof. exact SedMorSkew.r_x_skew_symm. Qed.

(** The companion C1538 MorZDSymmetry functor runs over the richer
    CDAlgInnerTrace (= CDTraceZero.CDAlgTraceZero plus four structural
    identities), so a concrete sedenion instantiation of THAT interface
    additionally requires concrete real_part / im_part with the
    decomposition laws (decompose, conj_decompose, im_part_perp_one,
    im_part_is_imaginary) and the purely-imaginary square identity
    imaginary_square (mul x x = scale (- norm_sq x) one for conj x = neg x),
    at dim 16.  The negation/scale/add/one laws it also needs are already in
    CDNegLemmas (sed_neg_neg, sed_add_comm, sed_add_zero_left,
    sed_add_neg_cancel, sed_neg_add, sed_scale_zero, sed_mul_one_right,
    sed_mul_zero_left/right).  That instantiation is deferred: the concrete
    sedenion ZD-symmetry it would re-derive is already established
    independently and soundly by the direct computation in
    C1538_MorZDSymmetry.C1538_full, so the abstract instantiation adds
    generality but no new concrete result. *)
