(** * C1538_MorZDSymmetry: Moreno (1997) Corollary 1.6 -- ZD symmetry.

    Moreno's Corollary 1.6 (part 1) is the four-way chain, holding for ALL
    x, y with no hypotheses:
      xy = 0  iff  yx = 0  iff  x_bar * y = 0  iff  x * y_bar = 0.

    What THIS file establishes is narrower than that full chain:
    PART I proves the two-way symmetry xy = 0 iff yx = 0 for a purely
    imaginary nonzero x (extra hypotheses Moreno does not need), plus the
    one-directional conjugate lemma xy = 0 -> x_bar*y = 0.  PART II verifies
    the concrete dim-16 instance (the Moreno-Froloff pair e3+e10, e6-e15):
    both products vanish and both factors are nonzero.

    Source: Moreno 1997, arXiv:q-alg/9710013v1, Corollary 1.6.
    Claim:  C-1538.

    Structure of this file:
    PART I:   Abstract two-way symmetry over the CDAlgInnerTrace interface.
    PART II:  Concrete sedenion verification (cbv + ring at dim=16).

    PART I derives the symmetry from the conjugate anti-automorphism plus
    the trace-zero lemma; both come from the interface / the verified
    TraceZero functor rather than being assumed.  PART II is independent and
    establishes the concrete result by direct computation. *)

(** ================================================================== *)
(** * PART I: Abstract proof from CDAlgMoreno axioms.                  *)
(** ================================================================== *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import CDTraceZero.
Open Scope R_scope.

(** Interface for the abstract ZD-symmetry derivation.

    This INCLUDES the verified trace-zero interface CDAlgTraceZero -- which
    already carries conj_antimorphism and the real/imaginary decomposition
    that PROVING trace-zero needs -- and adds the four extra structural
    identities the skew-symmetry / anti-automorphism route uses.  Folding
    conj_antimorphism into the interface (rather than asserting it as a free
    axiom mid-proof) turns it into a discharge obligation for any model;
    trace-zero is then DERIVED from this interface, not assumed. *)
Module Type CDAlgInnerTrace.
  Include CDTraceZero.CDAlgTraceZero.

  (** Right adjoint identity.  The left adjoint identity and the conjugate
      anti-automorphism are in the included interface; R_x skew-symmetry
      needs the right form too. *)
  Axiom inner_adj_right : forall x y z,
    inner (mul x y) z = inner x (mul z (conj y)).

  (** Conjugation is involutive; multiplication annihilates zero. *)
  Axiom conj_invol    : forall x, conj (conj x) = x.
  Axiom mul_zero_left  : forall x, mul zero x = zero.
  Axiom mul_zero_right : forall x, mul x zero = zero.
End CDAlgInnerTrace.

(** * ZD Symmetry derived from the axioms. *)
Module MorZDSymmetry (Alg : CDAlgInnerTrace).
  Import Alg.

  (* ================================================================== *)
  (** ** Derived inner product lemma. *)
  (* ================================================================== *)

  Lemma inner_zero_right : forall x, inner x zero = 0.
  Proof. intros. rewrite inner_symm. apply inner_zero_left. Qed.

  (* ================================================================== *)
  (** ** L_x skew-symmetric (Proposition 1.7, proved in C1539). *)
  (* ================================================================== *)

  (** Reproduced here for self-containment -- same 4-rewrite proof. *)
  Lemma l_x_skew : forall x y z,
    conj x = neg x ->
    inner (mul x y) z = - inner y (mul x z).
  Proof.
    intros x y z Hx.
    rewrite inner_adj_left, Hx, neg_mul_left, inner_neg_right.
    ring.
  Qed.

  (** R_x skew-symmetric -- mirror proof using right adjoint. *)
  Lemma r_x_skew : forall x y z,
    conj x = neg x ->
    inner (mul y x) z = - inner y (mul z x).
  Proof.
    intros x y z Hx.
    rewrite inner_adj_right, Hx, neg_mul_right, inner_neg_right.
    ring.
  Qed.

  (* ================================================================== *)
  (** ** xy = 0 implies y perp Im(L_x) (provable). *)
  (* ================================================================== *)

  Lemma zd_implies_perp_L : forall x y,
    conj x = neg x ->
    mul x y = zero ->
    forall z, inner y (mul x z) = 0.
  Proof.
    intros x y Hpure Hxy z.
    assert (H0 : inner (mul x y) z = 0).
    { rewrite Hxy. apply inner_zero_left. }
    rewrite (l_x_skew x y z Hpure) in H0.
    lra.
  Qed.

  (** yx = 0 implies y perp Im(R_x). *)
  Lemma zd_implies_perp_R : forall x y,
    conj x = neg x ->
    mul y x = zero ->
    forall z, inner y (mul z x) = 0.
  Proof.
    intros x y Hpure Hyx z.
    assert (H0 : inner (mul y x) z = 0).
    { rewrite Hyx. apply inner_zero_left. }
    rewrite (r_x_skew x y z Hpure) in H0.
    lra.
  Qed.

  (** y perp Im(R_x) implies yx = 0 (from r_x_skew + non-degeneracy). *)
  Lemma perp_R_implies_yx_zero : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul z x) = 0) ->
    mul y x = zero.
  Proof.
    intros x y Hpure Hperp.
    apply inner_nondeg.
    intros z.
    rewrite (r_x_skew x y z Hpure).
    specialize (Hperp z). lra.
  Qed.

  (** y perp Im(L_x) implies xy = 0. *)
  Lemma perp_L_implies_xy_zero : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul x z) = 0) ->
    mul x y = zero.
  Proof.
    intros x y Hpure Hperp.
    apply inner_nondeg.
    intros z.
    rewrite (l_x_skew x y z Hpure).
    specialize (Hperp z). lra.
  Qed.

  (* ================================================================== *)
  (** ** The conjugate anti-automorphism + trace-zero route.            *)
  (* ================================================================== *)

  (** Instead of the Im(L_x) = Im(R_x) approach, we use:
      1. conj(xy) = conj(y) * conj(x)  [anti-automorphism]
      2. xy = 0 and x purely imaginary and x != 0 => y purely imaginary
         [trace-zero, proved in CDTraceZero.v]
      3. For both x AND y purely imaginary:
         conj(xy) = (-y)*(-x) = yx, so xy = 0 => yx = 0.

      This completely replaces the Im(L_x) = Im(R_x) argument. *)

  (** Conjugate anti-automorphism is now an obligation of the
      CDAlgInnerTrace interface (inherited from CDAlgTraceZero via Include),
      so any model must discharge it -- concretely CDConjAntimorph.v's
      sed_conj_antimorphism -- rather than it being assumed here. *)

  (** Trace-zero is DERIVED, not axiomatized: instantiate the verified
      TraceZero functor on this same algebra.  Its zd_implies_y_imaginary is
      exactly the statement the ZD-symmetry argument needs (so named in
      CDTraceZero.v: "the form needed by C1538_MorZDSymmetry.v"). *)
  Module TZ := CDTraceZero.TraceZero Alg.
  Definition zd_trace_zero := TZ.zd_implies_y_imaginary.

  (** Double negation in products: (-a)(-b) = ab. *)
  Lemma neg_neg_mul : forall a b,
    mul (neg a) (neg b) = mul a b.
  Proof.
    intros a b.
    rewrite neg_mul_left. rewrite neg_mul_right.
    apply neg_neg.
  Qed.

  (* ================================================================== *)
  (** ** Ker(L_x) subset Ker(R_x) -- the main direction.               *)
  (* ================================================================== *)

  (** If x purely imaginary, nonzero, and xy = 0, then yx = 0.

      Proof:
      1. xy = 0                                        [given]
      2. conj(y) = neg(y) = -y                         [zd_trace_zero]
      3. conj(xy) = conj(y)*conj(x) = (-y)*(-x) = yx  [anti-automorphism]
      4. conj(xy) = conj(0) = 0                         [conj_zero]
      5. So yx = 0.                                     [from 3,4] *)

  Theorem ker_lx_subset_ker_rx : forall x y,
    conj x = neg x ->
    x <> zero ->
    mul x y = zero ->
    mul y x = zero.
  Proof.
    intros x y Hpure Hxnz Hxy.
    (* Step 2: y is purely imaginary. *)
    assert (Hypure : conj y = neg y).
    { exact (zd_trace_zero x y Hpure Hxy Hxnz). }
    (* Step 3-4: conj(xy) = conj(y)*conj(x) = (-y)*(-x). *)
    assert (Hconj : mul (conj y) (conj x) = zero).
    { rewrite <- conj_antimorphism. rewrite Hxy. exact conj_zero. }
    (* Substitute conj(y) = -y, conj(x) = -x. *)
    rewrite Hypure in Hconj.
    rewrite Hpure in Hconj.
    (* Hconj: mul (neg y) (neg x) = zero *)
    (* neg(y)*neg(x) = yx, so yx = 0. *)
    rewrite neg_mul_left in Hconj.
    rewrite neg_mul_right in Hconj.
    rewrite neg_neg in Hconj.
    exact Hconj.
  Qed.

  (** Symmetric direction: yx = 0 => xy = 0.
      Same argument with roles swapped (y is given purely imaginary). *)
  Theorem ker_rx_subset_ker_lx : forall x y,
    conj x = neg x ->
    x <> zero ->
    mul y x = zero ->
    mul x y = zero.
  Proof.
    intros x y Hpure Hxnz Hyx.
    (* Step 1: conj(yx) = conj(x)*conj(y) = 0. *)
    assert (Hconj : mul (conj x) (conj y) = zero).
    { rewrite <- conj_antimorphism. rewrite Hyx. exact conj_zero. }
    (* Step 2: conj(x) = neg(x), so neg(x)*conj(y) = 0. *)
    rewrite Hpure in Hconj.
    rewrite neg_mul_left in Hconj.
    (* Hconj: neg (mul x (conj y)) = zero *)
    assert (Hxcy : mul x (conj y) = zero).
    { rewrite <- (neg_neg (mul x (conj y))). rewrite Hconj. exact neg_zero. }
    (* Step 3: x purely imaginary, x nonzero, x*conj(y) = 0.
       By zd_trace_zero: conj(conj y) = neg(conj y). *)
    assert (Hccy : conj (conj y) = neg (conj y)).
    { exact (zd_trace_zero x (conj y) Hpure Hxcy Hxnz). }
    (* Step 4: conj(conj y) = y, so y = neg(conj y), hence conj y = neg y. *)
    rewrite conj_invol in Hccy.
    assert (Hypure : conj y = neg y).
    { assert (Htmp := f_equal neg Hccy).
      rewrite neg_neg in Htmp. symmetry. exact Htmp. }
    (* Step 5: conj(yx) = conj(x)*conj(y) = (-x)*(-y) = xy. *)
    assert (Hconj2 : mul (conj x) (conj y) = zero).
    { rewrite <- conj_antimorphism. rewrite Hyx. exact conj_zero. }
    rewrite Hypure in Hconj2. rewrite Hpure in Hconj2.
    (* Hconj2: mul (neg x) (neg y) = zero *)
    rewrite (neg_neg_mul x y) in Hconj2.
    (* Hconj2: mul x y = zero *)
    exact Hconj2.
  Qed.

  (* ================================================================== *)
  (** ** Moreno Corollary 1.6: ZD symmetry. *)
  (* ================================================================== *)

  (** Moreno Corollary 1.6: ZD symmetry for nonzero purely imaginary x. *)
  Theorem zd_symmetry : forall x y,
    conj x = neg x ->
    x <> zero ->
    (mul x y = zero <-> mul y x = zero).
  Proof.
    intros x y Hpure Hxnz. split.
    - exact (ker_lx_subset_ker_rx x y Hpure Hxnz).
    - exact (ker_rx_subset_ker_lx x y Hpure Hxnz).
  Qed.

  (** Trivial case: x = 0 implies xy = 0 = yx for all y. *)
  Lemma zd_symmetry_zero_x : forall y,
    mul zero y = zero /\ mul y zero = zero.
  Proof.
    intros y. split.
    - exact (mul_zero_left y).
    - exact (mul_zero_right y).
  Qed.

  (* ================================================================== *)
  (** ** Corollary: x_bar * y = 0 when xy = 0 (purely imaginary x). *)
  (* ================================================================== *)

  Lemma zd_conj_vanishes : forall x y,
    conj x = neg x ->
    mul x y = zero ->
    mul (conj x) y = zero.
  Proof.
    intros x y Hpure Hxy.
    rewrite Hpure.
    rewrite neg_mul_left.
    rewrite Hxy.
    exact neg_zero.
  Qed.

End MorZDSymmetry.

(** ================================================================== *)
(** * PART II: Concrete sedenion verification.                         *)
(** ================================================================== *)

(** Direct computational proof that sedenion ZD symmetry holds for the
    Moreno-Froloff zero divisor pair:
      sed_zd_a = e3 + e10
      sed_zd_b = e6 - e15
    Both sed_zd_a * sed_zd_b = 0 and sed_zd_b * sed_zd_a = 0. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm ZD_Criterion.

(** Forward direction: a * b = 0.
    Already verified in C908_SedenionZeroDivisor; reproved here for
    self-containment. *)
Theorem sed_zd_product_ab : sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof.
  exact sed_zd_product_zero.
Qed.

(** Reverse direction: b * a = 0.
    This is the ZD symmetry (Corollary 1.6) verified computationally. *)
Theorem sed_zd_product_ba : sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  exact sed_zd_product_zero_rev.
Qed.

(** Both products vanish: the complete ZD symmetry witness. *)
Theorem C1538_sedenion_zd_symmetry :
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  exact (conj sed_zd_product_ab sed_zd_product_ba).
Qed.

(** Nonzero check: both elements are nonzero (neither is sed_zero).
    Extract a distinguishing component from each. *)
Lemma sed_zd_a_nonzero : sed_zd_a <> sed_zero.
Proof.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  assert (Hd := f_equal (fun x => qd (oct_lo x)) Hlo).
  cbv [sed_zd_a sed_zero sed_lo oct_lo qd oct_zero quat_zero] in Hd.
  lra.
Qed.

Lemma sed_zd_b_nonzero : sed_zd_b <> sed_zero.
Proof.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  assert (Hc := f_equal (fun x => qc (oct_hi x)) Hlo).
  cbv [sed_zd_b sed_zero sed_lo oct_hi qc oct_zero quat_zero] in Hc.
  lra.
Qed.

(** Full Corollary 1.6 at dim=16: a and b are nonzero, a*b = 0, b*a = 0. *)
Theorem C1538_full :
  sed_zd_a <> sed_zero /\
  sed_zd_b <> sed_zero /\
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  repeat split.
  - exact sed_zd_a_nonzero.
  - exact sed_zd_b_nonzero.
  - exact sed_zd_product_ab.
  - exact sed_zd_product_ba.
Qed.

Theorem C1538_brown_fundamental_major_theorem_fused :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  exact zd_fundamental_major_theorem_fused.
Qed.
