(** * Brown1972: Paper-scoped Rocq index for Brown (1972).

    Source:
      R.B. Brown, "Structure of zero divisors in higher dimensional real
      Cayley-Dickson algebras" (Brown 1972 lane in the repo's paper corpus).

    This file is the Rocq-facing paper lane for Brown (1972). It exposes the
    current Brown-specific formalization surface under one import, while also
    recording which dissertation chapters already have direct Rocq landings and
    which still live only in the Rust paper crate.

    Source-driven inventory note:
    - the on-disk Brown source packet shows a nine-chapter dissertation plus
      three appendices
    - Chapters I-II are contextual source chapters (introduction and review of
      literature) with no numbered theorem payload in the dissertation text
    - Chapter III is foundational but not the whole Brown picture
    - the theorem-dense Brown backlog is really spread across Chapters IV-VII,
      with Appendix C still relevant for the historical computation lane

    Chapter/page surfacing status:
    - Chapter III, pp. 11-16, Theorem 3.1, Theorem 3.3, Lemma 3.7,
      Theorem 3.9, and Lemma 3.10:
      a standard-octonion Brown 3.1 trace surface, a standard-octonion
      Brown 3.3 / 3.7 involution-quadratic surface, a weaker shared
      octonion/sedenion quadratic/conjugation core, and the abstract Rocq
      norm/involution surface with standard-octonion and direct
      standard-sedenion Brown 3.9 / 3.10 witnesses are landed here; Rust lane
      `crates/brown_1972/src/norm_symmetry.rs` remains the computational
      mirror for the broader quadratic/conjugation exploration beyond these
      concrete 8D/16D witnesses.
    - Chapter IV, pp. 20-22, Theorems 4.2-4.3 and Corollary 4.4:
      source-driven standard-tower witnesses for 4.2, 4.3, and 4.4 are now
      landed here.
    - Chapter V, pp. 27-30, Theorems 5.11-5.17:
      a generic one-generated/trace-zero exponent surface is now landed here,
      instantiated concretely for quaternions and octonions; Rust lane
      `crates/brown_1972/src/exponent_properties.rs` remains the broader
      computational mirror for farther non-quaternion/generalized follow-on
      exploration.
    - Chapter VI, pp. 30-42, Theorems 6.2-6.17:
      a standard-octonion Brown 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15
      basis-associator surface is now landed, a source-faithful standard-
      octonion anticommutator lane for Brown 6.16 / 6.17 is now landed, a
      direct standard-sedenion adjoined-element / polynomial lane for
      6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 is now landed, that
      Brown 6.4-6.7 lane is also packaged as a broader adjoined/polynomial
      lift interface above literal `mkSed` coordinates, and proof-faithful
      constructive 6.9 witnesses are now landed; the printed Brown 6.9
      pointwise iff wording does not survive unchanged in the repo's literal
      standard-pair coordinates, so the current Rocq surface records the
      constructive implications and the family form Brown's p.35 proof
      actually uses; the next Chapter VI work is any farther non-standard-
      model lift beside the broader Chapter III quadratic/conjugation lift.
    - Chapter VII, pp. 45-56, Theorems 7.3-7.18:
      direct Rocq landing via `ZD_Criterion.v`, `C1538_MorZDSymmetry.v`, and
      `BrownAssessorEquivalence.v`.
    - Appendix C, pp. 78-89:
      Rust lane `crates/brown_1972/src/pl1_emulator.rs`; Rocq extraction bridge
      is still open.

    Current Brown Rocq companion map:
    - ZD_Criterion.v             : Brown Theorem 7.15 concrete criterion lane
    - C1538_MorZDSymmetry.v      : Brown 7.3-style symmetry witness at dim 16
    - BrownAssessorEquivalence.v : Brown to de Marrais assessor / box-kite bridge

    Brown-adjacent support reused by these lanes, but not themselves Brown 1972
    paper surfaces, includes `CDPowerAssociative.v` and later Moreno bridges.

    Remaining Brown-specific Rocq backlog:
    - broader Chapter V exponent surface beyond the current quaternion/octonion
      one-generated/trace-zero surface
    - broader Chapter III quadratic/conjugation lane beyond the landed Brown
      3.1 / 3.3 / 3.7 standard-octonion source surface, the weaker shared
      octonion/sedenion quadratic/conjugation core, and the concrete
      octonion/sedenion 3.9 / 3.10 witnesses
    - broader Brown-numbered Chapter VI basis-element theorem lanes beyond the
      landed standard-octonion 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15
      basis-associator surface, the landed standard-octonion 6.16 / 6.17
      anticommutator surface, and the new broader 6.4-6.7 adjoined/
      polynomial lift interface
    - remaining Chapter VII numbering gaps plus Appendix C extraction bridge in Rocq

    The executable Rust companion for this paper is `crates/brown_1972/`. *)

From Stdlib Require Import List Reals ZArith Lia Lra.
Import ListNotations.
Open Scope R_scope.

From OpenGororoba Require Import
  ZDGraph
  Sedenion
  CayleyDicksonAlgebra
  OctonionNorm
  CDAssociator
  CDPowerAssociative
  CDSignBridge
  CDConjAntimorph
  CDLinearLemmas
  CDNegLemmas
  CDInverse
  SedenionAssociator
  DicksonCDProcess
  SedenionAlternativityFails.
From OpenGororoba Require Export
  C1538_MorZDSymmetry
  ZD_Criterion
  BrownAssessorEquivalence.
From OpenGororoba Require Import
  Brown1972ChapterIII
  Brown1972ChapterIV.

Theorem brown1972_lemma_6_10_i_octonion : forall i j : nat,
  (i < 8)%nat -> (j < 8)%nat ->
  i = 0 \/ j = 0 \/ i = j ->
  oct_mul (oct_e i) (oct_e j) = oct_mul (oct_e j) (oct_e i).
Proof.
  intros i j Hi Hj Hcase.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_lemma_6_10_ii_octonion : forall i j : nat,
  (i < 8)%nat -> (j < 8)%nat ->
  i <> 0%nat -> j <> 0%nat -> i <> j ->
  oct_mul (oct_e i) (oct_e j) = oct_neg (oct_mul (oct_e j) (oct_e i)).
Proof.
  intros i j Hi Hj Hi0 Hj0 Hij.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_6_11_octonion : forall i j : nat,
  (1 <= i)%nat -> (i < 8)%nat -> (j < 8)%nat ->
  oct_mul (oct_e i) (oct_mul (oct_e i) (oct_e j)) = oct_neg (oct_e j) /\
  oct_mul (oct_mul (oct_e j) (oct_e i)) (oct_e i) = oct_neg (oct_e j).
Proof.
  intros i j Hi Hi8 Hj8.
  split.
  - pose proof (brown1972_corollary_4_4_octonion_left (oct_e i) (oct_e j))
      as Halt.
    apply brown1972_oct_sub_eq_zero in Halt.
    rewrite <- Halt.
    rewrite (oct_mul_self_neg i Hi Hi8).
    apply brown1972_oct_mul_neg_e0_left.
  - pose proof (brown1972_corollary_4_4_octonion_right (oct_e i) (oct_e j))
      as Halt.
    apply brown1972_oct_sub_eq_zero in Halt.
    rewrite Halt.
    rewrite (oct_mul_self_neg i Hi Hi8).
    apply brown1972_oct_mul_neg_e0_right.
Qed.

Theorem brown1972_corollary_6_12_octonion : forall i : nat, forall x : CDOct,
  (i < 8)%nat ->
  oct_assoc (oct_e i) (oct_e i) x = oct_zero.
Proof.
  intros i x Hi.
  apply brown1972_corollary_4_4_octonion_left.
Qed.

Lemma brown1972_oct_lxor_lt8 : forall i j : nat,
  (i < 8)%nat -> (j < 8)%nat -> (Nat.lxor i j < 8)%nat.
Proof.
  intros i j Hi Hj.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  vm_compute; lia.
Qed.

Lemma brown1972_sign_to_R_square_one : forall s : Z,
  (sign_to_R s * sign_to_R s = 1)%R.
Proof.
  intro s.
  destruct (sign_to_R_pm1 s) as [Hs|Hs]; rewrite Hs; ring.
Qed.

Lemma brown1972_ch6_614_coeff_reduce : forall a b c d : R,
  (c * c = 1)%R ->
  (d * d = 1)%R ->
  (a * b = (a * b * c * d * d * c))%R.
Proof.
  intros a b c d Hc Hd.
  replace (a * b * c * d * d * c)%R with (a * b * (c * c) * (d * d))%R by ring.
  rewrite Hc, Hd.
  ring.
Qed.

Definition brown1972_ch6_613_oct_rhs (i j : nat) : CDOct :=
  if Nat.eqb i 0 then oct_e j
  else if Nat.eqb j 0 then oct_neg (oct_e 0)
  else if Nat.eqb i j then oct_neg (oct_e j)
  else oct_e j.

Theorem brown1972_theorem_6_13_octonion : forall i j : nat,
  (i < 8)%nat -> (j < 8)%nat ->
  oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e i) =
  brown1972_ch6_613_oct_rhs i j.
Proof.
  intros i j Hi Hj.
  unfold brown1972_ch6_613_oct_rhs.
  destruct (Nat.eqb i 0) eqn:Hi0.
  - apply Nat.eqb_eq in Hi0. subst i.
    change (oct_e 0) with brown1972_oct_one.
    rewrite brown1972_oct_mul_one_left.
    rewrite oct_mul_one_right.
    reflexivity.
  - apply Nat.eqb_neq in Hi0.
    destruct (Nat.eqb j 0) eqn:Hj0.
    + apply Nat.eqb_eq in Hj0. subst j.
      assert (Hi1 : (1 <= i)%nat) by lia.
      change (oct_e 0) with brown1972_oct_one.
      rewrite oct_mul_one_right.
      exact (oct_mul_self_neg i Hi1 Hi).
    + apply Nat.eqb_neq in Hj0.
      destruct (Nat.eqb i j) eqn:Hij.
      * apply Nat.eqb_eq in Hij. subst i.
        assert (Hj1 : (1 <= j)%nat) by lia.
        rewrite (oct_mul_self_neg j Hj1 Hj).
        apply brown1972_oct_mul_neg_e0_left.
      * apply Nat.eqb_neq in Hij.
        rewrite (brown1972_lemma_6_10_ii_octonion i j Hi Hj Hi0 Hj0 Hij).
        rewrite oct_neg_mul_left.
        assert (Hi1 : (1 <= i)%nat) by lia.
        destruct (brown1972_theorem_6_11_octonion i j Hi1 Hi Hj) as [_ Hright].
        rewrite Hright.
        apply oct_neg_neg.
Qed.

Definition brown1972_ch6_614_oct_epsilon (i j k : nat) : R :=
  (sign_to_R (oct_sign i j) *
   sign_to_R (oct_sign (Nat.lxor i j) k) *
   sign_to_R (oct_sign i (Nat.lxor j k)) *
   sign_to_R (oct_sign j k))%R.

Definition brown1972_ch6_615_positive (i j k : nat) : bool :=
  Nat.eqb i 0 || Nat.eqb j 0 || Nat.eqb k 0 ||
  Nat.eqb i j || Nat.eqb j k || Nat.eqb i k ||
  Nat.eqb (Nat.lxor i j) k.

Definition brown1972_ch6_615_oct_rhs (i j k : nat) : CDOct :=
  if brown1972_ch6_615_positive i j k
  then oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k))
  else oct_neg (oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k))).

Theorem brown1972_theorem_6_14_octonion : forall i j k : nat,
  (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
  oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e k) =
  oct_scale (brown1972_ch6_614_oct_epsilon i j k)
            (oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k))).
Proof.
  intros i j k Hi Hj Hk.
  unfold brown1972_ch6_614_oct_epsilon.
  assert (Hij : (Nat.lxor i j < 8)%nat).
  { apply brown1972_oct_lxor_lt8; assumption. }
  assert (Hjk : (Nat.lxor j k < 8)%nat).
  { apply brown1972_oct_lxor_lt8; assumption. }
  rewrite oct_basis_mul_xor with (i := i) (j := j) by assumption.
  rewrite oct_mul_scale_left.
  rewrite oct_basis_mul_xor with (i := Nat.lxor i j) (j := k) by assumption.
  rewrite brown1972_oct_scale_scale.
  rewrite oct_basis_mul_xor with (i := j) (j := k) by assumption.
  rewrite oct_mul_scale_right.
  rewrite oct_basis_mul_xor with (i := i) (j := Nat.lxor j k) by assumption.
  rewrite brown1972_oct_scale_scale.
  rewrite Nat.lxor_assoc.
  rewrite brown1972_oct_scale_scale.
  apply (f_equal2 oct_scale).
  - set (a := sign_to_R (oct_sign i j)).
    set (b := sign_to_R (oct_sign (Nat.lxor i j) k)).
    set (c := sign_to_R (oct_sign i (Nat.lxor j k))).
    set (d := sign_to_R (oct_sign j k)).
    assert (Hsq1 : (c * c = 1)%R).
    { unfold c. apply brown1972_sign_to_R_square_one. }
    assert (Hsq2 : (d * d = 1)%R).
    { unfold d. apply brown1972_sign_to_R_square_one. }
    apply brown1972_ch6_614_coeff_reduce; assumption.
  - reflexivity.
Qed.

Lemma brown1972_ch6_615_epsilon_classify : forall i j k : nat,
  (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
  brown1972_ch6_614_oct_epsilon i j k =
  if brown1972_ch6_615_positive i j k then 1%R else (-1)%R.
Proof.
  intros i j k Hi Hj Hk.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct k as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  vm_compute; ring.
Qed.

Theorem brown1972_lemma_6_15_octonion : forall i j k : nat,
  (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
  oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e k) =
  brown1972_ch6_615_oct_rhs i j k.
Proof.
  intros i j k Hi Hj Hk.
  unfold brown1972_ch6_615_oct_rhs.
  rewrite brown1972_theorem_6_14_octonion by assumption.
  rewrite brown1972_ch6_615_epsilon_classify by assumption.
  destruct (brown1972_ch6_615_positive i j k) eqn:Hcase.
  - apply brown1972_oct_scale_one.
  - destruct (oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k)))
      as [[x1 x2 x3 x4] [x5 x6 x7 x8]].
    cbv [oct_neg oct_scale quat_neg quat_scale qa qb qc qd].
    apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); ring.
    + apply (f_equal4 mkQuat); ring.
Qed.

Record Brown1972ChapterVIOctonionBasisSurface := {
  brown1972_ch6_l610_i_oct :
    forall i j : nat,
      (i < 8)%nat -> (j < 8)%nat ->
      i = 0 \/ j = 0 \/ i = j ->
      oct_mul (oct_e i) (oct_e j) = oct_mul (oct_e j) (oct_e i);
  brown1972_ch6_l610_ii_oct :
    forall i j : nat,
      (i < 8)%nat -> (j < 8)%nat ->
      i <> 0%nat -> j <> 0%nat -> i <> j ->
      oct_mul (oct_e i) (oct_e j) = oct_neg (oct_mul (oct_e j) (oct_e i));
  brown1972_ch6_t611_oct :
    forall i j : nat,
      (1 <= i)%nat -> (i < 8)%nat -> (j < 8)%nat ->
      oct_mul (oct_e i) (oct_mul (oct_e i) (oct_e j)) = oct_neg (oct_e j) /\
      oct_mul (oct_mul (oct_e j) (oct_e i)) (oct_e i) = oct_neg (oct_e j);
  brown1972_ch6_c612_oct :
    forall i : nat, forall x : CDOct,
      (i < 8)%nat ->
      oct_assoc (oct_e i) (oct_e i) x = oct_zero;
  brown1972_ch6_t613_oct :
    forall i j : nat,
      (i < 8)%nat -> (j < 8)%nat ->
      oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e i) =
      brown1972_ch6_613_oct_rhs i j;
  brown1972_ch6_t614_oct :
    forall i j k : nat,
      (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
      oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e k) =
      oct_scale (brown1972_ch6_614_oct_epsilon i j k)
                (oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k)));
  brown1972_ch6_l615_oct :
    forall i j k : nat,
      (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
      oct_mul (oct_mul (oct_e i) (oct_e j)) (oct_e k) =
      brown1972_ch6_615_oct_rhs i j k
}.

Definition brown1972_octonion_chapter_vi_basis_surface :
  Brown1972ChapterVIOctonionBasisSurface.
Proof.
  refine {| brown1972_ch6_l610_i_oct := brown1972_lemma_6_10_i_octonion;
            brown1972_ch6_l610_ii_oct := brown1972_lemma_6_10_ii_octonion;
            brown1972_ch6_t611_oct := brown1972_theorem_6_11_octonion;
            brown1972_ch6_c612_oct := brown1972_corollary_6_12_octonion;
            brown1972_ch6_t613_oct := brown1972_theorem_6_13_octonion;
            brown1972_ch6_t614_oct := brown1972_theorem_6_14_octonion;
            brown1972_ch6_l615_oct := brown1972_lemma_6_15_octonion |}.
Defined.

Definition brown1972_oct_imag_dot (A B : CDOct) : R :=
  match A, B with
  | mkOct (mkQuat _ a1 a2 a3) (mkQuat a4 a5 a6 a7),
    mkOct (mkQuat _ b1 b2 b3) (mkQuat b4 b5 b6 b7) =>
      (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7)%R
  end.

Definition brown1972_ch6_616_rhs (A B : CDOct) : CDOct :=
  match A, B with
  | mkOct (mkQuat a0 a1 a2 a3) (mkQuat a4 a5 a6 a7),
    mkOct (mkQuat b0 b1 b2 b3) (mkQuat b4 b5 b6 b7) =>
      mkOct
        (mkQuat
           (2 * (a0 * b0 - (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 +
                            a5 * b5 + a6 * b6 + a7 * b7)))
           (2 * (a0 * b1 + a1 * b0))
           (2 * (a0 * b2 + a2 * b0))
           (2 * (a0 * b3 + a3 * b0)))
        (mkQuat
           (2 * (a0 * b4 + a4 * b0))
           (2 * (a0 * b5 + a5 * b0))
           (2 * (a0 * b6 + a6 * b0))
           (2 * (a0 * b7 + a7 * b0)))
  end.

Theorem brown1972_lemma_6_16_octonion : forall A B : CDOct,
  oct_add (oct_mul A B) (oct_mul B A) = brown1972_ch6_616_rhs A B.
Proof.
  intros [[a0 a1 a2 a3] [a4 a5 a6 a7]]
         [[b0 b1 b2 b3] [b4 b5 b6 b7]].
  cbv [brown1972_ch6_616_rhs brown1972_oct_imag_dot].
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_6_17_octonion : forall A B : CDOct,
  A <> oct_zero ->
  B <> oct_zero ->
  oct_add (oct_mul A B) (oct_mul B A) = oct_zero <->
  brown1972_oct_trace A = 0%R /\
  brown1972_oct_trace B = 0%R /\
  brown1972_oct_imag_dot A B = 0%R.
Proof.
  intros A B HAnz HBnz.
  split.
  - destruct A as [[a0 a1 a2 a3] [a4 a5 a6 a7]];
    destruct B as [[b0 b1 b2 b3] [b4 b5 b6 b7]].
    intro Hzero.
    rewrite brown1972_lemma_6_16_octonion in Hzero.
    cbv [brown1972_ch6_616_rhs brown1972_oct_imag_dot brown1972_oct_trace
         oct_zero quat_zero oct_lo oct_hi qa qb qc qd] in Hzero |- *.
    inversion Hzero; clear Hzero; subst.
    repeat match goal with
    | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion Hq; clear Hq; subst
    end.
    assert (Hreal :
      (a0 * b0 - (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 +
                  a5 * b5 + a6 * b6 + a7 * b7) = 0)%R) by lra.
    assert (Hc1 : (a0 * b1 + a1 * b0 = 0)%R) by lra.
    assert (Hc2 : (a0 * b2 + a2 * b0 = 0)%R) by lra.
    assert (Hc3 : (a0 * b3 + a3 * b0 = 0)%R) by lra.
    assert (Hc4 : (a0 * b4 + a4 * b0 = 0)%R) by lra.
    assert (Hc5 : (a0 * b5 + a5 * b0 = 0)%R) by lra.
    assert (Hc6 : (a0 * b6 + a6 * b0 = 0)%R) by lra.
    assert (Hc7 : (a0 * b7 + a7 * b0 = 0)%R) by lra.
    assert (Hb0 : b0 = 0%R).
    {
      destruct (Req_EM_T a0 0) as [Ha0|Ha0].
      - subst a0.
        destruct (Req_EM_T b0 0) as [Hb0|Hb0]; auto.
        assert (Ha1 : a1 = 0%R) by nra.
        assert (Ha2 : a2 = 0%R) by nra.
        assert (Ha3 : a3 = 0%R) by nra.
        assert (Ha4 : a4 = 0%R) by nra.
        assert (Ha5 : a5 = 0%R) by nra.
        assert (Ha6 : a6 = 0%R) by nra.
        assert (Ha7 : a7 = 0%R) by nra.
        subst.
        exfalso. apply HAnz. reflexivity.
      - destruct (Req_EM_T b0 0) as [Hb0|Hb0]; auto.
        assert (Hsum :
          (a0 * (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 +
                 a5 * b5 + a6 * b6 + a7 * b7) +
           b0 * (a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 +
                 a5 * a5 + a6 * a6 + a7 * a7) = 0)%R).
        {
          replace
            (a0 * (a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 +
                   a5 * b5 + a6 * b6 + a7 * b7) +
             b0 * (a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 +
                   a5 * a5 + a6 * a6 + a7 * a7))%R
            with
            (a1 * (a0 * b1 + a1 * b0) +
             a2 * (a0 * b2 + a2 * b0) +
             a3 * (a0 * b3 + a3 * b0) +
             a4 * (a0 * b4 + a4 * b0) +
             a5 * (a0 * b5 + a5 * b0) +
             a6 * (a0 * b6 + a6 * b0) +
             a7 * (a0 * b7 + a7 * b0))%R by ring.
          rewrite Hc1, Hc2, Hc3, Hc4, Hc5, Hc6, Hc7.
          ring.
        }
        assert (Hquad :
          (b0 * (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 +
                 a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7) = 0)%R).
        { nra. }
        assert (Hsum0 :
          (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 +
           a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7 = 0)%R) by nra.
        assert (HnormA : oct_norm_sq (mkOct (mkQuat a0 a1 a2 a3) (mkQuat a4 a5 a6 a7)) = 0%R).
        {
          unfold oct_norm_sq, quat_norm_sq.
          simpl.
          nra.
        }
        exfalso.
        apply HAnz.
        apply brown1972_oct_norm_zero.
        exact HnormA.
    }
    assert (Ha0 : a0 = 0%R).
    {
      destruct (Req_EM_T a0 0) as [Ha0|Ha0]; auto.
      assert (Hb1 : b1 = 0%R) by nra.
      assert (Hb2 : b2 = 0%R) by nra.
      assert (Hb3 : b3 = 0%R) by nra.
      assert (Hb4 : b4 = 0%R) by nra.
      assert (Hb5 : b5 = 0%R) by nra.
      assert (Hb6 : b6 = 0%R) by nra.
      assert (Hb7 : b7 = 0%R) by nra.
      subst.
      exfalso. apply HBnz. reflexivity.
    }
    split.
    + lra.
    + split.
      * lra.
      * unfold brown1972_oct_imag_dot. simpl. nra.
  - destruct A as [[a0 a1 a2 a3] [a4 a5 a6 a7]];
    destruct B as [[b0 b1 b2 b3] [b4 b5 b6 b7]].
    intros [HA [HB Hdot]].
    rewrite brown1972_lemma_6_16_octonion.
    cbv [brown1972_ch6_616_rhs brown1972_oct_imag_dot brown1972_oct_trace
         oct_zero quat_zero oct_lo oct_hi qa qb qc qd] in HA, HB, Hdot |- *.
    apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); nra.
    + apply (f_equal4 mkQuat); nra.
Qed.

Record Brown1972ChapterVIOctonionAnticommutatorSurface := {
  brown1972_ch6_l616_oct :
    forall A B : CDOct,
      oct_add (oct_mul A B) (oct_mul B A) = brown1972_ch6_616_rhs A B;
  brown1972_ch6_t617_oct :
    forall A B : CDOct,
      A <> oct_zero ->
      B <> oct_zero ->
      oct_add (oct_mul A B) (oct_mul B A) = oct_zero <->
      brown1972_oct_trace A = 0%R /\
      brown1972_oct_trace B = 0%R /\
      brown1972_oct_imag_dot A B = 0%R
}.

Definition brown1972_octonion_chapter_vi_anticommutator_surface :
  Brown1972ChapterVIOctonionAnticommutatorSurface.
Proof.
  refine {| brown1972_ch6_l616_oct := brown1972_lemma_6_16_octonion;
            brown1972_ch6_t617_oct := brown1972_theorem_6_17_octonion |}.
Defined.

Definition brown1972_sed_adjoined_e : CDSed := sed_e 8.

Ltac brown1972_close_sed_ring :=
  cbv [brown1972_sed_adjoined_e sed_assoc sed_add sed_sub sed_neg
       sed_mul sed_conj sed_scale sed_zero sed_e
       oct_assoc oct_add oct_sub oct_neg oct_mul oct_conj oct_scale oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_scale quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct);
   [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct);
   [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].

Lemma brown1972_sed_mul_adjoined_right : forall a : CDSed,
  sed_mul a brown1972_sed_adjoined_e = mkSed (oct_neg (sed_hi a)) (sed_lo a).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [brown1972_sed_adjoined_e sed_mul sed_e sed_lo sed_hi
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_sed_mul_adjoined_left : forall a : CDSed,
  sed_mul brown1972_sed_adjoined_e a =
  mkSed (oct_neg (oct_conj (sed_hi a))) (oct_conj (sed_lo a)).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [brown1972_sed_adjoined_e sed_mul sed_e sed_lo sed_hi
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_theorem_6_1_sedenion : forall a b : CDSed,
  sed_add (sed_assoc brown1972_sed_adjoined_e a b)
          (sed_assoc a brown1972_sed_adjoined_e b) =
  sed_zero.
Proof.
  intros a b.
  unfold sed_assoc, sed_sub, sed_add.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_left.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Theorem brown1972_lemma_6_2_i_sedenion : forall a b : CDSed,
  sed_add (sed_assoc a b brown1972_sed_adjoined_e)
          (sed_assoc a brown1972_sed_adjoined_e b) =
  sed_zero.
Proof.
  intros a b.
  unfold sed_assoc, sed_sub, sed_add.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_left.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Theorem brown1972_lemma_6_2_iii_sedenion : forall a b : CDSed,
  sed_add (sed_assoc brown1972_sed_adjoined_e a b)
          (sed_assoc brown1972_sed_adjoined_e b a) =
  sed_zero.
Proof.
  intros a b.
  unfold sed_assoc, sed_sub, sed_add.
  repeat rewrite brown1972_sed_mul_adjoined_left.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Theorem brown1972_lemma_6_2_iv_sedenion : forall a b : CDSed,
  sed_add (sed_assoc a b brown1972_sed_adjoined_e)
          (sed_assoc b a brown1972_sed_adjoined_e) =
  sed_zero.
Proof.
  intros a b.
  unfold sed_assoc, sed_sub, sed_add.
  repeat rewrite brown1972_sed_mul_adjoined_right.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Lemma brown1972_sed_add_self_eq_zero : forall x : CDSed,
  sed_add x x = sed_zero -> x = sed_zero.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]] H.
  cbv [sed_add sed_zero oct_add oct_zero quat_add quat_zero
       qa qb qc qd oct_lo oct_hi sed_lo sed_hi] in H.
  inversion H; clear H; subst.
  repeat match goal with
  | Hq : mkOct _ _ = mkOct _ _ |- _ => inversion Hq; clear Hq; subst
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion Hq; clear Hq; subst
  end.
  assert (Ha1 : a1 = 0%R) by lra.
  assert (Ha2 : a2 = 0%R) by lra.
  assert (Ha3 : a3 = 0%R) by lra.
  assert (Ha4 : a4 = 0%R) by lra.
  assert (Ha5 : a5 = 0%R) by lra.
  assert (Ha6 : a6 = 0%R) by lra.
  assert (Ha7 : a7 = 0%R) by lra.
  assert (Ha8 : a8 = 0%R) by lra.
  assert (Ha9 : a9 = 0%R) by lra.
  assert (Ha10 : a10 = 0%R) by lra.
  assert (Ha11 : a11 = 0%R) by lra.
  assert (Ha12 : a12 = 0%R) by lra.
  assert (Ha13 : a13 = 0%R) by lra.
  assert (Ha14 : a14 = 0%R) by lra.
  assert (Ha15 : a15 = 0%R) by lra.
  assert (Ha16 : a16 = 0%R) by lra.
  subst.
  reflexivity.
Qed.

Theorem brown1972_lemma_6_2_ii_sedenion : forall a : CDSed,
  sed_assoc brown1972_sed_adjoined_e a a = sed_zero /\
  sed_assoc a a brown1972_sed_adjoined_e = sed_zero /\
  sed_assoc brown1972_sed_adjoined_e brown1972_sed_adjoined_e a = sed_zero /\
  sed_assoc a brown1972_sed_adjoined_e brown1972_sed_adjoined_e = sed_zero.
Proof.
  intro a.
  split.
  - pose proof (brown1972_lemma_6_2_iii_sedenion a a) as Hdiag.
    apply brown1972_sed_add_self_eq_zero in Hdiag.
    exact Hdiag.
  - pose proof (brown1972_theorem_6_1_sedenion a a) as H61.
    pose proof (brown1972_lemma_6_2_i_sedenion a a) as H621.
    pose proof (brown1972_lemma_6_2_iii_sedenion a a) as Hdiag.
    apply brown1972_sed_add_self_eq_zero in Hdiag.
    rewrite Hdiag in H61.
    rewrite sed_add_zero_left in H61.
    split.
    + rewrite H61 in H621.
      rewrite sed_add_zero_right in H621.
      exact H621.
    + split.
      * pose proof (brown1972_theorem_6_1_sedenion brown1972_sed_adjoined_e a) as Hee.
        apply brown1972_sed_add_self_eq_zero in Hee.
        exact Hee.
      * pose proof (brown1972_lemma_6_2_i_sedenion a brown1972_sed_adjoined_e) as Haee.
        apply brown1972_sed_add_self_eq_zero in Haee.
        exact Haee.
Qed.

Theorem brown1972_theorem_6_3_i_sedenion : forall a b : CDSed,
  sed_mul (sed_mul (sed_mul brown1972_sed_adjoined_e a) brown1972_sed_adjoined_e) b =
  sed_mul brown1972_sed_adjoined_e
          (sed_mul a (sed_mul brown1972_sed_adjoined_e b)).
Proof.
  intros a b.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_left.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Theorem brown1972_theorem_6_3_ii_sedenion : forall a b : CDSed,
  sed_mul b (sed_mul (sed_mul brown1972_sed_adjoined_e a) brown1972_sed_adjoined_e) =
  sed_mul (sed_mul (sed_mul b brown1972_sed_adjoined_e) a) brown1972_sed_adjoined_e.
Proof.
  intros a b.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_right.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Theorem brown1972_theorem_6_3_iii_sedenion : forall a b : CDSed,
  sed_mul (sed_mul brown1972_sed_adjoined_e b)
          (sed_mul a brown1972_sed_adjoined_e) =
  sed_mul (sed_mul brown1972_sed_adjoined_e (sed_mul b a))
          brown1972_sed_adjoined_e.
Proof.
  intros a b.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_right.
  rewrite brown1972_sed_mul_adjoined_left.
  rewrite brown1972_sed_mul_adjoined_right.
  destruct a as [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
                 [[a9 a10 a11 a12] [a13 a14 a15 a16]]];
  destruct b as [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
                 [[b9 b10 b11 b12] [b13 b14 b15 b16]]];
  cbv [brown1972_sed_adjoined_e sed_mul sed_neg sed_lo sed_hi sed_zero sed_e
       oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
       quat_add quat_neg quat_mul quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  [apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]
  |apply (f_equal2 mkOct); [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring]].
Qed.

Record Brown1972ChapterVISedenionAdjoinedSurface := {
  brown1972_ch6_t61_sed :
    forall a b : CDSed,
      sed_add (sed_assoc brown1972_sed_adjoined_e a b)
              (sed_assoc a brown1972_sed_adjoined_e b) =
      sed_zero;
  brown1972_ch6_l62_i_sed :
    forall a b : CDSed,
      sed_add (sed_assoc a b brown1972_sed_adjoined_e)
              (sed_assoc a brown1972_sed_adjoined_e b) =
      sed_zero;
  brown1972_ch6_l62_ii_sed :
    forall a : CDSed,
      sed_assoc brown1972_sed_adjoined_e a a = sed_zero /\
      sed_assoc a a brown1972_sed_adjoined_e = sed_zero /\
      sed_assoc brown1972_sed_adjoined_e brown1972_sed_adjoined_e a = sed_zero /\
      sed_assoc a brown1972_sed_adjoined_e brown1972_sed_adjoined_e = sed_zero;
  brown1972_ch6_l62_iii_sed :
    forall a b : CDSed,
      sed_add (sed_assoc brown1972_sed_adjoined_e a b)
              (sed_assoc brown1972_sed_adjoined_e b a) =
      sed_zero;
  brown1972_ch6_l62_iv_sed :
    forall a b : CDSed,
      sed_add (sed_assoc a b brown1972_sed_adjoined_e)
              (sed_assoc b a brown1972_sed_adjoined_e) =
      sed_zero;
  brown1972_ch6_t63_i_sed :
    forall a b : CDSed,
      sed_mul (sed_mul (sed_mul brown1972_sed_adjoined_e a) brown1972_sed_adjoined_e) b =
      sed_mul brown1972_sed_adjoined_e
              (sed_mul a (sed_mul brown1972_sed_adjoined_e b));
  brown1972_ch6_t63_ii_sed :
    forall a b : CDSed,
      sed_mul b (sed_mul (sed_mul brown1972_sed_adjoined_e a) brown1972_sed_adjoined_e) =
      sed_mul (sed_mul (sed_mul b brown1972_sed_adjoined_e) a) brown1972_sed_adjoined_e;
  brown1972_ch6_t63_iii_sed :
    forall a b : CDSed,
      sed_mul (sed_mul brown1972_sed_adjoined_e b)
              (sed_mul a brown1972_sed_adjoined_e) =
      sed_mul (sed_mul brown1972_sed_adjoined_e (sed_mul b a))
              brown1972_sed_adjoined_e
}.

Definition brown1972_sedenion_chapter_vi_adjoined_surface :
  Brown1972ChapterVISedenionAdjoinedSurface.
Proof.
  refine {| brown1972_ch6_t61_sed := brown1972_theorem_6_1_sedenion;
            brown1972_ch6_l62_i_sed := brown1972_lemma_6_2_i_sedenion;
            brown1972_ch6_l62_ii_sed := brown1972_lemma_6_2_ii_sedenion;
            brown1972_ch6_l62_iii_sed := brown1972_lemma_6_2_iii_sedenion;
            brown1972_ch6_l62_iv_sed := brown1972_lemma_6_2_iv_sedenion;
            brown1972_ch6_t63_i_sed := brown1972_theorem_6_3_i_sedenion;
            brown1972_ch6_t63_ii_sed := brown1972_theorem_6_3_ii_sedenion;
            brown1972_ch6_t63_iii_sed := brown1972_theorem_6_3_iii_sedenion |}.
Defined.

Theorem brown1972_corollary_6_4_sedenion : forall A B : CDSed,
  sed_assoc B (sed_mul brown1972_sed_adjoined_e A) brown1972_sed_adjoined_e =
  sed_mul (sed_neg (sed_assoc B brown1972_sed_adjoined_e A))
          brown1972_sed_adjoined_e.
Proof.
  intros A B.
  unfold sed_assoc, sed_sub.
  rewrite brown1972_theorem_6_3_ii_sedenion.
  rewrite <- sed_neg_mul_left.
  rewrite <- sed_mul_add_left.
  rewrite sed_neg_add.
  rewrite sed_neg_neg.
  rewrite sed_add_comm.
  reflexivity.
Qed.

(** Brown 6.5/6.6 are stated in the dissertation with symbolic [a + e b]
    notation. In the repo's fixed standard-pair sedenion coordinates, the
    lower-octonion and adjoined-slot pieces are tracked explicitly by the
    following two embeddings. *)
Definition brown1972_sed_oct_embed (a : CDOct) : CDSed := mkSed a oct_zero.

Definition brown1972_sed_poly_embed (a : CDOct) : CDSed := mkSed oct_zero a.

Lemma brown1972_oct_add_assoc : forall x y z : CDOct,
  oct_add x (oct_add y z) = oct_add (oct_add x y) z.
Proof.
  intros [xlo xhi] [ylo yhi] [zlo zhi].
  unfold oct_add; simpl.
  apply (f_equal2 mkOct); unfold quat_add; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_sed_add_assoc : forall x y z : CDSed,
  sed_add x (sed_add y z) = sed_add (sed_add x y) z.
Proof.
  intros [xlo xhi] [ylo yhi] [zlo zhi].
  unfold sed_add; simpl.
  f_equal; apply brown1972_oct_add_assoc.
Qed.

Lemma brown1972_oct_add_rearrange : forall w x y z : CDOct,
  oct_add (oct_add w x) (oct_add y z) =
  oct_add (oct_add w z) (oct_add x y).
Proof.
  intros [wlo whi] [xlo xhi] [ylo yhi] [zlo zhi].
  unfold oct_add; simpl.
  apply (f_equal2 mkOct); unfold quat_add; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_sed_add_rearrange : forall w x y z : CDSed,
  sed_add (sed_add w x) (sed_add y z) =
  sed_add (sed_add w z) (sed_add x y).
Proof.
  intros [wlo whi] [xlo xhi] [ylo yhi] [zlo zhi].
  unfold sed_add; simpl.
  f_equal; apply brown1972_oct_add_rearrange.
Qed.

Lemma brown1972_sed_oct_poly_decompose : forall a b : CDOct,
  mkSed a b = sed_add (brown1972_sed_oct_embed a) (brown1972_sed_poly_embed b).
Proof.
  intros a b.
  unfold brown1972_sed_oct_embed, brown1972_sed_poly_embed, sed_add.
  simpl.
  f_equal.
  - symmetry. apply oct_add_zero_right.
  - symmetry. apply oct_add_zero_left.
Qed.

Lemma brown1972_sed_oct_embed_mul : forall a b : CDOct,
  sed_mul (brown1972_sed_oct_embed a) (brown1972_sed_oct_embed b) =
  brown1972_sed_oct_embed (oct_mul a b).
Proof.
  intros a b.
  destruct a as [alo ahi], b as [blo bhi].
  cbv [brown1972_sed_oct_embed].
  brown1972_close_sed_ring.
Qed.

Theorem brown1972_theorem_6_6_i_sedenion : forall a b : CDOct,
  sed_mul (brown1972_sed_oct_embed a) (brown1972_sed_poly_embed b) =
  brown1972_sed_poly_embed (oct_mul b a).
Proof.
  intros a b.
  destruct a as [alo ahi], b as [blo bhi].
  cbv [brown1972_sed_oct_embed brown1972_sed_poly_embed].
  brown1972_close_sed_ring.
Qed.

Theorem brown1972_theorem_6_6_ii_sedenion : forall a b : CDOct,
  sed_mul (brown1972_sed_poly_embed a) (brown1972_sed_oct_embed b) =
  brown1972_sed_poly_embed (oct_mul a (oct_conj b)).
Proof.
  intros a b.
  destruct a as [alo ahi], b as [blo bhi].
  cbv [brown1972_sed_oct_embed brown1972_sed_poly_embed].
  brown1972_close_sed_ring.
Qed.

Theorem brown1972_theorem_6_6_iii_sedenion : forall a b : CDOct,
  sed_mul (brown1972_sed_poly_embed a) (brown1972_sed_poly_embed b) =
  brown1972_sed_oct_embed (oct_neg (oct_mul (oct_conj b) a)).
Proof.
  intros a b.
  destruct a as [alo ahi], b as [blo bhi].
  cbv [brown1972_sed_oct_embed brown1972_sed_poly_embed].
  brown1972_close_sed_ring.
Qed.

Definition brown1972_ch6_67_polynomial_mul
  (a1 a2 b1 b2 : CDOct) : CDSed :=
  sed_add
    (sed_add
       (sed_mul (brown1972_sed_oct_embed a1) (brown1972_sed_oct_embed b1))
       (sed_mul (brown1972_sed_poly_embed a2) (brown1972_sed_poly_embed b2)))
    (sed_add
       (sed_mul (brown1972_sed_oct_embed a1) (brown1972_sed_poly_embed b2))
       (sed_mul (brown1972_sed_poly_embed a2) (brown1972_sed_oct_embed b1))).

Lemma brown1972_ch6_polynomial_mul_eq_cd :
  forall a1 a2 b1 b2 : CDOct,
    brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
    sed_mul (mkSed a1 a2) (mkSed b1 b2).
Proof.
  intros a1 a2 b1 b2.
  unfold brown1972_ch6_67_polynomial_mul.
  rewrite brown1972_sed_oct_poly_decompose.
  rewrite brown1972_sed_oct_poly_decompose.
  rewrite sed_mul_add_left.
  rewrite sed_mul_add_right.
  rewrite sed_mul_add_right.
  rewrite brown1972_sed_oct_embed_mul.
  rewrite brown1972_theorem_6_6_i_sedenion.
  rewrite brown1972_theorem_6_6_ii_sedenion.
  rewrite brown1972_theorem_6_6_iii_sedenion.
  rewrite <- brown1972_sed_add_assoc.
  rewrite brown1972_sed_add_rearrange.
  rewrite <- brown1972_sed_add_assoc.
  reflexivity.
Qed.

Theorem brown1972_theorem_6_5_standard_octonion_sedenion :
  forall a1 a2 b1 b2 : CDOct,
    brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
    sed_mul (mkSed a1 a2) (mkSed b1 b2).
Proof.
  exact brown1972_ch6_polynomial_mul_eq_cd.
Qed.

Theorem brown1972_theorem_6_5_standard_octonion_sedenion_lift :
  forall a1 a2 b1 b2 : CDOct,
    brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
    sed_mul
      (sed_add (brown1972_sed_oct_embed a1) (brown1972_sed_poly_embed a2))
      (sed_add (brown1972_sed_oct_embed b1) (brown1972_sed_poly_embed b2)).
Proof.
  intros a1 a2 b1 b2.
  rewrite <- brown1972_sed_oct_poly_decompose.
  rewrite <- brown1972_sed_oct_poly_decompose.
  exact (brown1972_theorem_6_5_standard_octonion_sedenion a1 a2 b1 b2).
Qed.

Corollary brown1972_corollary_6_7_sedenion :
  forall a1 a2 b1 b2 : CDOct,
    brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
    sed_mul (mkSed a1 a2) (mkSed b1 b2).
Proof.
  exact brown1972_ch6_polynomial_mul_eq_cd.
Qed.

Corollary brown1972_corollary_6_7_sedenion_lift :
  forall a1 a2 b1 b2 : CDOct,
    brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
    sed_mul
      (sed_add (brown1972_sed_oct_embed a1) (brown1972_sed_poly_embed a2))
      (sed_add (brown1972_sed_oct_embed b1) (brown1972_sed_poly_embed b2)).
Proof.
  intros a1 a2 b1 b2.
  exact (brown1972_theorem_6_5_standard_octonion_sedenion_lift a1 a2 b1 b2).
Qed.

Record Brown1972ChapterVIAdjoinedPolynomialLiftSurface
  (Base Ext : Type)
  (base_mul : Base -> Base -> Base)
  (base_neg : Base -> Base)
  (base_conj : Base -> Base)
  (ext_add : Ext -> Ext -> Ext)
  (ext_mul : Ext -> Ext -> Ext)
  (ext_assoc : Ext -> Ext -> Ext -> Ext)
  (ext_neg : Ext -> Ext)
  (base_embed poly_embed : Base -> Ext)
  (adjoined_e : Ext)
  (poly_mul : Base -> Base -> Base -> Base -> Ext) := {
  brown1972_ch6_c64_lift :
    forall A B : Ext,
      ext_assoc B (ext_mul adjoined_e A) adjoined_e =
      ext_mul (ext_neg (ext_assoc B adjoined_e A))
              adjoined_e;
  brown1972_ch6_t65_lift :
    forall a1 a2 b1 b2 : Base,
      poly_mul a1 a2 b1 b2 =
      ext_mul (ext_add (base_embed a1) (poly_embed a2))
              (ext_add (base_embed b1) (poly_embed b2));
  brown1972_ch6_t66_i_lift :
    forall a b : Base,
      ext_mul (base_embed a) (poly_embed b) =
      poly_embed (base_mul b a);
  brown1972_ch6_t66_ii_lift :
    forall a b : Base,
      ext_mul (poly_embed a) (base_embed b) =
      poly_embed (base_mul a (base_conj b));
  brown1972_ch6_t66_iii_lift :
    forall a b : Base,
      ext_mul (poly_embed a) (poly_embed b) =
      base_embed (base_neg (base_mul (base_conj b) a));
  brown1972_ch6_c67_lift :
    forall a1 a2 b1 b2 : Base,
      poly_mul a1 a2 b1 b2 =
      ext_mul (ext_add (base_embed a1) (poly_embed a2))
              (ext_add (base_embed b1) (poly_embed b2))
}.

Record Brown1972ChapterVISedenionPolynomialSurface := {
  brown1972_ch6_c64_sed :
    forall A B : CDSed,
      sed_assoc B (sed_mul brown1972_sed_adjoined_e A) brown1972_sed_adjoined_e =
      sed_mul (sed_neg (sed_assoc B brown1972_sed_adjoined_e A))
              brown1972_sed_adjoined_e;
  brown1972_ch6_t65_sed :
    forall a1 a2 b1 b2 : CDOct,
      brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
      sed_mul (mkSed a1 a2) (mkSed b1 b2);
  brown1972_ch6_t66_i_sed :
    forall a b : CDOct,
      sed_mul (brown1972_sed_oct_embed a) (brown1972_sed_poly_embed b) =
      brown1972_sed_poly_embed (oct_mul b a);
  brown1972_ch6_t66_ii_sed :
    forall a b : CDOct,
      sed_mul (brown1972_sed_poly_embed a) (brown1972_sed_oct_embed b) =
      brown1972_sed_poly_embed (oct_mul a (oct_conj b));
  brown1972_ch6_t66_iii_sed :
    forall a b : CDOct,
      sed_mul (brown1972_sed_poly_embed a) (brown1972_sed_poly_embed b) =
      brown1972_sed_oct_embed (oct_neg (oct_mul (oct_conj b) a));
  brown1972_ch6_c67_sed :
    forall a1 a2 b1 b2 : CDOct,
      brown1972_ch6_67_polynomial_mul a1 a2 b1 b2 =
      sed_mul (mkSed a1 a2) (mkSed b1 b2)
}.

Definition brown1972_sedenion_chapter_vi_polynomial_surface :
  Brown1972ChapterVISedenionPolynomialSurface.
Proof.
  refine {| brown1972_ch6_c64_sed := brown1972_corollary_6_4_sedenion;
            brown1972_ch6_t65_sed := brown1972_theorem_6_5_standard_octonion_sedenion;
            brown1972_ch6_t66_i_sed := brown1972_theorem_6_6_i_sedenion;
            brown1972_ch6_t66_ii_sed := brown1972_theorem_6_6_ii_sedenion;
            brown1972_ch6_t66_iii_sed := brown1972_theorem_6_6_iii_sedenion;
            brown1972_ch6_c67_sed := brown1972_corollary_6_7_sedenion |}.
Defined.

Definition brown1972_sedenion_chapter_vi_adjoined_polynomial_lift_surface :
  Brown1972ChapterVIAdjoinedPolynomialLiftSurface
    CDOct CDSed
    oct_mul oct_neg oct_conj
    sed_add sed_mul sed_assoc sed_neg
    brown1972_sed_oct_embed brown1972_sed_poly_embed
    brown1972_sed_adjoined_e
    brown1972_ch6_67_polynomial_mul.
Proof.
  refine {| brown1972_ch6_c64_lift := brown1972_corollary_6_4_sedenion;
            brown1972_ch6_t65_lift := brown1972_theorem_6_5_standard_octonion_sedenion_lift;
            brown1972_ch6_t66_i_lift := brown1972_theorem_6_6_i_sedenion;
            brown1972_ch6_t66_ii_lift := brown1972_theorem_6_6_ii_sedenion;
            brown1972_ch6_t66_iii_lift := brown1972_theorem_6_6_iii_sedenion;
            brown1972_ch6_c67_lift := brown1972_corollary_6_7_sedenion_lift |}.
Defined.

Lemma brown1972_sedenion_adjoined_polynomial_decompose : forall x : CDSed,
  x =
  sed_add (brown1972_sed_oct_embed (sed_lo x))
          (brown1972_sed_poly_embed (sed_hi x)).
Proof.
  intros [a b].
  exact (brown1972_sed_oct_poly_decompose a b).
Qed.

Theorem brown1972_theorem_6_5_sedenion_decomposed : forall x y : CDSed,
  brown1972_ch6_67_polynomial_mul (sed_lo x) (sed_hi x) (sed_lo y) (sed_hi y) =
  sed_mul x y.
Proof.
  intros x y.
  transitivity
    (sed_mul
       (sed_add (brown1972_sed_oct_embed (sed_lo x))
                (brown1972_sed_poly_embed (sed_hi x)))
       (sed_add (brown1972_sed_oct_embed (sed_lo y))
                (brown1972_sed_poly_embed (sed_hi y)))).
  - apply brown1972_theorem_6_5_standard_octonion_sedenion_lift.
  - rewrite <- (brown1972_sedenion_adjoined_polynomial_decompose x).
    rewrite <- (brown1972_sedenion_adjoined_polynomial_decompose y).
    reflexivity.
Qed.

Corollary brown1972_corollary_6_7_sedenion_decomposed : forall x y : CDSed,
  brown1972_ch6_67_polynomial_mul (sed_lo x) (sed_hi x) (sed_lo y) (sed_hi y) =
  sed_mul x y.
Proof.
  intros x y.
  transitivity
    (sed_mul
       (sed_add (brown1972_sed_oct_embed (sed_lo x))
                (brown1972_sed_poly_embed (sed_hi x)))
       (sed_add (brown1972_sed_oct_embed (sed_lo y))
                (brown1972_sed_poly_embed (sed_hi y)))).
  - apply brown1972_corollary_6_7_sedenion_lift.
  - rewrite <- (brown1972_sedenion_adjoined_polynomial_decompose x).
    rewrite <- (brown1972_sedenion_adjoined_polynomial_decompose y).
    reflexivity.
Qed.

Section BrownChapterVIAdjoinedPolynomialDecomposition.
  Context {Base Ext : Type}.
  Variable base_mul : Base -> Base -> Base.
  Variable base_neg : Base -> Base.
  Variable base_conj : Base -> Base.
  Variable ext_add : Ext -> Ext -> Ext.
  Variable ext_mul : Ext -> Ext -> Ext.
  Variable ext_assoc : Ext -> Ext -> Ext -> Ext.
  Variable ext_neg : Ext -> Ext.
  Variable base_embed poly_embed : Base -> Ext.
  Variable adjoined_e : Ext.
  Variable poly_mul : Base -> Base -> Base -> Base -> Ext.
  Variable ext_lo ext_hi : Ext -> Base.

  Variable Lift :
    Brown1972ChapterVIAdjoinedPolynomialLiftSurface
      Base Ext
      base_mul base_neg base_conj
      ext_add ext_mul ext_assoc ext_neg
      base_embed poly_embed
      adjoined_e poly_mul.

  Hypothesis ext_decompose : forall x : Ext,
    x = ext_add (base_embed (ext_lo x))
                (poly_embed (ext_hi x)).

  Theorem brown1972_ch6_t65_decomposed : forall x y : Ext,
    poly_mul (ext_lo x) (ext_hi x) (ext_lo y) (ext_hi y) =
    ext_mul x y.
  Proof.
    intros x y.
    destruct Lift as [H64 H65 H66i H66ii H66iii H67].
    transitivity
      (ext_mul (ext_add (base_embed (ext_lo x)) (poly_embed (ext_hi x)))
               (ext_add (base_embed (ext_lo y)) (poly_embed (ext_hi y)))).
    - apply H65.
    - rewrite <- (ext_decompose x).
      rewrite <- (ext_decompose y).
      reflexivity.
  Qed.

  Theorem brown1972_ch6_c67_decomposed : forall x y : Ext,
    poly_mul (ext_lo x) (ext_hi x) (ext_lo y) (ext_hi y) =
    ext_mul x y.
  Proof.
    intros x y.
    destruct Lift as [H64 H65 H66i H66ii H66iii H67].
    transitivity
      (ext_mul (ext_add (base_embed (ext_lo x)) (poly_embed (ext_hi x)))
               (ext_add (base_embed (ext_lo y)) (poly_embed (ext_hi y)))).
    - apply H67.
    - rewrite <- (ext_decompose x).
      rewrite <- (ext_decompose y).
      reflexivity.
  Qed.
End BrownChapterVIAdjoinedPolynomialDecomposition.

Record Brown1972ChapterVIAdjoinedPolynomialDecompositionSurface
    {Base Ext : Type}
    (poly_mul : Base -> Base -> Base -> Base -> Ext)
    (ext_lo ext_hi : Ext -> Base)
    (ext_mul : Ext -> Ext -> Ext) := {
  brown1972_ch6_t65_decomp :
    forall x y : Ext,
      poly_mul (ext_lo x) (ext_hi x) (ext_lo y) (ext_hi y) =
      ext_mul x y;
  brown1972_ch6_c67_decomp :
    forall x y : Ext,
      poly_mul (ext_lo x) (ext_hi x) (ext_lo y) (ext_hi y) =
      ext_mul x y
}.

Definition brown1972_sedenion_chapter_vi_adjoined_polynomial_decomposition_surface :
  Brown1972ChapterVIAdjoinedPolynomialDecompositionSurface
    brown1972_ch6_67_polynomial_mul sed_lo sed_hi sed_mul.
Proof.
  refine {| brown1972_ch6_t65_decomp := brown1972_theorem_6_5_sedenion_decomposed;
            brown1972_ch6_c67_decomp :=
              brown1972_corollary_6_7_sedenion_decomposed |}.
Defined.

Record Brown1972ChapterVIStabilizedSurface := {
  brown1972_ch6_stab_oct_basis :
    Brown1972ChapterVIOctonionBasisSurface;
  brown1972_ch6_stab_oct_anticommutator :
    Brown1972ChapterVIOctonionAnticommutatorSurface;
  brown1972_ch6_stab_sed_adjoined :
    Brown1972ChapterVISedenionAdjoinedSurface;
  brown1972_ch6_stab_sed_polynomial :
    Brown1972ChapterVISedenionPolynomialSurface;
  brown1972_ch6_stab_sed_lift :
    Brown1972ChapterVIAdjoinedPolynomialLiftSurface
      CDOct CDSed
      oct_mul oct_neg oct_conj
      sed_add sed_mul sed_assoc sed_neg
      brown1972_sed_oct_embed brown1972_sed_poly_embed
      brown1972_sed_adjoined_e
      brown1972_ch6_67_polynomial_mul;
  brown1972_ch6_stab_sed_decomp :
    Brown1972ChapterVIAdjoinedPolynomialDecompositionSurface
      brown1972_ch6_67_polynomial_mul sed_lo sed_hi sed_mul
}.

Definition brown1972_chapter_vi_stabilized_surface :
  Brown1972ChapterVIStabilizedSurface.
Proof.
  refine {| brown1972_ch6_stab_oct_basis :=
              brown1972_octonion_chapter_vi_basis_surface;
            brown1972_ch6_stab_oct_anticommutator :=
              brown1972_octonion_chapter_vi_anticommutator_surface;
            brown1972_ch6_stab_sed_adjoined :=
              brown1972_sedenion_chapter_vi_adjoined_surface;
            brown1972_ch6_stab_sed_polynomial :=
              brown1972_sedenion_chapter_vi_polynomial_surface;
            brown1972_ch6_stab_sed_lift :=
              brown1972_sedenion_chapter_vi_adjoined_polynomial_lift_surface;
            brown1972_ch6_stab_sed_decomp :=
              brown1972_sedenion_chapter_vi_adjoined_polynomial_decomposition_surface |}.
Defined.
Lemma brown1972_sed_hi_trace_zero_iff_adjoined_commutes_with_conj :
  forall x : CDSed,
    sed_mul x brown1972_sed_adjoined_e =
    sed_mul brown1972_sed_adjoined_e (sed_conj x) <->
    brown1972_oct_trace (sed_hi x) = 0%R.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  split; intro H.
  - cbv [brown1972_sed_adjoined_e brown1972_oct_trace
         sed_mul sed_conj sed_neg sed_hi sed_lo sed_e
         oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
         quat_add quat_neg quat_mul quat_conj quat_zero quat_one
         qa qb qc qd] in H |- *.
    inversion H; subst; lra.
  - cbv [brown1972_sed_adjoined_e brown1972_oct_trace
         sed_mul sed_conj sed_neg sed_hi sed_lo sed_e
         oct_add oct_neg oct_mul oct_conj oct_lo oct_hi oct_zero oct_e
         quat_add quat_neg quat_mul quat_conj quat_zero quat_one
         qa qb qc qd] in H |- *.
    assert (Ha9 : a9 = 0%R) by lra.
    subst a9.
    apply (f_equal2 mkSed).
    + apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
    + apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_lemma_6_8_sedenion : forall x : CDSed,
  sed_mul x brown1972_sed_adjoined_e =
  sed_mul brown1972_sed_adjoined_e (sed_conj x) <->
  brown1972_oct_trace (sed_hi x) = 0%R.
Proof.
  exact brown1972_sed_hi_trace_zero_iff_adjoined_commutes_with_conj.
Qed.

(** Brown's printed 6.9(i) wording is pointwise in [A,B], but the proof on
    p.35 uses the stronger family form and the literal pointwise iff is
    degenerate at [B = 0]. We therefore land the proof-faithful family iff,
    together with the forward pointwise consequence Brown actually needs. *)
Theorem brown1972_lemma_6_9_i_sedenion_forward : forall A B : CDSed,
  brown1972_oct_trace (sed_hi A) = 0%R ->
  sed_mul A (sed_mul brown1972_sed_adjoined_e B) =
  sed_mul brown1972_sed_adjoined_e (sed_mul (sed_conj A) B).
Proof.
  intros [a1 a2]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
          [[b9 b10 b11 b12] [b13 b14 b15 b16]]]
         Htr.
  destruct a1 as [[a1 a2' a3 a4] [a5 a6 a7 a8]].
  destruct a2 as [[a9 a10 a11 a12] [a13 a14 a15 a16]].
  cbv [brown1972_oct_trace brown1972_sed_adjoined_e
       sed_mul sed_conj sed_neg sed_hi sed_lo sed_e
       oct_mul oct_conj oct_neg oct_lo oct_hi oct_zero oct_e
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd] in Htr |- *.
  assert (Ha9 : a9 = 0%R) by lra.
  subst a9.
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_lemma_6_9_i_sedenion_family : forall A : CDSed,
  (forall B : CDSed,
      sed_mul A (sed_mul brown1972_sed_adjoined_e B) =
      sed_mul brown1972_sed_adjoined_e (sed_mul (sed_conj A) B)) <->
  brown1972_oct_trace (sed_hi A) = 0%R.
Proof.
  intro A.
  split.
  - intro Hall.
    specialize (Hall sed_one).
    rewrite sed_mul_one_right in Hall.
    rewrite sed_mul_one_right in Hall.
    exact (proj1 (brown1972_lemma_6_8_sedenion A) Hall).
  - intro Htr.
    intro B.
    exact (brown1972_lemma_6_9_i_sedenion_forward A B Htr).
Qed.

(** Brown's scanned p.35 wording for 6.9(ii) is pointwise, but the literal
    standard-pair converse degenerates on easy cases such as [A = 1]. In the
    repo's concrete pair coordinates, the constructive direction also needs
    the same [T(a_2)=0] purity used in 6.9(i). *)
Theorem brown1972_lemma_6_9_ii_sedenion_of_trace_conditions :
    forall A B : CDSed,
  brown1972_oct_trace (sed_hi A) = 0%R ->
  brown1972_oct_trace (sed_hi (sed_mul A B)) = 0%R ->
  brown1972_oct_trace (sed_hi B) = 0%R ->
  sed_mul (sed_mul brown1972_sed_adjoined_e A) B =
  sed_mul brown1972_sed_adjoined_e (sed_mul B A).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
         [[b9 b10 b11 b12] [b13 b14 b15 b16]]]
         Ha Hab Hb.
  cbv [brown1972_oct_trace brown1972_sed_adjoined_e
       sed_mul sed_conj sed_neg sed_hi sed_lo sed_e
       oct_mul oct_conj oct_neg oct_lo oct_hi oct_zero oct_e
       quat_mul quat_add quat_neg quat_conj quat_norm_sq
       quat_one quat_zero qa qb qc qd] in Ha, Hab, Hb |- *.
  ring_simplify in Ha.
  ring_simplify in Hab.
  ring_simplify in Hb.
  assert (Ha9 : a9 = 0%R) by nra.
  assert (Hb9 : b9 = 0%R) by nra.
  subst a9.
  subst b9.
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); nra.
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); nra.
Qed.

(** Brown's scanned p.35 wording for 6.9(iii) has the same issue: we land the
    proof-faithful constructive implication rather than a false literal
    pointwise converse in standard-pair coordinates. *)
Theorem brown1972_lemma_6_9_iii_sedenion_of_trace_conditions :
    forall A B : CDSed,
  brown1972_oct_trace (sed_hi (sed_mul A (sed_conj B))) = 0%R ->
  brown1972_oct_trace (sed_hi B) = 0%R ->
  sed_mul (sed_mul brown1972_sed_adjoined_e A)
          (sed_mul brown1972_sed_adjoined_e B) =
  sed_neg (sed_mul B (sed_conj A)).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
         [[b9 b10 b11 b12] [b13 b14 b15 b16]]]
         Hab Hb.
  cbv [brown1972_oct_trace brown1972_sed_adjoined_e
       sed_mul sed_conj sed_neg sed_hi sed_lo sed_e
       oct_mul oct_conj oct_neg oct_lo oct_hi oct_zero oct_e
       quat_mul quat_add quat_neg quat_conj quat_norm_sq
       quat_one quat_zero qa qb qc qd] in Hab, Hb |- *.
  ring_simplify in Hab.
  ring_simplify in Hb.
  assert (Hb9 : b9 = 0%R) by nra.
  subst b9.
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); nra.
  - apply (f_equal2 mkOct); apply (f_equal4 mkQuat); nra.
Qed.

Record Brown1972ChapterVIAdjoinedConjugationDecompositionSurface
  (Ext : Type)
  (trace_hi : Ext -> R)
  (mul : Ext -> Ext -> Ext)
  (conj neg : Ext -> Ext)
  (adjoined_e : Ext) := {
  brown1972_ch6_acd_68 :
    forall x,
      mul x adjoined_e = mul adjoined_e (conj x) <->
      trace_hi x = 0%R;
  brown1972_ch6_acd_69_i_forward :
    forall A B,
      trace_hi A = 0%R ->
      mul A (mul adjoined_e B) =
      mul adjoined_e (mul (conj A) B);
  brown1972_ch6_acd_69_i_family :
    forall A,
      (forall B,
          mul A (mul adjoined_e B) =
          mul adjoined_e (mul (conj A) B)) <->
      trace_hi A = 0%R;
  brown1972_ch6_acd_69_ii :
    forall A B,
      trace_hi A = 0%R ->
      trace_hi (mul A B) = 0%R ->
      trace_hi B = 0%R ->
      mul (mul adjoined_e A) B =
      mul adjoined_e (mul B A);
  brown1972_ch6_acd_69_iii :
    forall A B,
      trace_hi (mul A (conj B)) = 0%R ->
      trace_hi B = 0%R ->
      mul (mul adjoined_e A) (mul adjoined_e B) =
      neg (mul B (conj A))
}.

Definition brown1972_sedenion_chapter_vi_adjoined_conjugation_decomposition_surface :
  Brown1972ChapterVIAdjoinedConjugationDecompositionSurface
    CDSed
    (fun x => brown1972_oct_trace (sed_hi x))
    sed_mul sed_conj sed_neg brown1972_sed_adjoined_e.
Proof.
  refine {| brown1972_ch6_acd_68 :=
              brown1972_lemma_6_8_sedenion;
            brown1972_ch6_acd_69_i_forward :=
              brown1972_lemma_6_9_i_sedenion_forward;
            brown1972_ch6_acd_69_i_family :=
              brown1972_lemma_6_9_i_sedenion_family;
            brown1972_ch6_acd_69_ii :=
              brown1972_lemma_6_9_ii_sedenion_of_trace_conditions;
            brown1972_ch6_acd_69_iii :=
              brown1972_lemma_6_9_iii_sedenion_of_trace_conditions |}.
Defined.

Record Brown1972ChapterVIExtendedStabilizedSurface := {
  brown1972_ch6_ext_base :
    Brown1972ChapterVIStabilizedSurface;
  brown1972_ch6_ext_adjoined_conjugation :
    Brown1972ChapterVIAdjoinedConjugationDecompositionSurface
      CDSed
      (fun x => brown1972_oct_trace (sed_hi x))
      sed_mul sed_conj sed_neg brown1972_sed_adjoined_e
}.

Definition brown1972_chapter_vi_extended_stabilized_surface :
  Brown1972ChapterVIExtendedStabilizedSurface.
Proof.
  refine {| brown1972_ch6_ext_base :=
              brown1972_chapter_vi_stabilized_surface;
            brown1972_ch6_ext_adjoined_conjugation :=
              brown1972_sedenion_chapter_vi_adjoined_conjugation_decomposition_surface |}.
Defined.

Record Brown1972ChapterVIReusableAdjoinedInterface
  (Base Ext : Type)
  (base_mul : Base -> Base -> Base)
  (base_neg base_conj : Base -> Base)
  (ext_add ext_mul : Ext -> Ext -> Ext)
  (ext_assoc : Ext -> Ext -> Ext -> Ext)
  (ext_neg ext_conj : Ext -> Ext)
  (trace_hi : Ext -> R)
  (base_embed poly_embed : Base -> Ext)
  (adjoined_e : Ext)
  (poly_mul : Base -> Base -> Base -> Base -> Ext) := {
  brown1972_ch6_reuse_poly :
    Brown1972ChapterVIAdjoinedPolynomialLiftSurface
      Base Ext
      base_mul base_neg base_conj
      ext_add ext_mul ext_assoc ext_neg
      base_embed poly_embed
      adjoined_e poly_mul;
  brown1972_ch6_reuse_conj :
    Brown1972ChapterVIAdjoinedConjugationDecompositionSurface
      Ext trace_hi ext_mul ext_conj ext_neg adjoined_e
}.

Definition brown1972_sedenion_chapter_vi_reusable_adjoined_interface :
  Brown1972ChapterVIReusableAdjoinedInterface
    CDOct CDSed
    oct_mul oct_neg oct_conj
    sed_add sed_mul sed_assoc sed_neg sed_conj
    (fun x => brown1972_oct_trace (sed_hi x))
    brown1972_sed_oct_embed brown1972_sed_poly_embed
    brown1972_sed_adjoined_e
    brown1972_ch6_67_polynomial_mul.
Proof.
  refine {| brown1972_ch6_reuse_poly :=
              brown1972_sedenion_chapter_vi_adjoined_polynomial_lift_surface;
            brown1972_ch6_reuse_conj :=
              brown1972_sedenion_chapter_vi_adjoined_conjugation_decomposition_surface |}.
Defined.

Record Brown1972ChapterVIReusableAnchorSurface := {
  brown1972_ch6_anchor_base :
    Brown1972ChapterVIExtendedStabilizedSurface;
  brown1972_ch6_anchor_reusable_adjoined :
    Brown1972ChapterVIReusableAdjoinedInterface
      CDOct CDSed
      oct_mul oct_neg oct_conj
      sed_add sed_mul sed_assoc sed_neg sed_conj
      (fun x => brown1972_oct_trace (sed_hi x))
      brown1972_sed_oct_embed brown1972_sed_poly_embed
      brown1972_sed_adjoined_e
      brown1972_ch6_67_polynomial_mul
}.

Definition brown1972_chapter_vi_reusable_anchor_surface :
  Brown1972ChapterVIReusableAnchorSurface.
Proof.
  refine {| brown1972_ch6_anchor_base :=
              brown1972_chapter_vi_extended_stabilized_surface;
            brown1972_ch6_anchor_reusable_adjoined :=
              brown1972_sedenion_chapter_vi_reusable_adjoined_interface |}.
Defined.

Record Brown1972ChapterVIBaseExtensionLiftSurface
  (Base Ext : Type)
  (base_mul : Base -> Base -> Base)
  (base_neg base_conj : Base -> Base)
  (ext_add ext_mul : Ext -> Ext -> Ext)
  (ext_assoc : Ext -> Ext -> Ext -> Ext)
  (ext_neg ext_conj : Ext -> Ext)
  (trace_hi : Ext -> R)
  (base_embed poly_embed : Base -> Ext)
  (adjoined_e : Ext)
  (poly_mul : Base -> Base -> Base -> Base -> Ext)
  (ext_lo ext_hi : Ext -> Base) := {
  brown1972_ch6_bel_reuse :
    Brown1972ChapterVIReusableAdjoinedInterface
      Base Ext
      base_mul base_neg base_conj
      ext_add ext_mul ext_assoc ext_neg ext_conj
      trace_hi base_embed poly_embed adjoined_e poly_mul;
  brown1972_ch6_bel_decomp :
    Brown1972ChapterVIAdjoinedPolynomialDecompositionSurface
      poly_mul ext_lo ext_hi ext_mul
}.

Definition brown1972_sedenion_chapter_vi_base_extension_lift_surface :
  Brown1972ChapterVIBaseExtensionLiftSurface
    CDOct CDSed
    oct_mul oct_neg oct_conj
    sed_add sed_mul sed_assoc sed_neg sed_conj
    (fun x => brown1972_oct_trace (sed_hi x))
    brown1972_sed_oct_embed brown1972_sed_poly_embed
    brown1972_sed_adjoined_e
    brown1972_ch6_67_polynomial_mul
    sed_lo sed_hi.
Proof.
  refine {| brown1972_ch6_bel_reuse :=
              brown1972_sedenion_chapter_vi_reusable_adjoined_interface;
            brown1972_ch6_bel_decomp :=
              brown1972_sedenion_chapter_vi_adjoined_polynomial_decomposition_surface |}.
Defined.

Record Brown1972ChapterVIBroaderReusableAnchorSurface := {
  brown1972_ch6_broader_base :
    Brown1972ChapterVIReusableAnchorSurface;
  brown1972_ch6_broader_ext :
    Brown1972ChapterVIBaseExtensionLiftSurface
      CDOct CDSed
      oct_mul oct_neg oct_conj
      sed_add sed_mul sed_assoc sed_neg sed_conj
      (fun x => brown1972_oct_trace (sed_hi x))
      brown1972_sed_oct_embed brown1972_sed_poly_embed
      brown1972_sed_adjoined_e
      brown1972_ch6_67_polynomial_mul
      sed_lo sed_hi
}.

Definition brown1972_chapter_vi_broader_reusable_anchor_surface :
  Brown1972ChapterVIBroaderReusableAnchorSurface.
Proof.
  refine {| brown1972_ch6_broader_base :=
              brown1972_chapter_vi_reusable_anchor_surface;
            brown1972_ch6_broader_ext :=
              brown1972_sedenion_chapter_vi_base_extension_lift_surface |}.
Defined.
