(** * Brown1972ChapterVII: Zero-divisor wrapper surface for Brown (1972).

    This file extracts the current Brown Chapter VII Rocq landing from the
    paper-level aggregator. The theorem content still comes from the existing
    zero-divisor companion files, but the Brown-facing chapter surface now
    lives in its own chapter file.

    Current sourced landing:
    - Brown 7.3 concrete symmetry witness via `C1538_MorZDSymmetry.v`
    - Brown 7.15 fundamental criterion witness via `ZD_Criterion.v`
    - Brown/de Marrais box-kite partition and bridge summaries via
      `BrownAssessorEquivalence.v`

    The remaining Chapter VII backlog is still the broader Brown-numbered
    zero-divisor lane beyond these landed wrappers. *)

From Stdlib Require Import List Reals ZArith Lia Lra.
Import ListNotations.
Open Scope R_scope.

From OpenGororoba Require Import
  BoxKite
  ZDGraph
  Sedenion
  CayleyDicksonAlgebra
  OctonionNorm
  CDAssociator
  CDPowerAssociative
  CDSignBridge
  CDConjAntimorph
  CDFusedBilinear
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
From OpenGororoba Require Import Brown1972ChapterVI Brown1972ChapterV Brown1972ChapterIII
  SStructuralGaps.

Theorem brown1972_chapter_vii_theorem_7_3_witness :
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  exact C1538_sedenion_zd_symmetry.
Qed.

Theorem brown1972_chapter_vii_theorem_7_3_witness_fused :
  sed_mul_fused sed_zd_a sed_zd_b = sed_zero /\
  sed_mul_fused sed_zd_b sed_zd_a = sed_zero.
Proof.
  split.
  - rewrite sed_mul_fused_eq.
    exact sed_zd_product_zero.
  - rewrite sed_mul_fused_eq.
    exact sed_zd_product_zero_rev.
Qed.

Ltac brown1972_ch7_close_sed_ring :=
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.

Lemma brown1972_chapter_vii_zd_a_basis_decompose :
  sed_zd_a = sed_add (sed_e 3) (sed_e 10).
Proof.
  vm_compute.
  brown1972_ch7_close_sed_ring.
Qed.

Lemma brown1972_chapter_vii_zd_b_basis_decompose :
  sed_zd_b = sed_add (sed_e 6) (sed_scale (-1) (sed_e 15)).
Proof.
  vm_compute.
  brown1972_ch7_close_sed_ring.
Qed.

Theorem brown1972_chapter_vii_theorem_7_3_witness_fused_basis :
  sed_mul_fused sed_zd_a sed_zd_b = sed_zero /\
  sed_mul_fused sed_zd_b sed_zd_a = sed_zero.
Proof.
  destruct sed_fused_bilinear_surface as [_ HaddL HaddR HscaleL HscaleR].
  split.
  - rewrite brown1972_chapter_vii_zd_a_basis_decompose.
    rewrite brown1972_chapter_vii_zd_b_basis_decompose.
    rewrite HaddL.
    rewrite HaddR.
    rewrite HaddR.
    rewrite HscaleR.
    repeat rewrite sed_mul_fused_basis_xor by lia.
    vm_compute.
    brown1972_ch7_close_sed_ring.
  - rewrite brown1972_chapter_vii_zd_b_basis_decompose.
    rewrite brown1972_chapter_vii_zd_a_basis_decompose.
    rewrite HaddL.
    rewrite HscaleL.
    rewrite HaddR.
    rewrite HaddR.
    repeat rewrite sed_mul_fused_basis_xor by lia.
    vm_compute.
    brown1972_ch7_close_sed_ring.
Qed.

Theorem brown1972_chapter_vii_theorem_7_15_fundamental :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  exact zd_fundamental_major_theorem_fused.
Qed.

Theorem brown1972_chapter_vii_theorem_7_15_fundamental_fused_support :
  zd_condition_ii zd_a1_fundamental zd_a2_fundamental
                  zd_b1_fundamental zd_b2_fundamental /\
  zd_condition_iii zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental.
Proof.
  split.
  - exact zd_fundamental_condition_ii_fused.
  - exact zd_fundamental_condition_iii_fused.
Qed.

(** The full Brown Cor. 7.16 symmetry-group reduction is still beyond the
    current formalized orbit infrastructure. The current Brown-facing landing
    is the finite structural summary that packages the fused 7.15 witness
    together with the box-kite count and signature data that the zero-divisor
    corpus already uses concretely. *)
Theorem brown1972_chapter_vii_corollary_7_16_structural_summary :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental /\
  ((42 * 4)%nat = 168%nat) /\
  ((7 * 6 * 4)%nat = 168%nat) /\
  ((6 * 4)%nat = 24%nat) /\
  (ZDGraph.boxkite_signatures =
     15%nat :: 10%nat :: 11%nat :: 12%nat :: 13%nat :: 14%nat :: 9%nat :: nil).
Proof.
  split.
  - exact brown1972_chapter_vii_theorem_7_15_fundamental.
  - split.
    + reflexivity.
    + split.
      * reflexivity.
      * split.
        { reflexivity. }
        { exact DeMarraisAssessors.bk_g_indices. }
Qed.

Theorem brown1972_chapter_vii_corollary_7_16_ii_fundamental_fused :
  oct_antiassociator_fused zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental = oct_zero /\
  oct_antiassociator_fused zd_b1_fundamental zd_a1_fundamental zd_b2_fundamental = oct_zero /\
  oct_antiassociator_fused zd_a1_fundamental zd_b2_fundamental zd_a2_fundamental = oct_zero /\
  oct_antiassociator_fused zd_b1_fundamental zd_a2_fundamental zd_b2_fundamental = oct_zero /\
  oct_antiassociator_fused zd_a2_fundamental zd_b1_fundamental zd_a1_fundamental = oct_zero /\
  oct_antiassociator_fused zd_b2_fundamental zd_a1_fundamental zd_b1_fundamental = oct_zero /\
  oct_antiassociator_fused zd_a2_fundamental zd_b2_fundamental zd_a1_fundamental = oct_zero /\
  oct_antiassociator_fused zd_b2_fundamental zd_a2_fundamental zd_b1_fundamental = oct_zero.
Proof.
  exact zd_fundamental_corollary_7_16_ii_fused.
Qed.

Theorem brown1972_chapter_vii_lemma_7_17_octonion :
  forall a b c : CDOct,
    oct_norm_sq a > 0 ->
    oct_norm_sq b > 0 ->
    oct_norm_sq c > 0 ->
    oct_antiassociator_fused a b c = oct_zero ->
    brown1972_oct_trace a = 0%R /\
    brown1972_oct_trace b = 0%R /\
    brown1972_oct_trace c = 0%R.
Proof.
  intros a b c Hna Hnb Hnc Hanti.
  rewrite oct_antiassociator_fused_eq in Hanti.
  exact (s2_brown_lemma717_abstract a b c Hna Hnb Hnc Hanti).
Qed.

Definition brown1972_ch7_718_basis_rhs (i j k : nat) : Prop :=
  (i <> 0)%nat /\
  (j <> 0)%nat /\
  (k <> 0)%nat /\
  i <> j /\
  j <> k /\
  i <> k /\
  Nat.lxor i j <> k.

Lemma brown1972_ch7_718_positive_false_iff :
  forall i j k : nat,
    (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
    (brown1972_ch6_615_positive i j k = false <->
     brown1972_ch7_718_basis_rhs i j k).
Proof.
  intros i j k Hi Hj Hk.
  unfold brown1972_ch7_718_basis_rhs, brown1972_ch6_615_positive.
  destruct i as [|[|[|[|[|[|[|[|i]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|j]]]]]]]]; try lia;
  destruct k as [|[|[|[|[|[|[|[|k]]]]]]]]; try lia;
  vm_compute; split; intro H; intuition congruence.
Qed.

Lemma brown1972_ch7_basis_double :
  forall n : nat,
    (n < 8)%nat ->
    oct_add (oct_e n) (oct_e n) = oct_scale 2 (oct_e n).
Proof.
  intros n Hn.
  destruct n as [|[|[|[|[|[|[|[|n]]]]]]]]; try lia;
  brown1972_close_oct_ring.
Qed.

Lemma brown1972_ch7_add_scaled_basis_same :
  forall c : R, forall n : nat,
    (n < 8)%nat ->
    oct_add (oct_scale c (oct_e n)) (oct_scale c (oct_e n)) =
    oct_scale (2 * c) (oct_e n).
Proof.
  intros c n Hn.
  rewrite <- brown1972_oct_scale_add_distr.
  rewrite brown1972_ch7_basis_double by exact Hn.
  rewrite brown1972_oct_scale_scale.
  replace (c * 2)%R with (2 * c)%R by ring.
  reflexivity.
Qed.

Theorem brown1972_chapter_vii_theorem_7_18_basis_xor_form :
  forall i j k : nat,
    (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
    (oct_antiassociator_fused (oct_e i) (oct_e j) (oct_e k) = oct_zero <->
     brown1972_ch7_718_basis_rhs i j k).
Proof.
  intros i j k Hi Hj Hk.
  assert (Hjk : (Nat.lxor j k < 8)%nat).
  { apply brown1972_oct_lxor_lt8; assumption. }
  assert (Hrhs :
    oct_mul (oct_e i) (oct_mul (oct_e j) (oct_e k)) =
    oct_scale
      (sign_to_R (oct_sign j k) * sign_to_R (oct_sign i (Nat.lxor j k)))
      (oct_e (Nat.lxor i (Nat.lxor j k)))).
  {
    destruct oct_fused_bilinear_surface as [_ _ _ _ HscaleR].
    repeat rewrite <- oct_mul_fused_eq.
    rewrite oct_mul_fused_basis_xor with (i := j) (j := k) by assumption.
    rewrite HscaleR.
    rewrite oct_mul_fused_basis_xor with (i := i) (j := Nat.lxor j k) by assumption.
    rewrite brown1972_oct_scale_scale.
    reflexivity.
  }
  assert (Hijk : (Nat.lxor i (Nat.lxor j k) < 8)%nat).
  { apply brown1972_oct_lxor_lt8; assumption. }
  rewrite oct_antiassociator_fused_eq.
  unfold oct_antiassociator.
  rewrite brown1972_lemma_6_15_octonion by assumption.
  rewrite Hrhs.
  destruct (brown1972_ch6_615_positive i j k) eqn:Hpos.
  - split.
    + intro Hzero.
      exfalso.
      unfold brown1972_ch6_615_oct_rhs in Hzero.
      rewrite Hpos in Hzero.
      simpl in Hzero.
      rewrite Hrhs in Hzero.
      set (c :=
        (sign_to_R (oct_sign j k) * sign_to_R (oct_sign i (Nat.lxor j k)))%R) in *.
      rewrite brown1972_ch7_add_scaled_basis_same in Hzero by exact Hijk.
      assert (Hnorm0 :
        oct_norm_sq (oct_scale (2 * c) (oct_e (Nat.lxor i (Nat.lxor j k)))) = 0%R).
      {
        apply (f_equal oct_norm_sq) in Hzero.
        replace (oct_norm_sq oct_zero) with 0%R in Hzero by
          (cbv [oct_norm_sq oct_zero quat_norm_sq quat_zero oct_lo oct_hi
                qa qb qc qd];
           ring).
        exact Hzero.
      }
      rewrite s2_oct_norm_sq_scale in Hnorm0.
      rewrite oct_e_norm in Hnorm0 by exact Hijk.
      assert (Hcjk : (sign_to_R (oct_sign j k) * sign_to_R (oct_sign j k) = 1)%R).
      { apply brown1972_sign_to_R_square_one. }
      assert (Hcix : (sign_to_R (oct_sign i (Nat.lxor j k)) *
                      sign_to_R (oct_sign i (Nat.lxor j k)) = 1)%R).
      { apply brown1972_sign_to_R_square_one. }
      unfold c in Hnorm0.
      nra.
    + intro Hbasis.
      pose proof (proj2 (brown1972_ch7_718_positive_false_iff i j k Hi Hj Hk) Hbasis)
        as Hfalse.
      rewrite Hpos in Hfalse.
      discriminate Hfalse.
  - split; intro Hbranch.
    { apply (proj1 (brown1972_ch7_718_positive_false_iff i j k Hi Hj Hk)).
      exact Hpos. }
    { clear Hbranch.
      unfold brown1972_ch6_615_oct_rhs.
      rewrite Hpos.
      simpl.
      rewrite Hrhs.
      rewrite oct_add_comm.
      apply oct_add_neg_cancel. }
Qed.

Theorem brown1972_chapter_vii_boxkite_partition_summary :
  length assessors = 42%nat /\
  length boxkites = 7%nat /\
  List.map (@length _) boxkites = [6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat] /\
  List.fold_left Nat.add (List.map (@length _) boxkites) 0%nat = 42%nat /\
  (ZDGraph.boxkite_signatures = [15%nat; 10%nat; 11%nat; 12%nat; 13%nat; 14%nat; 9%nat]).
Proof.
  exact boxkite_partition_summary.
Qed.

Theorem brown1972_chapter_vii_assessor_bridge_summary :
  ((42 * 4)%nat = 168%nat) /\
  ((7 * 6 * 4)%nat = 168%nat) /\
  ((6 * 4)%nat = 24%nat) /\
  (ZDGraph.boxkite_signatures =
     15%nat :: 10%nat :: 11%nat :: 12%nat :: 13%nat :: 14%nat :: 9%nat :: nil).
Proof.
  exact brown_demarrais_bridge.
Qed.

Record Brown1972ChapterVIIZeroDivisorSurface := {
  brown1972_ch7_zd_t73 :
    sed_mul sed_zd_a sed_zd_b = sed_zero /\
    sed_mul sed_zd_b sed_zd_a = sed_zero;
  brown1972_ch7_zd_t715 :
    is_zd_pair_major_theorem
      zd_a1_fundamental zd_a2_fundamental
      zd_b1_fundamental zd_b2_fundamental;
  brown1972_ch7_zd_partition :
    length assessors = 42%nat /\
    length boxkites = 7%nat /\
    List.map (@length _) boxkites =
      [6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat] /\
    List.fold_left Nat.add (List.map (@length _) boxkites) 0%nat = 42%nat /\
    (ZDGraph.boxkite_signatures =
      [15%nat; 10%nat; 11%nat; 12%nat; 13%nat; 14%nat; 9%nat]);
  brown1972_ch7_zd_bridge :
    ((42 * 4)%nat = 168%nat) /\
    ((7 * 6 * 4)%nat = 168%nat) /\
    ((6 * 4)%nat = 24%nat) /\
    (ZDGraph.boxkite_signatures =
      15%nat :: 10%nat :: 11%nat :: 12%nat :: 13%nat :: 14%nat :: 9%nat :: nil)
}.

Definition brown1972_chapter_vii_zero_divisor_surface :
  Brown1972ChapterVIIZeroDivisorSurface.
Proof.
  refine {| brown1972_ch7_zd_t73 :=
              brown1972_chapter_vii_theorem_7_3_witness;
            brown1972_ch7_zd_t715 :=
              brown1972_chapter_vii_theorem_7_15_fundamental;
            brown1972_ch7_zd_partition :=
              brown1972_chapter_vii_boxkite_partition_summary;
            brown1972_ch7_zd_bridge :=
              brown1972_chapter_vii_assessor_bridge_summary |}.
Defined.

Record Brown1972ChapterVIIReusableAnchorSurface := {
  brown1972_ch7_anchor_base :
    Brown1972ChapterVIIZeroDivisorSurface;
  brown1972_ch7_anchor_ch6 :
    Brown1972ChapterVIReusableAnchorSurface;
  brown1972_ch7_anchor_oct_basis_fused :
    CDOctBasisFusedSurface;
  brown1972_ch7_anchor_sed_basis_fused :
    CDSedFocusedBasisFusedSurface;
  brown1972_ch7_anchor_fused_sed :
    CDFusedBilinearSurface CDSed sed_add sed_mul sed_mul_fused sed_scale;
  brown1972_ch7_anchor_fused_t73 :
    sed_mul_fused sed_zd_a sed_zd_b = sed_zero /\
    sed_mul_fused sed_zd_b sed_zd_a = sed_zero;
  brown1972_ch7_anchor_fused_t73_basis :
    sed_mul_fused sed_zd_a sed_zd_b = sed_zero /\
    sed_mul_fused sed_zd_b sed_zd_a = sed_zero;
  brown1972_ch7_anchor_fused_t715 :
    zd_condition_ii zd_a1_fundamental zd_a2_fundamental
                    zd_b1_fundamental zd_b2_fundamental /\
    zd_condition_iii zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental;
  brown1972_ch7_anchor_c716 :
    is_zd_pair_major_theorem
      zd_a1_fundamental zd_a2_fundamental
      zd_b1_fundamental zd_b2_fundamental /\
    ((42 * 4)%nat = 168%nat) /\
    ((7 * 6 * 4)%nat = 168%nat) /\
    ((6 * 4)%nat = 24%nat) /\
    (ZDGraph.boxkite_signatures =
      15%nat :: 10%nat :: 11%nat :: 12%nat :: 13%nat :: 14%nat :: 9%nat :: nil);
  brown1972_ch7_anchor_c716_ii_fused :
    oct_antiassociator_fused zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental = oct_zero /\
    oct_antiassociator_fused zd_b1_fundamental zd_a1_fundamental zd_b2_fundamental = oct_zero /\
    oct_antiassociator_fused zd_a1_fundamental zd_b2_fundamental zd_a2_fundamental = oct_zero /\
    oct_antiassociator_fused zd_b1_fundamental zd_a2_fundamental zd_b2_fundamental = oct_zero /\
    oct_antiassociator_fused zd_a2_fundamental zd_b1_fundamental zd_a1_fundamental = oct_zero /\
    oct_antiassociator_fused zd_b2_fundamental zd_a1_fundamental zd_b1_fundamental = oct_zero /\
    oct_antiassociator_fused zd_a2_fundamental zd_b2_fundamental zd_a1_fundamental = oct_zero /\
    oct_antiassociator_fused zd_b2_fundamental zd_a2_fundamental zd_b1_fundamental = oct_zero;
  brown1972_ch7_anchor_l717 :
    forall a b c : CDOct,
      oct_norm_sq a > 0 ->
      oct_norm_sq b > 0 ->
      oct_norm_sq c > 0 ->
      oct_antiassociator_fused a b c = oct_zero ->
      brown1972_oct_trace a = 0%R /\
      brown1972_oct_trace b = 0%R /\
      brown1972_oct_trace c = 0%R;
  brown1972_ch7_anchor_t718 :
    forall i j k : nat,
      (i < 8)%nat -> (j < 8)%nat -> (k < 8)%nat ->
      (oct_antiassociator_fused (oct_e i) (oct_e j) (oct_e k) = oct_zero <->
       brown1972_ch7_718_basis_rhs i j k)
}.

Definition brown1972_chapter_vii_reusable_anchor_surface :
  Brown1972ChapterVIIReusableAnchorSurface.
Proof.
  refine {| brown1972_ch7_anchor_base :=
              brown1972_chapter_vii_zero_divisor_surface;
            brown1972_ch7_anchor_ch6 :=
              brown1972_chapter_vi_reusable_anchor_surface;
            brown1972_ch7_anchor_oct_basis_fused :=
              oct_basis_fused_surface;
            brown1972_ch7_anchor_sed_basis_fused :=
              sed_focused_basis_fused_surface;
            brown1972_ch7_anchor_fused_sed :=
              sed_fused_bilinear_surface;
            brown1972_ch7_anchor_fused_t73 :=
              brown1972_chapter_vii_theorem_7_3_witness_fused;
            brown1972_ch7_anchor_fused_t73_basis :=
              brown1972_chapter_vii_theorem_7_3_witness_fused_basis;
            brown1972_ch7_anchor_fused_t715 :=
              brown1972_chapter_vii_theorem_7_15_fundamental_fused_support;
            brown1972_ch7_anchor_c716 :=
              brown1972_chapter_vii_corollary_7_16_structural_summary;
            brown1972_ch7_anchor_c716_ii_fused :=
              brown1972_chapter_vii_corollary_7_16_ii_fundamental_fused;
            brown1972_ch7_anchor_l717 :=
              brown1972_chapter_vii_lemma_7_17_octonion;
            brown1972_ch7_anchor_t718 :=
              brown1972_chapter_vii_theorem_7_18_basis_xor_form |}.
Defined.
