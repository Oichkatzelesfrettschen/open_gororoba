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
From OpenGororoba Require Import Brown1972ChapterVI.

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
  exact zd_fundamental_major_theorem.
Qed.

Theorem brown1972_chapter_vii_boxkite_partition_summary :
  length assessors = 42 /\
  length boxkites = 7 /\
  List.map (@length _) boxkites = [6; 6; 6; 6; 6; 6; 6] /\
  List.fold_left Nat.add (List.map (@length _) boxkites) 0 = 42 /\
  (ZDGraph.boxkite_signatures = [15; 10; 11; 12; 13; 14; 9]).
Proof.
  exact boxkite_partition_summary.
Qed.

Theorem brown1972_chapter_vii_assessor_bridge_summary :
  (42 * 4 = 168) /\
  (7 * 6 * 4 = 168) /\
  (6 * 4 = 24) /\
  (ZDGraph.boxkite_signatures = 15 :: 10 :: 11 :: 12 :: 13 :: 14 :: 9 :: nil).
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
    length assessors = 42 /\
    length boxkites = 7 /\
    List.map (@length _) boxkites = [6; 6; 6; 6; 6; 6; 6] /\
    List.fold_left Nat.add (List.map (@length _) boxkites) 0 = 42 /\
    (ZDGraph.boxkite_signatures = [15; 10; 11; 12; 13; 14; 9]);
  brown1972_ch7_zd_bridge :
    (42 * 4 = 168) /\
    (7 * 6 * 4 = 168) /\
    (6 * 4 = 24) /\
    (ZDGraph.boxkite_signatures = 15 :: 10 :: 11 :: 12 :: 13 :: 14 :: 9 :: nil)
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
    sed_mul_fused sed_zd_b sed_zd_a = sed_zero
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
              brown1972_chapter_vii_theorem_7_3_witness_fused_basis |}.
Defined.
