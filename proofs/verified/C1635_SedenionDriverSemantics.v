(** * C-1635: Driver-facing sedenion semantic boundary.

    A dim-16 input can be routed as 8D (octonion) semantics exactly when its
    sedenion hi-half is zero.  The CD-4 frontier remains a separate 16D
    semantic path: concrete zero-divisor witnesses use nonzero hi-halves, norm
    multiplicativity fails at 16D, and the primitive assessor set partitions
    into 7 box-kites of 6 assessors each. *)

From Stdlib Require Import List Reals.
Import ListNotations.

From OpenGororoba Require Import
  Prelude
  CayleyDicksonAlgebra
  Sedenion
  OctonionNorm
  BoxKite
  ZDGraph
  Brown1972ChapterVII.
From OpenGororobaVerified Require Import
  C002_SedenionZDAndNormFail
  C003_AssessorsBoxkites
  C010_ConnectivityObstruction
  C1135_BoxKiteFusionRules
  C1630_SedenionOctonionDowncast.
From OpenGororobaVerified Require Import
  C1636_Cariow2013SedenionSchedule.
From OpenGororobaVerified Require Import
  C1637_R300SedenionZeroDivisor.
From OpenGororobaVerified Require Import
  C1638_OctonionDowncastNoZeroDivisors.

Open Scope R_scope.

Theorem C1635_zd_a_hi_nonzero : sed_hi sed_zd_a <> oct_zero.
Proof.
  intro H.
  assert (Hlo := f_equal oct_lo H).
  assert (Hqc := f_equal qc Hlo).
  cbv [sed_zd_a sed_hi oct_zero oct_lo quat_zero qc] in Hqc.
  lra.
Qed.

Theorem C1635_zd_b_hi_nonzero : sed_hi sed_zd_b <> oct_zero.
Proof.
  intro H.
  assert (Hhi := f_equal oct_hi H).
  assert (Hqd := f_equal qd Hhi).
  cbv [sed_zd_b sed_hi oct_zero oct_hi quat_zero qd] in Hqd.
  lra.
Qed.

Theorem C1635_zero_divisor_witness_requires_hi_half :
  sed_hi sed_zd_a <> oct_zero /\ sed_hi sed_zd_b <> oct_zero.
Proof.
  split.
  - exact C1635_zd_a_hi_nonzero.
  - exact C1635_zd_b_hi_nonzero.
Qed.

Theorem C1635_zd_a_not_downcast :
  sed_zd_a <> sed_octonion_downcast (sed_lo sed_zd_a).
Proof.
  intro Hshape.
  destruct (C1630_downcast_shape_iff sed_zd_a) as [_ Hdowncast_to_hi].
  exact (C1635_zd_a_hi_nonzero (Hdowncast_to_hi Hshape)).
Qed.

Theorem C1635_zd_b_not_downcast :
  sed_zd_b <> sed_octonion_downcast (sed_lo sed_zd_b).
Proof.
  intro Hshape.
  destruct (C1630_downcast_shape_iff sed_zd_b) as [_ Hdowncast_to_hi].
  exact (C1635_zd_b_hi_nonzero (Hdowncast_to_hi Hshape)).
Qed.

Theorem C1635_zero_divisor_witness_not_downcast :
  sed_zd_a <> sed_octonion_downcast (sed_lo sed_zd_a) /\
  sed_zd_b <> sed_octonion_downcast (sed_lo sed_zd_b).
Proof.
  split.
  - exact C1635_zd_a_not_downcast.
  - exact C1635_zd_b_not_downcast.
Qed.

Theorem C1635_nonzero_zero_divisor_not_downcast_operands :
  forall x y : CDSed,
    x <> sed_zero ->
    y <> sed_zero ->
    sed_mul x y = sed_zero ->
    ~ (sed_hi x = oct_zero /\ sed_hi y = oct_zero).
Proof.
  intros x y Hx_nonzero Hy_nonzero Hprod [Hx_hi Hy_hi].
  destruct
    (C1638_hi_zero_operands_no_zero_divisors x y Hx_hi Hy_hi Hprod)
    as [Hx_zero | Hy_zero].
  - exact (Hx_nonzero Hx_zero).
  - exact (Hy_nonzero Hy_zero).
Qed.

Record SedenionDriverSemanticsSurface := {
  sds_downcast_shape :
    forall x : CDSed,
      sed_hi x = oct_zero <-> x = sed_octonion_downcast (sed_lo x);
  sds_downcast_product :
    forall x y : CDSed,
      sed_hi x = oct_zero ->
      sed_hi y = oct_zero ->
      sed_mul x y = sed_octonion_downcast (oct_mul (sed_lo x) (sed_lo y));
  sds_downcast_product_hi_zero :
    forall x y : CDSed,
      sed_hi x = oct_zero ->
      sed_hi y = oct_zero ->
      sed_hi (sed_mul x y) = oct_zero;
  sds_downcast_no_zero_divisors :
    DowncastNoZeroDivisorSurface;
  sds_zero_divisor :
    sed_mul sed_zd_a sed_zd_b = sed_zero /\
    sed_zd_a <> sed_zero /\
    sed_zd_b <> sed_zero;
  sds_zero_divisor_requires_hi_half :
    sed_hi sed_zd_a <> oct_zero /\ sed_hi sed_zd_b <> oct_zero;
  sds_zero_divisor_not_downcast :
    sed_zd_a <> sed_octonion_downcast (sed_lo sed_zd_a) /\
    sed_zd_b <> sed_octonion_downcast (sed_lo sed_zd_b);
  sds_nonzero_zero_divisor_not_downcast_operands :
    forall x y : CDSed,
      x <> sed_zero ->
      y <> sed_zero ->
      sed_mul x y = sed_zero ->
      ~ (sed_hi x = oct_zero /\ sed_hi y = oct_zero);
  sds_r300_zero_divisor :
    R300SedenionZeroDivisorSurface;
  sds_norm_failure :
    exists x y : CDSed,
      sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y;
  sds_boxkite_partition :
    length assessors = 42%nat /\
    length boxkites = 7%nat /\
    List.map (@length _) boxkites =
      [6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat; 6%nat] /\
    List.fold_left Nat.add (List.map (@length _) boxkites) 0%nat = 42%nat;
  sds_boxkite_signatures_distinct :
    no_dups boxkite_signatures = true;
  sds_boxkite_fusion_discriminator :
    (sed_mul (sed_e 1) (sed_e 14) <> sed_mul (sed_e 2) (sed_e 13)) /\
    (sed_mul (sed_e 1) (sed_e 11) <> sed_mul (sed_e 3) (sed_e 9)) /\
    (sed_mul (sed_e 1) (sed_e 10) <> sed_mul (sed_e 2) (sed_e 9)) /\
    (sed_mul (sed_e 1) (sed_e 13) <> sed_mul (sed_e 5) (sed_e 9)) /\
    (sed_mul (sed_e 1) (sed_e 12) <> sed_mul (sed_e 2) (sed_e 15)) /\
    (sed_mul (sed_e 3) (sed_e 13) <> sed_mul (sed_e 1) (sed_e 15)) /\
    (sed_mul (sed_e 2) (sed_e 11) <> sed_mul (sed_e 3) (sed_e 10));
  sds_brown_zero_divisor_surface :
    Brown1972ChapterVIIZeroDivisorSurface;
  sds_cariow2013_schedule :
    Cariow2013SedenionScheduleSurface
}.

Definition C1635_sedenion_driver_semantics_surface :
  SedenionDriverSemanticsSurface.
Proof.
  refine {| sds_downcast_shape := C1630_downcast_shape_iff;
            sds_downcast_product := C1630_hi_zero_operands_downcast_mul;
            sds_downcast_product_hi_zero := C1630_hi_zero_operands_product_hi_zero;
            sds_downcast_no_zero_divisors :=
              C1638_downcast_no_zero_divisor_surface;
            sds_zero_divisor := C002_zero_divisors;
            sds_zero_divisor_requires_hi_half :=
              C1635_zero_divisor_witness_requires_hi_half;
            sds_zero_divisor_not_downcast :=
              C1635_zero_divisor_witness_not_downcast;
            sds_nonzero_zero_divisor_not_downcast_operands :=
              C1635_nonzero_zero_divisor_not_downcast_operands;
            sds_r300_zero_divisor :=
              C1637_r300_sed_zero_divisor_surface;
            sds_norm_failure := C002_norm_fails;
            sds_boxkite_partition := _;
            sds_boxkite_signatures_distinct := internal_edges_confined;
            sds_boxkite_fusion_discriminator := boxkite_two_fusion_channels;
            sds_brown_zero_divisor_surface :=
              brown1972_chapter_vii_zero_divisor_surface;
            sds_cariow2013_schedule :=
              C1636_cariow2013_sedenion_schedule_surface |}.
  exact
    (conj C003_assessor_count
      (conj C003_boxkite_count
        (conj C003_uniform_size C003_partition_complete))).
Defined.
