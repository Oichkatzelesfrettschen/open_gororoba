(** * C-1637: R300-facing sedenion zero-divisor witness.

    The R300/R2VB sedenion product probe uses the CD-4 witness
    (e1 + e10) * (e5 + e14) = 0.  This file keeps that tuple separate from the
    Moreno-Froloff witness (e3 + e10) * (e6 - e15) = 0, so driver-facing
    evidence does not conflate two representatives of the same 16D frontier. *)

From Stdlib Require Import List Reals.
Import ListNotations.

From OpenGororoba Require Import
  Prelude
  CayleyDicksonAlgebra
  Sedenion
  OctonionNorm
  BoxKite.
From OpenGororobaVerified Require Import C1630_SedenionOctonionDowncast.

Open Scope R_scope.

Definition r300_sed_zd_a : CDSed := sed_add (sed_e 1) (sed_e 10).
Definition r300_sed_zd_b : CDSed := sed_add (sed_e 5) (sed_e 14).
Definition r300_sed_zd_a_assessor_pair : nat * nat := (1%nat, 10%nat).
Definition r300_sed_zd_b_assessor_pair : nat * nat := (5%nat, 14%nat).

Ltac close_r300_sed_case :=
  cbv [r300_sed_zd_a r300_sed_zd_b sed_add sed_mul sed_e sed_zero
       oct_add oct_mul oct_conj oct_e oct_zero
       quat_add quat_mul quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.

Theorem C1637_r300_sed_zero_divisor_product :
  sed_mul r300_sed_zd_a r300_sed_zd_b = sed_zero.
Proof.
  close_r300_sed_case.
Qed.

Theorem C1637_r300_sed_zd_a_nonzero :
  r300_sed_zd_a <> sed_zero.
Proof.
  intro H.
  assert (Hhi := f_equal sed_hi H).
  assert (Hlo := f_equal oct_lo Hhi).
  assert (Hqc := f_equal qc Hlo).
  cbv [r300_sed_zd_a sed_add sed_e sed_zero oct_add oct_e oct_zero
       quat_add quat_zero quat_one sed_hi oct_lo qc] in Hqc.
  lra.
Qed.

Theorem C1637_r300_sed_zd_b_nonzero :
  r300_sed_zd_b <> sed_zero.
Proof.
  intro H.
  assert (Hhi := f_equal sed_hi H).
  assert (Hoh := f_equal oct_hi Hhi).
  assert (Hqc := f_equal qc Hoh).
  cbv [r300_sed_zd_b sed_add sed_e sed_zero oct_add oct_e oct_zero
       quat_add quat_zero quat_one sed_hi oct_hi qc] in Hqc.
  lra.
Qed.

Theorem C1637_r300_sed_zero_divisor :
  sed_mul r300_sed_zd_a r300_sed_zd_b = sed_zero /\
  r300_sed_zd_a <> sed_zero /\
  r300_sed_zd_b <> sed_zero.
Proof.
  split; [| split].
  - exact C1637_r300_sed_zero_divisor_product.
  - exact C1637_r300_sed_zd_a_nonzero.
  - exact C1637_r300_sed_zd_b_nonzero.
Qed.

Theorem C1637_r300_sed_zd_a_hi_nonzero :
  sed_hi r300_sed_zd_a <> oct_zero.
Proof.
  intro H.
  assert (Hlo := f_equal oct_lo H).
  assert (Hqc := f_equal qc Hlo).
  cbv [r300_sed_zd_a sed_add sed_e oct_add oct_e oct_zero
       quat_add quat_zero quat_one sed_hi oct_lo qc] in Hqc.
  lra.
Qed.

Theorem C1637_r300_sed_zd_b_hi_nonzero :
  sed_hi r300_sed_zd_b <> oct_zero.
Proof.
  intro H.
  assert (Hoh := f_equal oct_hi H).
  assert (Hqc := f_equal qc Hoh).
  cbv [r300_sed_zd_b sed_add sed_e oct_add oct_e oct_zero
       quat_add quat_zero quat_one sed_hi oct_hi qc] in Hqc.
  lra.
Qed.

Theorem C1637_r300_sed_zero_divisor_requires_hi_half :
  sed_hi r300_sed_zd_a <> oct_zero /\ sed_hi r300_sed_zd_b <> oct_zero.
Proof.
  split.
  - exact C1637_r300_sed_zd_a_hi_nonzero.
  - exact C1637_r300_sed_zd_b_hi_nonzero.
Qed.

Theorem C1637_r300_sed_zd_a_not_downcast :
  r300_sed_zd_a <> sed_octonion_downcast (sed_lo r300_sed_zd_a).
Proof.
  intro Hshape.
  destruct (C1630_downcast_shape_iff r300_sed_zd_a) as [_ Hdowncast_to_hi].
  exact (C1637_r300_sed_zd_a_hi_nonzero (Hdowncast_to_hi Hshape)).
Qed.

Theorem C1637_r300_sed_zd_b_not_downcast :
  r300_sed_zd_b <> sed_octonion_downcast (sed_lo r300_sed_zd_b).
Proof.
  intro Hshape.
  destruct (C1630_downcast_shape_iff r300_sed_zd_b) as [_ Hdowncast_to_hi].
  exact (C1637_r300_sed_zd_b_hi_nonzero (Hdowncast_to_hi Hshape)).
Qed.

Theorem C1637_r300_sed_zero_divisor_not_downcast :
  r300_sed_zd_a <> sed_octonion_downcast (sed_lo r300_sed_zd_a) /\
  r300_sed_zd_b <> sed_octonion_downcast (sed_lo r300_sed_zd_b).
Proof.
  split.
  - exact C1637_r300_sed_zd_a_not_downcast.
  - exact C1637_r300_sed_zd_b_not_downcast.
Qed.

Theorem C1637_r300_sed_zero_divisor_assessors :
  List.In r300_sed_zd_a_assessor_pair assessors /\
  List.In r300_sed_zd_b_assessor_pair assessors.
Proof.
  split.
  - simpl. left. reflexivity.
  - vm_compute. tauto.
Qed.

Theorem C1637_r300_sed_zero_divisor_boxkite_3 :
  List.In boxkite_3 boxkites /\
  List.In r300_sed_zd_a_assessor_pair boxkite_3 /\
  List.In r300_sed_zd_b_assessor_pair boxkite_3.
Proof.
  split.
  - simpl. right. right. left. reflexivity.
  - split.
    + simpl. left. reflexivity.
    + simpl. right. right. right. left. reflexivity.
Qed.

Record R300SedenionZeroDivisorSurface := {
  r300_sed_zd_surface_product :
    sed_mul r300_sed_zd_a r300_sed_zd_b = sed_zero;
  r300_sed_zd_surface_nonzero :
    r300_sed_zd_a <> sed_zero /\ r300_sed_zd_b <> sed_zero;
  r300_sed_zd_surface_requires_hi_half :
    sed_hi r300_sed_zd_a <> oct_zero /\ sed_hi r300_sed_zd_b <> oct_zero;
  r300_sed_zd_surface_not_downcast :
    r300_sed_zd_a <> sed_octonion_downcast (sed_lo r300_sed_zd_a) /\
    r300_sed_zd_b <> sed_octonion_downcast (sed_lo r300_sed_zd_b);
  r300_sed_zd_surface_assessors :
    List.In r300_sed_zd_a_assessor_pair assessors /\
    List.In r300_sed_zd_b_assessor_pair assessors;
  r300_sed_zd_surface_boxkite :
    List.In boxkite_3 boxkites /\
    List.In r300_sed_zd_a_assessor_pair boxkite_3 /\
    List.In r300_sed_zd_b_assessor_pair boxkite_3
}.

Definition C1637_r300_sed_zero_divisor_surface :
  R300SedenionZeroDivisorSurface.
Proof.
  refine
    {| r300_sed_zd_surface_product := C1637_r300_sed_zero_divisor_product;
       r300_sed_zd_surface_nonzero := _;
       r300_sed_zd_surface_requires_hi_half :=
         C1637_r300_sed_zero_divisor_requires_hi_half;
       r300_sed_zd_surface_not_downcast :=
         C1637_r300_sed_zero_divisor_not_downcast;
       r300_sed_zd_surface_assessors :=
         C1637_r300_sed_zero_divisor_assessors;
       r300_sed_zd_surface_boxkite :=
         C1637_r300_sed_zero_divisor_boxkite_3 |}.
  split.
  - exact C1637_r300_sed_zd_a_nonzero.
  - exact C1637_r300_sed_zd_b_nonzero.
Defined.
