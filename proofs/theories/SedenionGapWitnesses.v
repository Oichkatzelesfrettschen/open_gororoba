(** * SedenionGapWitnesses: Shared norm-4 associator witnesses at dim=16.

    Packages the concrete gap witnesses used by C-1134 and C-1137 so they
    can be derived from basis XOR + sign structure instead of repeated full
    sedenion norm expansion. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import SedenionAssociator CDLinearLemmas SedenionSignBridge.

Open Scope R_scope.

Ltac close_sed_gap_by p1 p2 p3 p4 :=
  unfold sed_assoc_norm_sq, sed_assoc, sed_sub, sed_add, sed_neg;
  rewrite p1, p2;
  repeat rewrite sed_mul_scale_left;
  repeat rewrite sed_mul_scale_right;
  rewrite p3, p4;
  cbv [sed_norm_sq sed_scale sed_e
       oct_norm_sq oct_scale oct_e oct_zero
       quat_norm_sq quat_scale quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  simpl;
  ring_simplify; lra.

Theorem sed_gap_e1_e9_e2_norm :
  sed_assoc_norm_sq (sed_e 1) (sed_e 9) (sed_e 2) = 4.
Proof. close_sed_gap_by sed_mul_e1_e9 sed_mul_e9_e2 sed_mul_e8_e2 sed_mul_e1_e11. Qed.

Theorem sed_gap_e1_e2_e4_norm :
  sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 4) = 4.
Proof. close_sed_gap_by sed_mul_e1_e2 sed_mul_e2_e4 sed_mul_e3_e4 sed_mul_e1_e6. Qed.

Theorem sed_missing_gap_1_9 :
  sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 9) = 4.
Proof. close_sed_gap_by sed_mul_e1_e2 sed_mul_e2_e9 sed_mul_e3_e9 sed_mul_e1_e11. Qed.

Theorem sed_missing_gap_2_10 :
  sed_assoc_norm_sq (sed_e 2) (sed_e 1) (sed_e 10) = 4.
Proof. close_sed_gap_by sed_mul_e2_e1 sed_mul_e1_e10 sed_mul_e3_e10 sed_mul_e2_e11. Qed.

Theorem sed_missing_gap_3_11 :
  sed_assoc_norm_sq (sed_e 3) (sed_e 1) (sed_e 11) = 4.
Proof. close_sed_gap_by sed_mul_e3_e1 sed_mul_e1_e11 sed_mul_e2_e11 sed_mul_e3_e10. Qed.

Theorem sed_missing_gap_4_12 :
  sed_assoc_norm_sq (sed_e 4) (sed_e 1) (sed_e 12) = 4.
Proof. close_sed_gap_by sed_mul_e4_e1 sed_mul_e1_e12 sed_mul_e5_e12 sed_mul_e4_e13. Qed.

Theorem sed_missing_gap_5_13 :
  sed_assoc_norm_sq (sed_e 5) (sed_e 1) (sed_e 13) = 4.
Proof. close_sed_gap_by sed_mul_e5_e1 sed_mul_e1_e13 sed_mul_e4_e13 sed_mul_e5_e12. Qed.

Theorem sed_missing_gap_6_14 :
  sed_assoc_norm_sq (sed_e 6) (sed_e 1) (sed_e 14) = 4.
Proof. close_sed_gap_by sed_mul_e6_e1 sed_mul_e1_e14 sed_mul_e7_e14 sed_mul_e6_e15. Qed.

Theorem sed_missing_gap_7_15 :
  sed_assoc_norm_sq (sed_e 7) (sed_e 1) (sed_e 15) = 4.
Proof. close_sed_gap_by sed_mul_e7_e1 sed_mul_e1_e15 sed_mul_e6_e15 sed_mul_e7_e14. Qed.
