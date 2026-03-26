(** * SedenionSignBridge: Concrete basis-product bridge for the dim=16 gap lane.

    This file intentionally covers only the ordered basis products needed by
    C-1134 and C-1137.  The earlier all-256 bridge was mathematically fine but
    too expensive as a cold-start dependency for this hotspot.  These focused
    lemmas preserve the structural proof style while keeping the bridge small. *)

From Stdlib Require Import Bool Arith Reals Lra Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

Open Scope R_scope.

Ltac sed_cd_cbv :=
  cbv [sed_mul sed_conj sed_scale sed_e sed_zero
       oct_mul oct_conj oct_scale oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_scale quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].

Ltac close_sed_case :=
  sed_cd_cbv;
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.

Lemma sed_mul_e1_e9 :
  sed_mul (sed_e 1) (sed_e 9) = sed_scale (-1) (sed_e 8).
Proof. close_sed_case. Qed.

Lemma sed_mul_e9_e2 :
  sed_mul (sed_e 9) (sed_e 2) = sed_scale (-1) (sed_e 11).
Proof. close_sed_case. Qed.

Lemma sed_mul_e8_e2 :
  sed_mul (sed_e 8) (sed_e 2) = sed_scale (-1) (sed_e 10).
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e11 :
  sed_mul (sed_e 1) (sed_e 11) = sed_e 10.
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e2 :
  sed_mul (sed_e 1) (sed_e 2) = sed_e 3.
Proof. close_sed_case. Qed.

Lemma sed_mul_e2_e4 :
  sed_mul (sed_e 2) (sed_e 4) = sed_e 6.
Proof. close_sed_case. Qed.

Lemma sed_mul_e3_e4 :
  sed_mul (sed_e 3) (sed_e 4) = sed_e 7.
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e6 :
  sed_mul (sed_e 1) (sed_e 6) = sed_scale (-1) (sed_e 7).
Proof. close_sed_case. Qed.

Lemma sed_mul_e2_e9 :
  sed_mul (sed_e 2) (sed_e 9) = sed_e 11.
Proof. close_sed_case. Qed.

Lemma sed_mul_e3_e9 :
  sed_mul (sed_e 3) (sed_e 9) = sed_scale (-1) (sed_e 10).
Proof. close_sed_case. Qed.

Lemma sed_mul_e2_e1 :
  sed_mul (sed_e 2) (sed_e 1) = sed_scale (-1) (sed_e 3).
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e10 :
  sed_mul (sed_e 1) (sed_e 10) = sed_scale (-1) (sed_e 11).
Proof. close_sed_case. Qed.

Lemma sed_mul_e3_e10 :
  sed_mul (sed_e 3) (sed_e 10) = sed_e 9.
Proof. close_sed_case. Qed.

Lemma sed_mul_e2_e11 :
  sed_mul (sed_e 2) (sed_e 11) = sed_scale (-1) (sed_e 9).
Proof. close_sed_case. Qed.

Lemma sed_mul_e3_e1 :
  sed_mul (sed_e 3) (sed_e 1) = sed_e 2.
Proof. close_sed_case. Qed.

Lemma sed_mul_e4_e1 :
  sed_mul (sed_e 4) (sed_e 1) = sed_scale (-1) (sed_e 5).
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e12 :
  sed_mul (sed_e 1) (sed_e 12) = sed_scale (-1) (sed_e 13).
Proof. close_sed_case. Qed.

Lemma sed_mul_e5_e12 :
  sed_mul (sed_e 5) (sed_e 12) = sed_e 9.
Proof. close_sed_case. Qed.

Lemma sed_mul_e4_e13 :
  sed_mul (sed_e 4) (sed_e 13) = sed_scale (-1) (sed_e 9).
Proof. close_sed_case. Qed.

Lemma sed_mul_e5_e1 :
  sed_mul (sed_e 5) (sed_e 1) = sed_e 4.
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e13 :
  sed_mul (sed_e 1) (sed_e 13) = sed_e 12.
Proof. close_sed_case. Qed.

Lemma sed_mul_e5_e13 :
  sed_mul (sed_e 5) (sed_e 13) = sed_scale (-1) (sed_e 8).
Proof. close_sed_case. Qed.

Lemma sed_mul_e6_e1 :
  sed_mul (sed_e 6) (sed_e 1) = sed_e 7.
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e14 :
  sed_mul (sed_e 1) (sed_e 14) = sed_e 15.
Proof. close_sed_case. Qed.

Lemma sed_mul_e7_e14 :
  sed_mul (sed_e 7) (sed_e 14) = sed_scale (-1) (sed_e 9).
Proof. close_sed_case. Qed.

Lemma sed_mul_e6_e15 :
  sed_mul (sed_e 6) (sed_e 15) = sed_e 9.
Proof. close_sed_case. Qed.

Lemma sed_mul_e7_e1 :
  sed_mul (sed_e 7) (sed_e 1) = sed_scale (-1) (sed_e 6).
Proof. close_sed_case. Qed.

Lemma sed_mul_e1_e15 :
  sed_mul (sed_e 1) (sed_e 15) = sed_scale (-1) (sed_e 14).
Proof. close_sed_case. Qed.
