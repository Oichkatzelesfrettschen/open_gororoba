(** * C-1140c: Pathion Quantized Gap -- Pairs 11-15.

    Continuation of C-1140a for missing edge pairs 11-15:
    (11,27), (12,28), (13,29), (14,30), (15,31). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import Pathion PathionAssociator CDConjAntimorph CDNegLemmas.

Open Scope R_scope.

Ltac close_pathion_gap assoc_lemma :=
  unfold pathion_assoc_norm_sq, pathion_assoc, pathion_sub, pathion_add, pathion_neg, pathion_norm_sq;
  cbn [pathion_e pathion_mul pathion_lo pathion_hi pathion_zero];
  repeat rewrite sed_conj_zero;
  repeat rewrite sed_mul_zero_left;
  repeat rewrite sed_mul_zero_right;
  repeat rewrite sed_add_zero_left;
  repeat rewrite sed_add_zero_right;
  repeat rewrite sed_neg_zero;
  cbv [sed_norm_sq sed_mul sed_add sed_sub sed_neg sed_conj sed_e sed_zero
       oct_norm_sq oct_mul oct_add oct_sub oct_neg oct_conj oct_e oct_zero
       quat_norm_sq quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       pathion_lo pathion_hi sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  ring_simplify; lra.

(** Missing edge (11, 27): probe e_1 gives |[e11, e1, e27]|^2 = 4. *)
Theorem pathion_gap_11_27 :
  pathion_assoc_norm_sq (pathion_e 11) (pathion_e 1) (pathion_e 27) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (12, 28): probe e_1 gives |[e12, e1, e28]|^2 = 4. *)
Theorem pathion_gap_12_28 :
  pathion_assoc_norm_sq (pathion_e 12) (pathion_e 1) (pathion_e 28) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (13, 29): probe e_1 gives |[e13, e1, e29]|^2 = 4. *)
Theorem pathion_gap_13_29 :
  pathion_assoc_norm_sq (pathion_e 13) (pathion_e 1) (pathion_e 29) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (14, 30): probe e_1 gives |[e14, e1, e30]|^2 = 4. *)
Theorem pathion_gap_14_30 :
  pathion_assoc_norm_sq (pathion_e 14) (pathion_e 1) (pathion_e 30) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (15, 31): probe e_1 gives |[e15, e1, e31]|^2 = 4. *)
Theorem pathion_gap_15_31 :
  pathion_assoc_norm_sq (pathion_e 15) (pathion_e 1) (pathion_e 31) = 4.
Proof using. close_pathion_gap I. Qed.
