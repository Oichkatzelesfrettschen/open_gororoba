(** * C-1140b: Pathion Quantized Gap -- Pairs 6-10.

    Continuation of C-1140a for missing edge pairs 6-10:
    (6,22), (7,23), (8,24), (9,25), (10,26). *)

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

(** Missing edge (6, 22): probe e_1 gives |[e6, e1, e22]|^2 = 4. *)
Theorem pathion_gap_6_22 :
  pathion_assoc_norm_sq (pathion_e 6) (pathion_e 1) (pathion_e 22) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (7, 23): probe e_1 gives |[e7, e1, e23]|^2 = 4. *)
Theorem pathion_gap_7_23 :
  pathion_assoc_norm_sq (pathion_e 7) (pathion_e 1) (pathion_e 23) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (8, 24): probe e_1 gives |[e8, e1, e24]|^2 = 4. *)
Theorem pathion_gap_8_24 :
  pathion_assoc_norm_sq (pathion_e 8) (pathion_e 1) (pathion_e 24) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (9, 25): probe e_1 gives |[e9, e1, e25]|^2 = 4. *)
Theorem pathion_gap_9_25 :
  pathion_assoc_norm_sq (pathion_e 9) (pathion_e 1) (pathion_e 25) = 4.
Proof using. close_pathion_gap I. Qed.

(** Missing edge (10, 26): probe e_1 gives |[e10, e1, e26]|^2 = 4. *)
Theorem pathion_gap_10_26 :
  pathion_assoc_norm_sq (pathion_e 10) (pathion_e 1) (pathion_e 26) = 4.
Proof using. close_pathion_gap I. Qed.
