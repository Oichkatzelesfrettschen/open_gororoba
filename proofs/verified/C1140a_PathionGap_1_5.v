(** * C-1140a: Pathion Quantized Gap -- Pairs 1-5.

    The 15 missing edges in the pathion ZD graph (involution pairs
    (i, i XOR 16) for i in 1..15) exhibit a QUANTIZED topological gap.

    This file covers pairs 1-5: (1,17), (2,18), (3,19), (4,20), (5,21).

    For each pair, we exhibit a specific probe and show
    pathion_assoc_norm_sq = 4 via cbv + ring_simplify + lra. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import Pathion PathionAssociator.

Open Scope R_scope.

(** Missing edge (1, 17): probe e_2 gives |[e1, e2, e17]|^2 = 4. *)
Theorem pathion_gap_1_17 :
  pathion_assoc_norm_sq (pathion_e 1) (pathion_e 2) (pathion_e 17) = 4.
Proof using. pathion_norm_sq_4. Qed.

(** Missing edge (2, 18): probe e_1 gives |[e2, e1, e18]|^2 = 4. *)
Theorem pathion_gap_2_18 :
  pathion_assoc_norm_sq (pathion_e 2) (pathion_e 1) (pathion_e 18) = 4.
Proof using. pathion_norm_sq_4. Qed.

(** Missing edge (3, 19): probe e_1 gives |[e3, e1, e19]|^2 = 4. *)
Theorem pathion_gap_3_19 :
  pathion_assoc_norm_sq (pathion_e 3) (pathion_e 1) (pathion_e 19) = 4.
Proof using. pathion_norm_sq_4. Qed.

(** Missing edge (4, 20): probe e_1 gives |[e4, e1, e20]|^2 = 4. *)
Theorem pathion_gap_4_20 :
  pathion_assoc_norm_sq (pathion_e 4) (pathion_e 1) (pathion_e 20) = 4.
Proof using. pathion_norm_sq_4. Qed.

(** Missing edge (5, 21): probe e_1 gives |[e5, e1, e21]|^2 = 4. *)
Theorem pathion_gap_5_21 :
  pathion_assoc_norm_sq (pathion_e 5) (pathion_e 1) (pathion_e 21) = 4.
Proof using. pathion_norm_sq_4. Qed.
