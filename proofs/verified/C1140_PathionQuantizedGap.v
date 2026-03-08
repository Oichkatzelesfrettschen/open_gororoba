(** * C-1140: Pathion Quantized Topological Gap (All 15 Pairs).

    CLAIM: ALL 15 missing edges in the dim=32 pathion ZD graph
    (involution pairs (i, i XOR 16) for i in 1..15) exhibit a
    QUANTIZED topological gap: |[e_i, e_k, e_{i XOR 16}]|^2 = 4.

    This extends C-1137 (dim=16, 7 pairs) to dim=32 (15 pairs),
    providing the SECOND concrete data point for the universal
    quantized gap conjecture across the Cayley-Dickson tower.

    The conjunction imports individual proofs from C1140a/b/c files
    (split for parallel compilation). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import Pathion PathionAssociator.
From OpenGororobaVerified Require Import
  C1140a_PathionGap_1_5
  C1140b_PathionGap_6_10
  C1140c_PathionGap_11_15.

Open Scope R_scope.

(** * Main theorem: ALL 15 missing edges have quantized gap = 4. *)

Theorem pathion_all_missing_edges_quantized_gap :
  pathion_assoc_norm_sq (pathion_e 1)  (pathion_e 2) (pathion_e 17) = 4 /\
  pathion_assoc_norm_sq (pathion_e 2)  (pathion_e 1) (pathion_e 18) = 4 /\
  pathion_assoc_norm_sq (pathion_e 3)  (pathion_e 1) (pathion_e 19) = 4 /\
  pathion_assoc_norm_sq (pathion_e 4)  (pathion_e 1) (pathion_e 20) = 4 /\
  pathion_assoc_norm_sq (pathion_e 5)  (pathion_e 1) (pathion_e 21) = 4 /\
  pathion_assoc_norm_sq (pathion_e 6)  (pathion_e 1) (pathion_e 22) = 4 /\
  pathion_assoc_norm_sq (pathion_e 7)  (pathion_e 1) (pathion_e 23) = 4 /\
  pathion_assoc_norm_sq (pathion_e 8)  (pathion_e 1) (pathion_e 24) = 4 /\
  pathion_assoc_norm_sq (pathion_e 9)  (pathion_e 1) (pathion_e 25) = 4 /\
  pathion_assoc_norm_sq (pathion_e 10) (pathion_e 1) (pathion_e 26) = 4 /\
  pathion_assoc_norm_sq (pathion_e 11) (pathion_e 1) (pathion_e 27) = 4 /\
  pathion_assoc_norm_sq (pathion_e 12) (pathion_e 1) (pathion_e 28) = 4 /\
  pathion_assoc_norm_sq (pathion_e 13) (pathion_e 1) (pathion_e 29) = 4 /\
  pathion_assoc_norm_sq (pathion_e 14) (pathion_e 1) (pathion_e 30) = 4 /\
  pathion_assoc_norm_sq (pathion_e 15) (pathion_e 1) (pathion_e 31) = 4.
Proof.
  repeat split;
    [ exact pathion_gap_1_17
    | exact pathion_gap_2_18
    | exact pathion_gap_3_19
    | exact pathion_gap_4_20
    | exact pathion_gap_5_21
    | exact pathion_gap_6_22
    | exact pathion_gap_7_23
    | exact pathion_gap_8_24
    | exact pathion_gap_9_25
    | exact pathion_gap_10_26
    | exact pathion_gap_11_27
    | exact pathion_gap_12_28
    | exact pathion_gap_13_29
    | exact pathion_gap_14_30
    | exact pathion_gap_15_31 ].
Qed.

(** * Corollary: The topological gap is strictly positive on ALL
    involution axes at dim=32. *)

Corollary pathion_missing_edge_gap_positive :
  forall n : R,
  (n = pathion_assoc_norm_sq (pathion_e 1)  (pathion_e 2) (pathion_e 17) \/
   n = pathion_assoc_norm_sq (pathion_e 2)  (pathion_e 1) (pathion_e 18) \/
   n = pathion_assoc_norm_sq (pathion_e 3)  (pathion_e 1) (pathion_e 19) \/
   n = pathion_assoc_norm_sq (pathion_e 4)  (pathion_e 1) (pathion_e 20) \/
   n = pathion_assoc_norm_sq (pathion_e 5)  (pathion_e 1) (pathion_e 21) \/
   n = pathion_assoc_norm_sq (pathion_e 6)  (pathion_e 1) (pathion_e 22) \/
   n = pathion_assoc_norm_sq (pathion_e 7)  (pathion_e 1) (pathion_e 23) \/
   n = pathion_assoc_norm_sq (pathion_e 8)  (pathion_e 1) (pathion_e 24) \/
   n = pathion_assoc_norm_sq (pathion_e 9)  (pathion_e 1) (pathion_e 25) \/
   n = pathion_assoc_norm_sq (pathion_e 10) (pathion_e 1) (pathion_e 26) \/
   n = pathion_assoc_norm_sq (pathion_e 11) (pathion_e 1) (pathion_e 27) \/
   n = pathion_assoc_norm_sq (pathion_e 12) (pathion_e 1) (pathion_e 28) \/
   n = pathion_assoc_norm_sq (pathion_e 13) (pathion_e 1) (pathion_e 29) \/
   n = pathion_assoc_norm_sq (pathion_e 14) (pathion_e 1) (pathion_e 30) \/
   n = pathion_assoc_norm_sq (pathion_e 15) (pathion_e 1) (pathion_e 31)) ->
  n > 0.
Proof.
  intros n H.
  destruct H as [H|[H|[H|[H|[H|[H|[H|[H|[H|[H|[H|[H|[H|[H|H]]]]]]]]]]]]]];
    subst n;
    [ rewrite pathion_gap_1_17
    | rewrite pathion_gap_2_18
    | rewrite pathion_gap_3_19
    | rewrite pathion_gap_4_20
    | rewrite pathion_gap_5_21
    | rewrite pathion_gap_6_22
    | rewrite pathion_gap_7_23
    | rewrite pathion_gap_8_24
    | rewrite pathion_gap_9_25
    | rewrite pathion_gap_10_26
    | rewrite pathion_gap_11_27
    | rewrite pathion_gap_12_28
    | rewrite pathion_gap_13_29
    | rewrite pathion_gap_14_30
    | rewrite pathion_gap_15_31 ]; lra.
Qed.
