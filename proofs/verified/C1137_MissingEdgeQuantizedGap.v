(** * C-1137: Missing-Edge Quantized Topological Gap.

    CLAIM: The 7 missing edges in the sedenion ZD graph (involution
    pairs (i, i XOR 8) for i in 1..7) exhibit a QUANTIZED topological
    gap: the associator norm squared |[e_i, e_k, e_{i XOR 8}]|^2 = 4
    for specific probes e_k.

    This is NOT zero friction (as originally hypothesized) but a
    UNIFORM quantized value of 4 across all 7 involution axes.

    The quantized gap = 2 (norm = sqrt(4) = 2) represents a fundamental
    algebraic constant of the Cayley-Dickson tower at dim=16: the
    last-doubling involution generates a fixed non-associative torque.

    STRATEGY: For each of the 7 missing pairs, we exhibit a specific
    probe and show the associator norm squared = 4 via cbv + ring.

    Mirrors: crates/algebra_experimental/src/majorana_braiding.rs
             (test_missing_edge_protection) *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import SedenionAssociator SedenionGapWitnesses.

Open Scope R_scope.

(** * Quantized gap witnesses for each missing edge.

    Missing edges at dim=16: (i, i XOR 8) for i in 1..7:
    (1,9), (2,10), (3,11), (4,12), (5,13), (6,14), (7,15). *)

(** Missing edge (1, 9): probe e_2 gives |[e1, e2, e9]|^2 = 4. *)
Theorem gap_1_9 :
  sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 9) = 4.
Proof. exact sed_missing_gap_1_9. Qed.

(** Missing edge (2, 10): probe e_1 gives |[e2, e1, e10]|^2 = 4. *)
Theorem gap_2_10 :
  sed_assoc_norm_sq (sed_e 2) (sed_e 1) (sed_e 10) = 4.
Proof. exact sed_missing_gap_2_10. Qed.

(** Missing edge (3, 11): probe e_1 gives |[e3, e1, e11]|^2 = 4. *)
Theorem gap_3_11 :
  sed_assoc_norm_sq (sed_e 3) (sed_e 1) (sed_e 11) = 4.
Proof. exact sed_missing_gap_3_11. Qed.

(** Missing edge (4, 12): probe e_1 gives |[e4, e1, e12]|^2 = 4. *)
Theorem gap_4_12 :
  sed_assoc_norm_sq (sed_e 4) (sed_e 1) (sed_e 12) = 4.
Proof. exact sed_missing_gap_4_12. Qed.

(** Missing edge (5, 13): probe e_1 gives |[e5, e1, e13]|^2 = 4. *)
Theorem gap_5_13 :
  sed_assoc_norm_sq (sed_e 5) (sed_e 1) (sed_e 13) = 4.
Proof. exact sed_missing_gap_5_13. Qed.

(** Missing edge (6, 14): probe e_1 gives |[e6, e1, e14]|^2 = 4. *)
Theorem gap_6_14 :
  sed_assoc_norm_sq (sed_e 6) (sed_e 1) (sed_e 14) = 4.
Proof. exact sed_missing_gap_6_14. Qed.

(** Missing edge (7, 15): probe e_1 gives |[e7, e1, e15]|^2 = 4. *)
Theorem gap_7_15 :
  sed_assoc_norm_sq (sed_e 7) (sed_e 1) (sed_e 15) = 4.
Proof. exact sed_missing_gap_7_15. Qed.

(** * Main theorem: ALL 7 missing edges have quantized gap = 4. *)

Theorem all_missing_edges_quantized_gap :
  sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 9) = 4 /\
  sed_assoc_norm_sq (sed_e 2) (sed_e 1) (sed_e 10) = 4 /\
  sed_assoc_norm_sq (sed_e 3) (sed_e 1) (sed_e 11) = 4 /\
  sed_assoc_norm_sq (sed_e 4) (sed_e 1) (sed_e 12) = 4 /\
  sed_assoc_norm_sq (sed_e 5) (sed_e 1) (sed_e 13) = 4 /\
  sed_assoc_norm_sq (sed_e 6) (sed_e 1) (sed_e 14) = 4 /\
  sed_assoc_norm_sq (sed_e 7) (sed_e 1) (sed_e 15) = 4.
Proof.
  repeat split;
    [exact gap_1_9 | exact gap_2_10 | exact gap_3_11
    | exact gap_4_12 | exact gap_5_13 | exact gap_6_14
    | exact gap_7_15].
Qed.

(** * Corollary: The topological gap is strictly positive on ALL
    involution axes. No missing edge is "protected" (zero friction). *)

Corollary missing_edge_gap_positive :
  forall n : R,
  (n = sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 9) \/
   n = sed_assoc_norm_sq (sed_e 2) (sed_e 1) (sed_e 10) \/
   n = sed_assoc_norm_sq (sed_e 3) (sed_e 1) (sed_e 11) \/
   n = sed_assoc_norm_sq (sed_e 4) (sed_e 1) (sed_e 12) \/
   n = sed_assoc_norm_sq (sed_e 5) (sed_e 1) (sed_e 13) \/
   n = sed_assoc_norm_sq (sed_e 6) (sed_e 1) (sed_e 14) \/
   n = sed_assoc_norm_sq (sed_e 7) (sed_e 1) (sed_e 15)) ->
  n > 0.
Proof.
  intros n H.
  destruct H as [H|[H|[H|[H|[H|[H|H]]]]]]; subst n;
    [rewrite gap_1_9 | rewrite gap_2_10 | rewrite gap_3_11
    | rewrite gap_4_12 | rewrite gap_5_13 | rewrite gap_6_14
    | rewrite gap_7_15]; lra.
Qed.
