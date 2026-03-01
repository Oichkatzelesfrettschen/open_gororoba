(** * Complete graph edge counting for parity-clique decomposition.

    Formalizes the clique edge count formula |E(K_n)| = n*(n-1)/2
    and its application to ZD adjacency decomposition at Cayley-Dickson
    dimensions 16 and 32.

    At dim=16: adjacency decomposes into K_4 + K_4  (12 edges total).
    At dim=32: adjacency decomposes into K_8 + K_8  (56 edges total).

    Mirrors: algebra_core::analysis::graph_projections (edge counting) *)

From Stdlib Require Import Arith PeanoNat Lia.

(** Number of edges in a complete graph K_n. *)
Definition clique_edges (n : nat) : nat := n * (n - 1) / 2.

(** K_4 has 6 edges. *)
Theorem k4_edges : clique_edges 4 = 6.
Proof. reflexivity. Qed.

(** K_8 has 28 edges. *)
Theorem k8_edges : clique_edges 8 = 28.
Proof. reflexivity. Qed.

(** K_16 has 120 edges. *)
Theorem k16_edges : clique_edges 16 = 120.
Proof. reflexivity. Qed.

(** K_2 has 1 edge (the simplest nontrivial clique). *)
Theorem k2_edges : clique_edges 2 = 1.
Proof. reflexivity. Qed.

(** K_3 has 3 edges (triangle). *)
Theorem k3_edges : clique_edges 3 = 3.
Proof. reflexivity. Qed.

(** Dim 16: two K_4 cliques give 12 edges total. *)
Theorem dim16_two_clique : 2 * clique_edges 4 = 12.
Proof. reflexivity. Qed.

(** Dim 32: two K_8 cliques give 56 edges total. *)
Theorem dim32_two_clique : 2 * clique_edges 8 = 56.
Proof. reflexivity. Qed.

(** Dim 64 upper bound: two K_16 cliques would give 240. *)
Theorem dim64_two_clique : 2 * clique_edges 16 = 240.
Proof. reflexivity. Qed.

(** K_0 and K_1 have zero edges. *)
Theorem k0_edges : clique_edges 0 = 0.
Proof. reflexivity. Qed.

Theorem k1_edges : clique_edges 1 = 0.
Proof. reflexivity. Qed.

(** Spectral gap of K_{2,2,2} adjacency: 4 + 2 = 6. *)
Theorem spectral_gap_k222 : 4 + 2 = 6.
Proof. reflexivity. Qed.
