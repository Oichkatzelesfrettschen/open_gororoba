(** * C958b_ZDAdjacencyAnalytical: Analytical ZD adjacency generator.

    Mirrors the Rust function zd_graph_adjacency_analytical from
    e8_crystal_bridge.rs. Generates the edge list using the closed-form
    condition: vertices i, j (both not in {0, dim/2}) are connected
    iff N.lxor i j <> dim/2.

    We compute the full edge list at 16D via vm_compute and verify
    the count matches the formula from C958_ZDGraphTopology.v.

    Cross-validated by: test_zd_adjacency_analytical_matches_brute_force_16d *)

From Stdlib Require Import NArith List Bool.
From OpenGororobaVerified Require Import C958_ZDGraphTopology.
Import ListNotations.

Open Scope N_scope.

(** Test whether a vertex is in the giant component (not a singleton). *)
Definition is_giant_vertex (dim v : N) : bool :=
  negb (N.eqb v 0) && negb (N.eqb v (dim / 2)).

(** Test whether two giant-component vertices are connected:
    they are iff their XOR is not dim/2. *)
Definition is_zd_edge (dim i j : N) : bool :=
  is_giant_vertex dim i &&
  is_giant_vertex dim j &&
  negb (N.eqb (N.lxor i j) (dim / 2)).

(** Count edges (i < j) in the analytical adjacency at a given dim.
    We enumerate all pairs and filter. *)
Fixpoint count_edges_from (dim i : N) (fuel : nat) : N :=
  match fuel with
  | O => 0
  | S fuel' =>
    let edges_from_i :=
      (fix inner (j : N) (f2 : nat) : N :=
         match f2 with
         | O => 0
         | S f2' =>
           (if is_zd_edge dim i j then 1 else 0) + inner (j + 1) f2'
         end) (i + 1) (N.to_nat (dim - i - 1))
    in
    edges_from_i + count_edges_from dim (i + 1) fuel'
  end.

Definition analytical_edge_count (dim : N) : N :=
  count_edges_from dim 0 (N.to_nat dim).

(** THEOREM: At dim=16, the analytical generator produces exactly 84 edges.
    This matches zd_edges_at_16 from C958_ZDGraphTopology. *)
Theorem analytical_edges_16 : analytical_edge_count 16 = 84.
Proof. vm_compute. reflexivity. Qed.

(** THEOREM: At dim=32, the analytical generator produces exactly 420 edges. *)
Theorem analytical_edges_32 : analytical_edge_count 32 = 420.
Proof. vm_compute. reflexivity. Qed.
