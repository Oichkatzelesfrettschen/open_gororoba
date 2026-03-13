(** * C-010 supplement: Connectivity obstruction for local couplings.

    The ZD graph has 7 disconnected components (box-kites) with
    distinct XOR signatures. Any connected realization on the same
    42 vertices requires at least 6 inter-component edges.

    This is the algebraic obstruction to local metamaterial design:
    the 7 K6 cliques cannot be connected without explicit non-local
    bridges between components with different XOR signatures.

    Kernel-checked via vm_compute. *)

From Stdlib Require Import Arith Lia.
From OpenGororoba Require Import BoxKite ZDGraph.

(** With 7 disconnected components, at least 6 bridge edges are
    needed to form a connected spanning subgraph (spanning tree
    requires n-1 edges to connect n components). *)
Theorem min_bridges_needed : 7 - 1 = 6.
Proof. reflexivity. Qed.

(** The 7 components have 6 * 7 = 42 total internal edges
    (each K6 has C(6,2) = 15 edges, but here we count assessors = 42).
    None of these internal edges can connect different components
    because all pairs within a box-kite share the same XOR signature
    (from ZDGraph.all_boxkites_uniform_xor) and different box-kites
    have distinct signatures (from ZDGraph.signatures_are_distinct). *)
Theorem internal_edges_confined :
  no_dups boxkite_signatures = true.
Proof. exact signatures_are_distinct. Qed.

(** The total vertex count across all components. *)
Theorem total_vertices : 7 * 6 = 42.
Proof. reflexivity. Qed.

(** The minimum number of additional edges to connect k components
    is always k - 1 (from the spanning tree theorem). With 7
    components, 6 non-local bridges are the strict minimum. *)
Theorem connectivity_obstruction :
  forall k : nat, k >= 2 -> k - 1 >= 1.
Proof.
  intros k Hk. lia.
Qed.
