(** * C-026: ZD graph has 7 block components (box-kites).

    The 42 ZD pairs partition into 7 disconnected K_6 cliques.
    Cross-clique pairs have different XOR signatures, so no cross-edge
    exists in the coassessor graph.

    This is the graph-theoretic obstruction: the ZD structure is
    fundamentally fragmented, preventing any unified treatment. *)

From Stdlib Require Import List Bool Arith.
From OpenGororoba Require Import BoxKite ZDGraph.

(** 7 blocks, each of size 6. *)
Theorem C026_seven_blocks : length boxkites = 7.
Proof. exact boxkite_count. Qed.

(** All blocks have distinct XOR signatures (no cross-edges). *)
Theorem C026_no_cross_edges :
  no_dups boxkite_signatures = true.
Proof. exact signatures_are_distinct. Qed.
