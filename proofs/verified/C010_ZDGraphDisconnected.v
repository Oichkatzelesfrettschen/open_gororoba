(** * C-010: The ZD graph has 7 disconnected components.

    The 42 primitive ZD pairs form a graph where edges connect
    pairs within the same box-kite. The 7 box-kites have distinct
    XOR signatures, proving they are disconnected components.

    Kernel-checked via vm_compute in ZDGraph.v. *)

From Stdlib Require Import List Bool Arith.
From OpenGororoba Require Import BoxKite ZDGraph.
Import ListNotations.

(** All box-kites have uniform internal XOR signatures. *)
Theorem C010_internal_consistency :
  List.forallb
    (fun bk =>
       let sigs := boxkite_xor_sigs bk in
       List.forallb (Nat.eqb (hd 0 sigs)) sigs)
    boxkites = true.
Proof. exact all_boxkites_uniform_xor. Qed.

(** The 7 XOR signatures are all distinct. *)
Theorem C010_distinct_components :
  no_dups boxkite_signatures = true.
Proof. exact signatures_are_distinct. Qed.

(** The signatures themselves. *)
Theorem C010_signature_values :
  boxkite_signatures = [15; 10; 11; 12; 13; 14; 9].
Proof. exact seven_distinct_signatures. Qed.
