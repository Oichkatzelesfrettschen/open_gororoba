(** * C-017: XOR bucket condition is necessary for box-kite membership.

    Pairs (i,j) and (k,l) share a box-kite only if
    Nat.lxor i j = Nat.lxor k l.

    Proved by showing all 7 box-kites have uniform internal XOR
    signatures and all 7 signatures are distinct.

    Kernel-checked via vm_compute in ZDGraph.v. *)

From Stdlib Require Import List Bool Arith.
From OpenGororoba Require Import BoxKite ZDGraph.

(** XOR uniformity within each box-kite. *)
Theorem C017_xor_uniform :
  List.forallb
    (fun bk =>
       let sigs := boxkite_xor_sigs bk in
       List.forallb (Nat.eqb (hd 0 sigs)) sigs)
    boxkites = true.
Proof. exact all_boxkites_uniform_xor. Qed.

(** XOR signatures are distinct across box-kites. *)
Theorem C017_xor_necessary :
  no_dups boxkite_signatures = true.
Proof. exact signatures_are_distinct. Qed.
