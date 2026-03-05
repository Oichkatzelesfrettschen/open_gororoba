(** * ZDGraph: Zero-divisor graph structure and XOR bucket condition.

    The 42 primitive ZD pairs form a disconnected graph with 7 components
    (box-kites). The XOR bucket condition governs which pairs can share
    a box-kite: pairs (i,j) and (k,l) are in the same box-kite iff
    Nat.lxor i j = Nat.lxor k l.

    Key results:
    - 7 disconnected components (one per box-kite)
    - XOR bucket is necessary for same-component membership

    Data source: crates/algebra_analysis/src/boxkites.rs *)

From Stdlib Require Import List Arith Bool.
From OpenGororoba Require Import BoxKite.
Import ListNotations.

(** XOR bucket compatibility: two pairs share a box-kite iff
    their XOR signatures match. *)
Definition xor_sig (p : nat * nat) : nat :=
  Nat.lxor (fst p) (snd p).

(** Compute the XOR signature of each pair in a box-kite. *)
Definition boxkite_xor_sigs (bk : list (nat * nat)) : list nat :=
  List.map xor_sig bk.

(** All pairs within boxkite_1 share the same XOR signature. *)
Theorem bk1_uniform_xor :
  let sigs := boxkite_xor_sigs boxkite_1 in
  List.forallb (Nat.eqb (hd 0 sigs)) sigs = true.
Proof. vm_compute. reflexivity. Qed.

(** All pairs within boxkite_2 share the same XOR signature. *)
Theorem bk2_uniform_xor :
  let sigs := boxkite_xor_sigs boxkite_2 in
  List.forallb (Nat.eqb (hd 0 sigs)) sigs = true.
Proof. vm_compute. reflexivity. Qed.

(** All 7 box-kites have uniform internal XOR signatures. *)
Theorem all_boxkites_uniform_xor :
  List.forallb
    (fun bk =>
       let sigs := boxkite_xor_sigs bk in
       List.forallb (Nat.eqb (hd 0 sigs)) sigs)
    boxkites = true.
Proof. vm_compute. reflexivity. Qed.

(** The 7 box-kite XOR signatures are all distinct. *)
Definition boxkite_signatures : list nat :=
  List.map (fun bk => xor_sig (hd (0,0) bk)) boxkites.

Theorem seven_distinct_signatures :
  boxkite_signatures = [15; 10; 11; 12; 13; 14; 9].
Proof. vm_compute. reflexivity. Qed.

(** Helper: check nat membership in a list. *)
Fixpoint nat_mem (n : nat) (l : list nat) : bool :=
  match l with
  | [] => false
  | x :: rest => if Nat.eqb n x then true else nat_mem n rest
  end.

(** No duplicates in the signature list (all 7 are distinct). *)
Fixpoint no_dups (l : list nat) : bool :=
  match l with
  | [] => true
  | x :: rest => negb (nat_mem x rest) && no_dups rest
  end.

Theorem signatures_are_distinct :
  no_dups boxkite_signatures = true.
Proof. vm_compute. reflexivity. Qed.
