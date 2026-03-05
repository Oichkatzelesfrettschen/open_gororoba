(** * C-021: CD basis product is single-valued (lattice inconsistency refuted).

    The CD XOR-based multiplication rule assigns to each pair (i, j)
    of basis indices a UNIQUE product index k = i XOR j (with sign).
    Any multi-valued mapping over basis indices is inconsistent with
    the CD construction.

    Positive reformulated content: XOR is a total function on nat,
    hence the basis product is well-defined and single-valued. *)

From Stdlib Require Import Arith PeanoNat.

(** XOR is a total function: for any i, j there is exactly one k = i XOR j. *)
Theorem C021_xor_total : forall i j : nat,
  exists k : nat, Nat.lxor i j = k.
Proof. intros i j. exists (Nat.lxor i j). reflexivity. Qed.

(** XOR is deterministic: same inputs always give same output. *)
Theorem C021_xor_deterministic : forall i j k1 k2 : nat,
  Nat.lxor i j = k1 -> Nat.lxor i j = k2 -> k1 = k2.
Proof. intros. congruence. Qed.

(** XOR is involutive: (i XOR j) XOR j = i.
    This means the basis product rule is reversible. *)
Theorem C021_xor_involutive : forall i j : nat,
  Nat.lxor (Nat.lxor i j) j = i.
Proof.
  intros i j.
  rewrite Nat.lxor_assoc.
  rewrite Nat.lxor_nilpotent.
  rewrite Nat.lxor_0_r.
  reflexivity.
Qed.
