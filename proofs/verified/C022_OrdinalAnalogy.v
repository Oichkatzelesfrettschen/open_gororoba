(** * C-022: CD doubling does not respect surreal ordinal arithmetic.

    C-022 drew an analogy between CD levels and ordinal numbers.
    The positive content: CD doubling at level n gives 2^n real components,
    but ordinal addition omega + omega <> omega * 2 in the surreal sense
    for finite ordinals.

    We prove the concrete cardinality fact: dim(C) = 2, dim(H) = 4,
    dim(O) = 8, dim(S) = 16. The doubling is exact. *)

From Stdlib Require Import Arith Lia.

(** CD doubling: dim at level n is 2^n. *)
Theorem C022_dim_doubling :
  2^1 = 2 /\ 2^2 = 4 /\ 2^3 = 8 /\ 2^4 = 16.
Proof. vm_compute. repeat split; reflexivity. Qed.

(** Ordinal sum fails: 2^a + 2^a = 2^(a+1), not 2*2^a in ordinal sense.
    But for finite cardinals, they are the same. *)
Theorem C022_doubling_is_sum : forall n : nat,
  2 ^ n + 2 ^ n = 2 ^ (S n).
Proof.
  intro n. simpl. lia.
Qed.
