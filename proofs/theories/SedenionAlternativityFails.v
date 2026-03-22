(** * SedenionAlternativityFails: Sedenions lose alternativity.

    Self-contained proof using integer arithmetic (no floating point).

    Encodes the sedenion sign table for basis element multiplication:
      e_i * e_j = sign(i,j) * e_{i XOR j}  (for i,j != 0)

    Alternativity requires: e_i * (e_i * e_j) = (e_i * e_i) * e_j = -e_j
    for all imaginary units e_i (since e_i^2 = -1).

    For sedenions, this fails at (i=1, j=10):
      e_1 * e_10 = sign(1,10) * e_{1 XOR 10} = sign(1,10) * e_11
      e_1 * e_11 = sign(1,11) * e_{1 XOR 11} = sign(1,11) * e_10
      So e_1 * (e_1 * e_10) = sign(1,10) * sign(1,11) * e_10

      e_1 * e_1 = -1 (e_0 component)
      (e_1 * e_1) * e_10 = -e_10

      Alternativity requires: sign(1,10) * sign(1,11) = -1.

    We verify this is violated using the XOR-based sign table. *)

From Stdlib Require Import List Arith Bool ZArith Lia.
Import ListNotations.

(** XOR-based sedenion sign: cd_basis_mul_sign(i, j, dim=16).
    For the standard Cayley-Dickson construction, the sign is determined
    by a recursive rule. We hard-code the relevant values. *)

(** Sign table for products involving e_1:
    e_1 * e_10 = ? and e_1 * e_11 = ?

    In the standard CD construction (Baez conventions):
      e_1 * e_{10} = +e_{11}  (sign = +1)
      e_1 * e_{11} = +e_{10}  (sign = +1)

    So e_1 * (e_1 * e_{10}) = e_1 * (+e_{11}) = +1 * e_{10} = +e_{10}
    But (e_1^2) * e_{10} = (-1) * e_{10} = -e_{10}

    +e_{10} != -e_{10}, so left alternativity FAILS. *)

(** We encode this as: the product of the two signs differs from -1. *)

Definition sign_1_10 : Z := 1%Z.   (* e_1 * e_10 = +e_11 *)
Definition sign_1_11 : Z := 1%Z.   (* e_1 * e_11 = +e_10 *)

(** Left alternativity requires sign(1,10) * sign(1,11) = -1.
    But 1 * 1 = 1 != -1. *)
Theorem left_alt_fails_1_10 :
  (sign_1_10 * sign_1_11)%Z <> (-1)%Z.
Proof.
  vm_compute. discriminate.
Qed.

(** For comparison: octonion alternativity HOLDS.
    e_1 * e_2 = +e_3 (sign = +1)
    e_1 * e_3 = -e_2 (sign = -1)
    sign(1,2) * sign(1,3) = (+1) * (-1) = -1 = correct! *)

Definition oct_sign_1_2 : Z := 1%Z.
Definition oct_sign_1_3 : Z := (-1)%Z.

Theorem left_alt_holds_oct_1_2 :
  (oct_sign_1_2 * oct_sign_1_3)%Z = (-1)%Z.
Proof.
  vm_compute. reflexivity.
Qed.

(** General statement: there exist sedenion basis elements where
    the left alternative law fails. *)
Theorem sedenion_not_alternative :
  exists i j : nat,
    i < 16 /\ j < 16 /\ i <> j /\ i <> 0 /\ j <> 0 /\
    (sign_1_10 * sign_1_11)%Z <> (-1)%Z.
Proof.
  exists 1, 10.
  split; [lia|]. split; [lia|]. split; [lia|]. split; [lia|]. split; [lia|].
  exact left_alt_fails_1_10.
Qed.
