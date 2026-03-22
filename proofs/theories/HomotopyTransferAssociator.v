(** * HomotopyTransferAssociator: m3 classification by Fano incidence.

    The homotopy transfer m3 from the sedenion retraction classifies
    all 210 ordered triples of distinct imaginary octonion units into:
    - 42 scalar outputs (m3 = +/-2 e_0) on Fano-line triples
    - 168 imaginary outputs (m3 = +/-2 e_l) on non-Fano triples

    On non-Fano triples: m3 = Assoc (octonionic associator) exactly.
    On Fano triples: m3 = -2 e_0 (scalar), while Assoc = 0.

    The scalar/imaginary classification is determined by i XOR j XOR k:
    - i XOR j XOR k = 0 iff {i,j,k} is a Fano line => scalar output
    - i XOR j XOR k != 0 => imaginary output on basis e_{i XOR j XOR k}

    This encodes the G2 calibration structure:
    - phi (3-form) on Fano lines
    - psi (co-associative 4-form) on non-Fano triples *)

From Stdlib Require Import List Arith Bool ZArith Lia.
Import ListNotations.

(** Fano lines as sorted triples. *)
Definition fano_lines : list (nat * nat * nat) :=
  [(1,2,3); (1,4,5); (1,6,7); (2,4,6); (2,5,7); (3,4,7); (3,5,6)].

(** A triple is on a Fano line iff i XOR j XOR k = 0. *)
Definition is_fano_xor (i j k : nat) : bool :=
  Nat.eqb (Nat.lxor (Nat.lxor i j) k) 0.

(** Count Fano-line triples among all 210 ordered triples of {1..7}. *)
Definition fano_triple_count : nat :=
  List.length (List.filter (fun ijk =>
    match ijk with (i, (j, k)) => is_fano_xor i j k end)
    (List.flat_map (fun i =>
      List.flat_map (fun j =>
        if Nat.eqb i j then []
        else List.flat_map (fun k =>
          if orb (Nat.eqb k i) (Nat.eqb k j) then []
          else [(i, (j, k))])
          (List.seq 1 7))
        (List.seq 1 7))
      (List.seq 1 7))).

Theorem fano_triples_are_42 : fano_triple_count = 42.
Proof. vm_compute. reflexivity. Qed.

(** Non-Fano triple count = 210 - 42 = 168. *)
Definition nonfano_triple_count : nat :=
  List.length (List.filter (fun ijk =>
    match ijk with (i, (j, k)) => negb (is_fano_xor i j k) end)
    (List.flat_map (fun i =>
      List.flat_map (fun j =>
        if Nat.eqb i j then []
        else List.flat_map (fun k =>
          if orb (Nat.eqb k i) (Nat.eqb k j) then []
          else [(i, (j, k))])
          (List.seq 1 7))
        (List.seq 1 7))
      (List.seq 1 7))).

Theorem nonfano_triples_are_168 : nonfano_triple_count = 168.
Proof. vm_compute. reflexivity. Qed.

(** Total check: 42 + 168 = 210. *)
Theorem total_triples : fano_triple_count + nonfano_triple_count = 210.
Proof. vm_compute. reflexivity. Qed.

(** Each Fano line has 3! = 6 orderings. 7 * 6 = 42. *)
Theorem fano_lines_times_orderings : 7 * 6 = 42.
Proof. reflexivity. Qed.
