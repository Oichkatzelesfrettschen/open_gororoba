(** * F4 Cartan matrix derives from the simple roots.

    F4 is the smallest non-simply-laced exceptional Lie algebra. Its Dynkin
    diagram has two long simple roots, two short simple roots, and a double
    bond connecting alpha_2 (long) to alpha_3 (short).

    Simple roots in R^4 (math normalization, long^2 = 2, short^2 = 1):
        alpha_1 = e_2 - e_3                            (long)
        alpha_2 = e_3 - e_4                            (long)
        alpha_3 = e_4                                  (short)
        alpha_4 = (1/2)(e_1 - e_2 - e_3 - e_4)         (short)

    Asymmetric Cartan formula:
        A_ij = 2 (alpha_i, alpha_j) / (alpha_j, alpha_j)

    Mirrors `crates/gororoba_algebra/src/lie/f4/casimir.rs` (which works with
    the 24 positive roots; the Cartan itself is the standard textbook one). *)

From Stdlib Require Import QArith.
From Stdlib Require Import List.
Import ListNotations.
From Stdlib Require Import Bool.

Open Scope Q_scope.

Definition alpha1 : list Q := [   0  ;   1  ;  - 1 ;   0   ].
Definition alpha2 : list Q := [   0  ;   0  ;    1 ;  - 1   ].
Definition alpha3 : list Q := [   0  ;   0  ;    0 ;    1   ].
Definition alpha4 : list Q := [ 1#2 ; -(1#2) ; -(1#2) ; -(1#2) ].

Definition simple_roots : list (list Q) :=
  [alpha1; alpha2; alpha3; alpha4].

Fixpoint dot (a b : list Q) : Q :=
  match a, b with
  | x :: xs, y :: ys => x * y + dot xs ys
  | _, _             => 0
  end.

(** Asymmetric Cartan derivation -- denominator is `(alpha_j, alpha_j)`. *)
Definition cartan_entry (i j : nat) : Q :=
  match nth_error simple_roots i, nth_error simple_roots j with
  | Some ai, Some aj => 2 * dot ai aj / dot aj aj
  | _, _             => 0
  end.

(** Stated F4 Cartan matrix (Bourbaki). *)
Definition stated (i j : nat) : Q :=
  match i, j with
  | 0%nat, 0%nat =>  2 | 0%nat, 1%nat => -1
  | 1%nat, 0%nat => -1 | 1%nat, 1%nat =>  2 | 1%nat, 2%nat => -2
  | 2%nat, 1%nat => -1 | 2%nat, 2%nat =>  2 | 2%nat, 3%nat => -1
  | 3%nat, 2%nat => -1 | 3%nat, 3%nat =>  2
  | _, _ => 0
  end.

Definition indices : list nat := [0; 1; 2; 3]%nat.

Definition pairs : list (nat * nat) :=
  flat_map (fun i => map (fun j => (i, j)) indices) indices.

Definition entry_matches (ij : nat * nat) : bool :=
  let (i, j) := ij in Qeq_bool (cartan_entry i j) (stated i j).

Definition all_match : bool := forallb entry_matches pairs.

Theorem f4_cartan_derives_from_simple_roots : all_match = true.
Proof. vm_compute. reflexivity. Qed.

(** Long simple roots have squared length 2; short have squared length 1. *)
Theorem f4_root_lengths_are_correct :
  Qeq_bool (dot alpha1 alpha1) 2 = true /\
  Qeq_bool (dot alpha2 alpha2) 2 = true /\
  Qeq_bool (dot alpha3 alpha3) 1 = true /\
  Qeq_bool (dot alpha4 alpha4) 1 = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

(** F4 has a double bond at positions (1, 2): one entry -2, the other -1. *)
Theorem f4_has_double_bond_at_1_2 :
  Qeq_bool (cartan_entry 1 2) (-2) = true /\
  Qeq_bool (cartan_entry 2 1) (-1) = true.
Proof. split; vm_compute; reflexivity. Qed.
