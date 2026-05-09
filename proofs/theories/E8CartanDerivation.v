(** * E8 Cartan matrix derives from the simple roots.

    Mirrors the test
      crates/gororoba_algebra/src/lie/e8/root_system.rs::tests::cartan_matrix_derives_from_simple_roots
    by verifying, over exact rationals, that the 8x8 Cartan matrix returned
    by `e8_cartan_matrix()` equals the one obtained from `e8_simple_roots()`
    via
        A_ij = 2 (alpha_i, alpha_j) / (alpha_i, alpha_i).

    The Dynkin diagram numbering puts the branch at node 4:

      alpha_0 - alpha_1 - alpha_2 - alpha_3 - alpha_4 - alpha_5
                                              |
                                              alpha_6 - alpha_7

    Encoded over Q (rationals) so we can close the proof by vm_compute. *)

From Stdlib Require Import QArith.
From Stdlib Require Import List.
Import ListNotations.
From Stdlib Require Import Bool.

Open Scope Q_scope.

(** Eight simple roots, branch at node 4 (matches Rust code verbatim). *)
Definition alpha0 : list Q := [ 1 ;  - 1 ;   0 ;   0 ;   0 ;   0 ;   0 ;  0  ].
Definition alpha1 : list Q := [ 0 ;    1 ; - 1 ;   0 ;   0 ;   0 ;   0 ;  0  ].
Definition alpha2 : list Q := [ 0 ;    0 ;   1 ; - 1 ;   0 ;   0 ;   0 ;  0  ].
Definition alpha3 : list Q := [ 0 ;    0 ;   0 ;   1 ; - 1 ;   0 ;   0 ;  0  ].
Definition alpha4 : list Q := [ 0 ;    0 ;   0 ;   0 ;   1 ; - 1 ;   0 ;  0  ].
Definition alpha5 : list Q := [ 0 ;    0 ;   0 ;   0 ;   0 ;   1 ; - 1 ;  0  ].
Definition alpha6 : list Q := [ 0 ;    0 ;   0 ;   0 ;   0 ;   1 ;   1 ;  0  ].
Definition alpha7 : list Q := [ - (1#2) ; - (1#2) ; - (1#2) ; - (1#2) ;
                                - (1#2) ; - (1#2) ; - (1#2) ;   1#2   ].

Definition simple_roots : list (list Q) :=
  [alpha0; alpha1; alpha2; alpha3; alpha4; alpha5; alpha6; alpha7].

(** Euclidean inner product on equal-length Q-lists. *)
Fixpoint dot (a b : list Q) : Q :=
  match a, b with
  | x :: xs, y :: ys => x * y + dot xs ys
  | _, _             => 0
  end.

(** Cartan derivation `A_ij = 2 (a_i, a_j) / (a_i, a_i)`. *)
Definition cartan_entry (i j : nat) : Q :=
  match nth_error simple_roots i, nth_error simple_roots j with
  | Some ai, Some aj => 2 * dot ai aj / dot ai ai
  | _, _             => 0
  end.

(** Hardcoded 8x8 Cartan matrix from `e8_cartan_matrix()`. *)
Definition stated (i j : nat) : Q :=
  match i, j with
  | 0%nat, 0%nat =>  2 | 0%nat, 1%nat => -1
  | 1%nat, 0%nat => -1 | 1%nat, 1%nat =>  2 | 1%nat, 2%nat => -1
  | 2%nat, 1%nat => -1 | 2%nat, 2%nat =>  2 | 2%nat, 3%nat => -1
  | 3%nat, 2%nat => -1 | 3%nat, 3%nat =>  2 | 3%nat, 4%nat => -1
  | 4%nat, 3%nat => -1 | 4%nat, 4%nat =>  2 | 4%nat, 5%nat => -1
                                            | 4%nat, 6%nat => -1
  | 5%nat, 4%nat => -1 | 5%nat, 5%nat =>  2
  | 6%nat, 4%nat => -1 | 6%nat, 6%nat =>  2 | 6%nat, 7%nat => -1
  | 7%nat, 6%nat => -1 | 7%nat, 7%nat =>  2
  | _, _ => 0
  end.

(** Cartesian product `[0..8) x [0..8)`. *)
Definition indices : list nat := [0; 1; 2; 3; 4; 5; 6; 7]%nat.

Definition pairs : list (nat * nat) :=
  flat_map (fun i => map (fun j => (i, j)) indices) indices.

Definition entry_matches (ij : nat * nat) : bool :=
  let (i, j) := ij in Qeq_bool (cartan_entry i j) (stated i j).

Definition all_match : bool := forallb entry_matches pairs.

(** Main theorem: every entry of the derived Cartan matrix matches the
    hardcoded `e8_cartan_matrix()`. Closed by exact rational arithmetic
    in vm_compute (avoids float instabilities of the Rust regression). *)
Theorem e8_cartan_derives_from_simple_roots : all_match = true.
Proof. vm_compute. reflexivity. Qed.

(** Sanity: each simple root has squared length 2. *)
Theorem simple_roots_have_norm_squared_two :
  forallb (fun a => Qeq_bool (dot a a) 2) simple_roots = true.
Proof. vm_compute. reflexivity. Qed.

(** The branch node alpha_4 has degree 3 (connects to alpha_3, alpha_5, alpha_6). *)
Theorem alpha4_is_branch_node :
  Qeq_bool (cartan_entry 4 3) (-1) = true /\
  Qeq_bool (cartan_entry 4 5) (-1) = true /\
  Qeq_bool (cartan_entry 4 6) (-1) = true /\
  Qeq_bool (cartan_entry 4 7) ( 0) = true.
Proof.
  repeat split; vm_compute; reflexivity.
Qed.
