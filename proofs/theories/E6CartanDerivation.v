(** * E6 Cartan matrix derives from the simple roots.

    Mirrors the test
      crates/gororoba_algebra/src/lie/e6/root_system.rs::tests::cartan_matrix_derives_from_simple_roots.

    E6 is realised as the rank-6 sub-root-system of E8 obtained by dropping
    the first two simple roots `alpha_0, alpha_1`. The remaining simple roots
    are taken with the same numerical values they have in E8 (consistent
    branch-at-node-4 numbering).

    Encoded over Q so the proof closes by vm_compute. *)

From Stdlib Require Import QArith.
From Stdlib Require Import List.
Import ListNotations.
From Stdlib Require Import Bool.

Open Scope Q_scope.

(** beta_i = alpha_{i+1} of E8 (drop the first two E8 simple roots). *)
Definition beta1 : list Q := [ 0; 0;   1 ;  - 1 ;   0 ;   0 ;   0 ;  0  ].
Definition beta2 : list Q := [ 0; 0;   0 ;    1 ;  - 1 ;   0 ;   0 ;  0  ].
Definition beta3 : list Q := [ 0; 0;   0 ;    0 ;    1 ; - 1 ;   0 ;  0  ].
Definition beta4 : list Q := [ 0; 0;   0 ;    0 ;    0 ;   1 ; - 1 ;  0  ].
Definition beta5 : list Q := [ 0; 0;   0 ;    0 ;    0 ;   1 ;   1 ;  0  ].
Definition beta6 : list Q := [ - (1#2) ; - (1#2) ; - (1#2) ; - (1#2) ;
                               - (1#2) ; - (1#2) ; - (1#2) ;   1#2   ].

Definition simple_roots : list (list Q) :=
  [beta1; beta2; beta3; beta4; beta5; beta6].

Fixpoint dot (a b : list Q) : Q :=
  match a, b with
  | x :: xs, y :: ys => x * y + dot xs ys
  | _, _             => 0
  end.

(** Symmetric Cartan formula (E6 is simply-laced, all roots squared length 2). *)
Definition cartan_entry (i j : nat) : Q :=
  match nth_error simple_roots i, nth_error simple_roots j with
  | Some ai, Some aj => 2 * dot ai aj / dot ai ai
  | _, _             => 0
  end.

(** Stated E6 Cartan matrix from `e6_cartan_matrix()`. Branch at node 2
    (= beta_3 in 1-indexed naming): degree 3, connects to beta_2, beta_4, beta_5. *)
Definition stated (i j : nat) : Q :=
  match i, j with
  | 0%nat, 0%nat =>  2 | 0%nat, 1%nat => -1
  | 1%nat, 0%nat => -1 | 1%nat, 1%nat =>  2 | 1%nat, 2%nat => -1
  | 2%nat, 1%nat => -1 | 2%nat, 2%nat =>  2 | 2%nat, 3%nat => -1
                                            | 2%nat, 4%nat => -1
  | 3%nat, 2%nat => -1 | 3%nat, 3%nat =>  2
  | 4%nat, 2%nat => -1 | 4%nat, 4%nat =>  2 | 4%nat, 5%nat => -1
  | 5%nat, 4%nat => -1 | 5%nat, 5%nat =>  2
  | _, _ => 0
  end.

Definition indices : list nat := [0; 1; 2; 3; 4; 5]%nat.

Definition pairs : list (nat * nat) :=
  flat_map (fun i => map (fun j => (i, j)) indices) indices.

Definition entry_matches (ij : nat * nat) : bool :=
  let (i, j) := ij in Qeq_bool (cartan_entry i j) (stated i j).

Definition all_match : bool := forallb entry_matches pairs.

Theorem e6_cartan_derives_from_simple_roots : all_match = true.
Proof. vm_compute. reflexivity. Qed.

(** Every simple root has squared length 2 (simply-laced). *)
Theorem e6_simple_roots_have_norm_squared_two :
  forallb (fun a => Qeq_bool (dot a a) 2) simple_roots = true.
Proof. vm_compute. reflexivity. Qed.

(** beta_3 (the branch node, index 2) has degree 3: connects to beta_2, beta_4, beta_5. *)
Theorem e6_branch_node_has_degree_three :
  Qeq_bool (cartan_entry 2 1) (-1) = true /\
  Qeq_bool (cartan_entry 2 3) (-1) = true /\
  Qeq_bool (cartan_entry 2 4) (-1) = true /\
  Qeq_bool (cartan_entry 2 5) ( 0) = true.
Proof. repeat split; vm_compute; reflexivity. Qed.
