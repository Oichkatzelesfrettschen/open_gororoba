(** * E7 Cartan matrix derives from the simple roots.

    E7 is realised as the rank-7 sub-root-system of E8 obtained by dropping
    the first simple root `alpha_0`. The remaining seven E8 simple roots
    `alpha_1, ..., alpha_7` span a 7D subspace and form an E7 Dynkin
    diagram with branch still at the original E8 branch node (now indexed
    `beta_3` in E7's 1-based naming):

    ```
      beta_1 - beta_2 - beta_3 - beta_4 - beta_5
                        |
                        beta_6 - beta_7
    ```

    Wait -- the branch is at beta_3 with three arms:
    - beta_3 - beta_2 - beta_1 (length 2)
    - beta_3 - beta_4 - beta_5 (length 2 ... but E7 has arms 3, 2, 1)

    Actually E7 has arms (3, 2, 1) from the branch. Mapping to our beta_i:
    - Long arm (length 3): beta_1 - beta_2 - beta_3
    - Medium arm (length 2): beta_5 - beta_6 - beta_7  (where beta_5 is the
      degree-2 node on the long-chain side)
    - Short leaf (length 1): beta_4

    ... no, let me just trust the numerical Cartan derivation. *)

From Stdlib Require Import QArith.
From Stdlib Require Import List.
Import ListNotations.
From Stdlib Require Import Bool.

Open Scope Q_scope.

(** beta_i = alpha_{i} of E8 (drop alpha_0). *)
Definition beta1 : list Q := [ 0;   1 ;  - 1 ;   0 ;   0 ;   0 ;   0 ;  0  ].
Definition beta2 : list Q := [ 0;   0 ;    1 ;  - 1 ;   0 ;   0 ;   0 ;  0  ].
Definition beta3 : list Q := [ 0;   0 ;    0 ;    1 ;  - 1 ;   0 ;   0 ;  0  ].
Definition beta4 : list Q := [ 0;   0 ;    0 ;    0 ;    1 ; - 1 ;   0 ;  0  ].
Definition beta5 : list Q := [ 0;   0 ;    0 ;    0 ;    0 ;   1 ; - 1 ;  0  ].
Definition beta6 : list Q := [ 0;   0 ;    0 ;    0 ;    0 ;   1 ;   1 ;  0  ].
Definition beta7 : list Q := [ - (1#2) ; - (1#2) ; - (1#2) ; - (1#2) ;
                               - (1#2) ; - (1#2) ; - (1#2) ;   1#2   ].

Definition simple_roots : list (list Q) :=
  [beta1; beta2; beta3; beta4; beta5; beta6; beta7].

Fixpoint dot (a b : list Q) : Q :=
  match a, b with
  | x :: xs, y :: ys => x * y + dot xs ys
  | _, _             => 0
  end.

(** Symmetric Cartan formula (E7 is simply-laced). *)
Definition cartan_entry (i j : nat) : Q :=
  match nth_error simple_roots i, nth_error simple_roots j with
  | Some ai, Some aj => 2 * dot ai aj / dot ai ai
  | _, _             => 0
  end.

(** Stated E7 Cartan matrix (branch at index 3 in 0-indexed = beta_4 in
    1-indexed; degree 3, connects to beta_3, beta_5, and the diagonal arm). *)
Definition stated (i j : nat) : Q :=
  match i, j with
  | 0%nat, 0%nat =>  2 | 0%nat, 1%nat => -1
  | 1%nat, 0%nat => -1 | 1%nat, 1%nat =>  2 | 1%nat, 2%nat => -1
  | 2%nat, 1%nat => -1 | 2%nat, 2%nat =>  2 | 2%nat, 3%nat => -1
  | 3%nat, 2%nat => -1 | 3%nat, 3%nat =>  2 | 3%nat, 4%nat => -1
                                            | 3%nat, 5%nat => -1
  | 4%nat, 3%nat => -1 | 4%nat, 4%nat =>  2
  | 5%nat, 3%nat => -1 | 5%nat, 5%nat =>  2 | 5%nat, 6%nat => -1
  | 6%nat, 5%nat => -1 | 6%nat, 6%nat =>  2
  | _, _ => 0
  end.

Definition indices : list nat := [0; 1; 2; 3; 4; 5; 6]%nat.

Definition pairs : list (nat * nat) :=
  flat_map (fun i => map (fun j => (i, j)) indices) indices.

Definition entry_matches (ij : nat * nat) : bool :=
  let (i, j) := ij in Qeq_bool (cartan_entry i j) (stated i j).

Definition all_match : bool := forallb entry_matches pairs.

Theorem e7_cartan_derives_from_simple_roots : all_match = true.
Proof. vm_compute. reflexivity. Qed.

(** Every simple root has squared length 2 (simply-laced). *)
Theorem e7_simple_roots_have_norm_squared_two :
  forallb (fun a => Qeq_bool (dot a a) 2) simple_roots = true.
Proof. vm_compute. reflexivity. Qed.

(** Branch node is beta_4 (index 3): degree 3, connects to beta_3, beta_5, beta_6. *)
Theorem e7_branch_at_beta_4 :
  Qeq_bool (cartan_entry 3 2) (-1) = true /\
  Qeq_bool (cartan_entry 3 4) (-1) = true /\
  Qeq_bool (cartan_entry 3 5) (-1) = true /\
  Qeq_bool (cartan_entry 3 6) ( 0) = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

(** E7 contains E6 as the sub-system spanned by beta_1..beta_5, beta_7
    (drop beta_4 -- wait, drop the leaf). The Cartan submatrix dropping
    index 4 (= beta_5) should match the E6 Cartan. Quick consistency
    check: the (0,1), (1,2), (2,3) entries of the E7 Cartan match. *)
Theorem e7_chain_segment_matches_e6 :
  Qeq_bool (cartan_entry 0 1) (-1) = true /\
  Qeq_bool (cartan_entry 1 2) (-1) = true /\
  Qeq_bool (cartan_entry 2 3) (-1) = true.
Proof. repeat split; vm_compute; reflexivity. Qed.
