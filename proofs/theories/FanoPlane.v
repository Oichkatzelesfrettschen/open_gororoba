(** * FanoPlane: The Fano plane PG(2,2) -- 7 points, 7 lines.

    The Fano plane governs octonion multiplication sign rules.
    Its 7 lines correspond to the 7 imaginary octonion triples
    (i,j,k) where e_i * e_j = +/- e_k.

    Properties:
    - 7 points (octonion basis indices 1..7)
    - 7 lines (triples)
    - Every two points lie on exactly one line
    - Every line has exactly 3 points

    Data source: standard Fano plane, matches Baez (2001). *)

From Stdlib Require Import List Arith Bool.
Import ListNotations.

(** The 7 Fano lines as triples of octonion basis indices (1-indexed). *)
Definition fano_lines : list (list nat) :=
  [[1; 2; 3];
   [1; 4; 5];
   [1; 6; 7];
   [2; 4; 6];
   [2; 5; 7];
   [3; 4; 7];
   [3; 5; 6]].

(** There are exactly 7 lines. *)
Theorem fano_line_count : length fano_lines = 7.
Proof. reflexivity. Qed.

(** Each line has exactly 3 points. *)
Theorem fano_line_sizes :
  List.map (@length _) fano_lines = [3; 3; 3; 3; 3; 3; 3].
Proof. reflexivity. Qed.

(** Check if a nat is a member of a list. *)
Fixpoint nat_mem (n : nat) (l : list nat) : bool :=
  match l with
  | [] => false
  | x :: rest => if Nat.eqb n x then true else nat_mem n rest
  end.

(** Count how many lines contain a given point. *)
Definition lines_through (p : nat) : nat :=
  length (List.filter (fun l => nat_mem p l) fano_lines).

(** Every point lies on exactly 3 lines. *)
Theorem fano_point_incidence :
  List.map lines_through [1; 2; 3; 4; 5; 6; 7] = [3; 3; 3; 3; 3; 3; 3].
Proof. vm_compute. reflexivity. Qed.
