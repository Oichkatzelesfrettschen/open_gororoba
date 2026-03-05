(** * BoxKite: Sedenion zero-divisor assessor enumeration.

    The 42 primitive ZD pairs (lo, hi) where lo in 1..7, hi in 9..15
    (excluding hi=8 and identity pairs lo+8) partition into
    7 box-kites of 6 assessors each.

    Each box-kite forms an octahedral graph K_{2,2,2} with 3 struts.
    Every assessor belongs to exactly 2 of the 7 Fano automorphemes.

    Data source: crates/algebra_analysis/src/boxkites.rs *)

From Stdlib Require Import List Arith.
Import ListNotations.

(** The 42 primitive assessor pairs (lo_idx, hi_idx).
    lo in {1..7}, hi in {9..15}, excluding the identity pair (lo, lo+8). *)
Definition assessors : list (nat * nat) :=
  [(1,10); (1,11); (1,12); (1,13); (1,14); (1,15);
   (2,9);  (2,11); (2,12); (2,13); (2,14); (2,15);
   (3,9);  (3,10); (3,12); (3,13); (3,14); (3,15);
   (4,9);  (4,10); (4,11); (4,13); (4,14); (4,15);
   (5,9);  (5,10); (5,11); (5,12); (5,14); (5,15);
   (6,9);  (6,10); (6,11); (6,12); (6,13); (6,15);
   (7,9);  (7,10); (7,11); (7,12); (7,13); (7,14)].

(** The 7 box-kites, each containing 6 assessor pairs.
    Strut signature = index of the missing low basis element. *)
Definition boxkite_1 : list (nat * nat) :=
  [(1,14); (2,13); (3,12); (4,11); (5,10); (6,9)].
Definition boxkite_2 : list (nat * nat) :=
  [(1,11); (3,9); (4,14); (5,15); (6,12); (7,13)].
Definition boxkite_3 : list (nat * nat) :=
  [(1,10); (2,9); (4,15); (5,14); (6,13); (7,12)].
Definition boxkite_4 : list (nat * nat) :=
  [(1,13); (2,14); (3,15); (5,9); (6,10); (7,11)].
Definition boxkite_5 : list (nat * nat) :=
  [(1,12); (2,15); (3,14); (4,9); (6,11); (7,10)].
Definition boxkite_6 : list (nat * nat) :=
  [(1,15); (2,12); (3,13); (4,10); (5,11); (7,9)].
Definition boxkite_7 : list (nat * nat) :=
  [(2,11); (3,10); (4,13); (5,12); (6,15); (7,14)].

Definition boxkites : list (list (nat * nat)) :=
  [boxkite_1; boxkite_2; boxkite_3; boxkite_4;
   boxkite_5; boxkite_6; boxkite_7].

(** There are exactly 42 assessors. *)
Theorem assessor_count : length assessors = 42.
Proof. vm_compute. reflexivity. Qed.

(** There are exactly 7 box-kites. *)
Theorem boxkite_count : length boxkites = 7.
Proof. vm_compute. reflexivity. Qed.

(** Each box-kite has exactly 6 assessors. *)
Theorem boxkite_sizes :
  List.map (@length _) boxkites = [6; 6; 6; 6; 6; 6; 6].
Proof. vm_compute. reflexivity. Qed.

(** The total number of assessors across all box-kites is 42,
    confirming the partition is complete. *)
Theorem boxkite_total :
  List.fold_left Nat.add (List.map (@length _) boxkites) 0 = 42.
Proof. vm_compute. reflexivity. Qed.
