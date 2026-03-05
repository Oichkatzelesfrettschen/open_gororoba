(** * C-003: 42 primitive assessors partition into 7 box-kites of 6.

    The sedenion zero-divisor pairs (lo, hi) with lo in 1..7, hi in 9..15
    (excluding identity pairs) number exactly 42 and partition into
    7 box-kites of 6 assessors each.

    Kernel-checked via vm_compute enumeration in BoxKite.v. *)

From Stdlib Require Import List.
From OpenGororoba Require Import BoxKite.
Import ListNotations.

(** There are exactly 42 assessors. *)
Theorem C003_assessor_count : length assessors = 42.
Proof. exact assessor_count. Qed.

(** There are exactly 7 box-kites. *)
Theorem C003_boxkite_count : length boxkites = 7.
Proof. exact boxkite_count. Qed.

(** Each box-kite has exactly 6 assessors. *)
Theorem C003_uniform_size :
  List.map (@length _) boxkites = [6; 6; 6; 6; 6; 6; 6].
Proof. exact boxkite_sizes. Qed.

(** 7 * 6 = 42: the partition is complete. *)
Theorem C003_partition_complete :
  List.fold_left Nat.add (List.map (@length _) boxkites) 0 = 42.
Proof. exact boxkite_total. Qed.
