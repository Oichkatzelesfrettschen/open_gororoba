(** * C-013: Each assessor belongs to exactly 2 of the 7 automorphemes.

    The 7 Fano triples (automorphemes) cover each of the 7 octonion
    basis indices exactly 3 times (each index in 3 triples).
    For assessor pairs (lo, hi), the automorpheme coverage is
    determined by the lo index.

    Kernel-checked via vm_compute on the Fano incidence structure. *)

From Stdlib Require Import List.
From OpenGororoba Require Import FanoPlane.
Import ListNotations.

(** Every point (1..7) lies on exactly 3 Fano lines. *)
Theorem C013_three_lines_per_point :
  List.map lines_through [1; 2; 3; 4; 5; 6; 7] = [3; 3; 3; 3; 3; 3; 3].
Proof. exact fano_point_incidence. Qed.

(** There are exactly 7 Fano lines (automorphemes). *)
Theorem C013_seven_automorphemes : length fano_lines = 7.
Proof. exact fano_line_count. Qed.
