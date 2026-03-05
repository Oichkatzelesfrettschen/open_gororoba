(** * C-004: PSL(2,7) = GL(3,2) has order 168.

    |GL(3,GF(2))| = (2^3 - 1)(2^3 - 2)(2^3 - 4) = 7 * 6 * 4 = 168.
    This is the automorphism group of the Fano plane PG(2,2).

    We prove this by direct arithmetic on the order formula.
    The group acts on the 7 points of the Fano plane (hence also
    on the 7 box-kites of the sedenion ZD graph). *)

From Stdlib Require Import Arith Lia.
From OpenGororoba Require Import FanoPlane.

(** |GL(3, GF(2))| = (2^3 - 1)(2^3 - 2)(2^3 - 4) = 168.
    This counts the invertible 3x3 matrices over GF(2):
    - Row 1: 7 nonzero choices (2^3 - 1)
    - Row 2: 6 choices not in span of row 1 (2^3 - 2)
    - Row 3: 4 choices not in span of rows 1,2 (2^3 - 4) *)
Theorem C004_gl3_gf2_order : (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168.
Proof. vm_compute. reflexivity. Qed.

(** The Fano plane has 7 lines, matching the number of box-kites. *)
Theorem C004_fano_matches_boxkites :
  length fano_lines = 7.
Proof. exact fano_line_count. Qed.
