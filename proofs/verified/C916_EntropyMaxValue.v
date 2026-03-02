(** * C-916: Binary entropy at p=1/2 equals ln(2).

    H(1/2) = ln(2), the maximum of the binary entropy function. *)

From Stdlib Require Import Reals Rpower.
From OpenGororoba Require Import BinaryEntropy.

Theorem C916_entropy_max_value :
  binary_entropy (1/2) = ln 2.
Proof. exact binary_entropy_max_value. Qed.
