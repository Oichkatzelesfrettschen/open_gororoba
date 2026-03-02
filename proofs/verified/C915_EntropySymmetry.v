(** * C-915: Binary entropy is symmetric about p = 1/2.

    H(p) = H(1-p) for all p in (0,1). *)

From Stdlib Require Import Reals Rpower.
From OpenGororoba Require Import BinaryEntropy.

Theorem C915_entropy_symmetry : forall p,
  0 < p < 1 -> binary_entropy p = binary_entropy (1 - p).
Proof. exact binary_entropy_symmetric. Qed.
