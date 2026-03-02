(** * C-917: Binary entropy is nonnegative.

    H(p) >= 0 for all p in (0,1). *)

From Stdlib Require Import Reals Rpower.
From OpenGororoba Require Import BinaryEntropy.

Theorem C917_entropy_nonneg : forall p,
  0 < p < 1 -> binary_entropy p >= 0.
Proof. exact binary_entropy_nonneg. Qed.
