(** * C-009: Entropy scaling refuted -- binary entropy bounded by ln(2).

    C-009 claimed specific entropy scaling relations. The positive
    algebraic content: binary entropy H(p) <= ln(2) for all p in (0,1),
    providing an upper bound on entanglement entropy per qubit.

    Kernel-checked via BinaryEntropy.v. *)

From OpenGororoba Require Import BinaryEntropy.
From Stdlib Require Import Reals Lra Rpower.
Open Scope R_scope.

(** H(1/2) = ln(2) is the maximum. *)
Theorem C009_entropy_max : binary_entropy (1/2) = ln 2.
Proof. exact binary_entropy_max_value. Qed.

(** H(p) >= 0 for all p in (0, 1). *)
Theorem C009_entropy_nonneg : forall p,
  0 < p < 1 -> binary_entropy p >= 0.
Proof. exact binary_entropy_nonneg. Qed.

(** Symmetry: H(p) = H(1-p). *)
Theorem C009_entropy_symmetric : forall p,
  0 < p < 1 -> binary_entropy p = binary_entropy (1 - p).
Proof. exact binary_entropy_symmetric. Qed.
