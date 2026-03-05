(** * C-020: All octonion basis elements have unit norm (refuted reformulation).

    C-020 made a legacy adjacency claim about basis elements.
    The positive algebraic content: every oct basis element e_i (i in 0..7)
    has norm 1, so no basis element can be a zero divisor.

    Kernel-checked via case analysis + ring in OctonionNorm.v. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Every octonion basis element has unit norm. *)
Theorem C020_basis_unit_norm : forall i : nat,
  (i < 8)%nat -> oct_norm_sq (oct_e i) = 1.
Proof. exact oct_basis_unit_norm. Qed.

(** Corollary: no basis element has zero norm. *)
Theorem C020_basis_nonzero_norm : forall i : nat,
  (i < 8)%nat -> oct_norm_sq (oct_e i) <> 0.
Proof.
  intros i Hi. rewrite oct_basis_unit_norm by exact Hi. lra.
Qed.
