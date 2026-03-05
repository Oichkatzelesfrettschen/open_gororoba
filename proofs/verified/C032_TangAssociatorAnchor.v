(** * C-032: Tang associator algebraic anchor.

    Tang (2025) uses associator norms for mass predictions.
    The Rocq anchor: the octonion associator [a,b,c] is well-defined
    and nonzero for the canonical witness triple (e1, e2, e4).

    This is the algebraic fact underpinning Tang's framework:
    the associator carries genuine physical content at dim >= 8. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** The octonion associator is nonzero for the canonical witness. *)
Theorem C032_assoc_nonzero :
  oct_assoc (oct_e 1) (oct_e 2) (oct_e 4) <> oct_zero.
Proof. exact oct_assoc_nonzero. Qed.

(** The associator norm is computable and positive. *)
Theorem C032_assoc_has_norm :
  oct_norm_sq (oct_assoc (oct_e 1) (oct_e 2) (oct_e 4)) > 0.
Proof.
  cbv [oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       oct_e oct_zero oct_norm_sq quat_norm_sq
       quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  nra.
Qed.
