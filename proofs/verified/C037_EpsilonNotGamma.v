(** * C-037: epsilon (Casimir) and gamma (Barbero-Immirzi) are independent.

    C-037 claimed a relationship between the F4 Casimir eigenvalue
    epsilon = 1/4 and the Barbero-Immirzi parameter gamma = ln(2)/(pi*sqrt(3)).
    This is refuted: they arise from different physical contexts
    (Lie algebra representation theory vs LQG black hole entropy).

    Positive content: epsilon = 1/4 is a pure algebraic ratio,
    independent of any LQG parameter. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import CasimirF4.
Open Scope R_scope.

(** epsilon is a pure algebraic ratio. *)
Theorem C037_epsilon_algebraic : f4_casimir_epsilon = 1 / 4.
Proof. exact f4_casimir_is_quarter. Qed.

(** epsilon <> 0 (non-degenerate coupling). *)
Theorem C037_epsilon_nonzero : f4_casimir_epsilon <> 0.
Proof.
  rewrite f4_casimir_is_quarter. lra.
Qed.
