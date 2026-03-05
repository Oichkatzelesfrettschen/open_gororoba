(** * CasimirF4: F4 Casimir eigenvalue epsilon = 1/4.

    The Casimir eigenvalue for the F4 exceptional Lie group representation
    relevant to octonion triality is epsilon = 6/24 = 1/4.

    This is pure arithmetic, but the physical significance is that it
    determines the coupling constant in the sedenion gravi-electromagnetism
    model (Chanyal 2024). *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

Definition f4_casimir_epsilon : R := 6 / 24.

Theorem f4_casimir_is_quarter : f4_casimir_epsilon = 1 / 4.
Proof. unfold f4_casimir_epsilon. field. Qed.

Theorem f4_casimir_reduced : 6 / 24 = 1 / 4.
Proof. field. Qed.
