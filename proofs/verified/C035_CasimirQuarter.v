(** * C-035: F4 Casimir eigenvalue epsilon = 1/4.

    The Casimir eigenvalue for F4 in the relevant representation
    is 6/24 = 1/4. This determines the coupling strength in the
    sedenion gravi-electromagnetism model.

    Kernel-checked arithmetic in CasimirF4.v. *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import CasimirF4.
Open Scope R_scope.

(** epsilon = 1/4. *)
Theorem C035_casimir : f4_casimir_epsilon = 1 / 4.
Proof. exact f4_casimir_is_quarter. Qed.
