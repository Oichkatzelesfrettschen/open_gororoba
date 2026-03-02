(** * BransDicke: Brans-Dicke PPN parameters.

    Defines the PPN gamma and Nordtvedt parameter for scalar-tensor
    gravity in the Brans-Dicke theory.  gamma_BD(omega) = (1+omega)/(2+omega)
    approaches GR (gamma=1) as omega -> infinity.

    Mirrors: gr_core/src/ppn_constraints.rs *)

From OpenGororoba Require Import Prelude.

(*<*bdgamma>*)
(** PPN gamma in Brans-Dicke theory. *)
Definition ppn_gamma_bd (omega : R) : R := (1 + omega) / (2 + omega).

(** Nordtvedt parameter: deviation from strong equivalence. *)
Definition nordtvedt_bd (omega : R) : R := 1 / (2 + omega).
(*</bdgamma>*)

(** Gamma deviation from GR: gamma - 1. *)
Definition ppn_gamma_deviation (omega : R) : R :=
  ppn_gamma_bd omega - 1.
