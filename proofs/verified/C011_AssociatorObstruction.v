(** * C-011: Associator obstruction to standard action principles.

    The sedenion associator is nonzero (C-030), which blocks the
    standard associative action principle from being applied directly.
    Any gravastar model using sedenions requires explicit handling
    of the non-associativity.

    This file documents the obstruction and cites C-030's proof. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C030_AssociatorNonzero.

(** The obstruction: sedenion associator is nonzero.
    This means (a*b)*c <> a*(b*c) for the embedded octonion witness. *)
Theorem C011_obstruction :
  sed_assoc sed_e1 sed_e2 sed_e4 <> sed_zero.
Proof. exact C030_sed_assoc_nonzero. Qed.
