(** * C-921: Associativity fails at dim=8 (property tower: second loss).

    Octonions are the first CD level to lose associativity.
    Proved by reference to C-909 (explicit witness). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C909_OctonionNonAssociative.

Theorem C921_oct_loses_associativity :
  oct_mul (oct_mul oe1 oe2) oe4 <> oct_mul oe1 (oct_mul oe2 oe4).
Proof. exact C909_octonion_non_associative. Qed.
