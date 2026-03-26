(** * Octonion right alternativity for basis element e2.
    (a * e2) * e2 = a * (e2 * e2).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e2 : forall a,
  oct_mul (oct_mul a (oct_e 2)) (oct_e 2) =
  oct_mul a (oct_mul (oct_e 2) (oct_e 2)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 2 a ltac:(lia)).
Qed.
