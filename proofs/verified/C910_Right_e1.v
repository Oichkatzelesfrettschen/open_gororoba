(** * Octonion right alternativity for basis element e1.
    (a * e1) * e1 = a * (e1 * e1).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e1 : forall a,
  oct_mul (oct_mul a (oct_e 1)) (oct_e 1) =
  oct_mul a (oct_mul (oct_e 1) (oct_e 1)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 1 a ltac:(lia)).
Qed.
