(** * Octonion right alternativity for basis element e5.
    (a * e5) * e5 = a * (e5 * e5).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e5 : forall a,
  oct_mul (oct_mul a (oct_e 5)) (oct_e 5) =
  oct_mul a (oct_mul (oct_e 5) (oct_e 5)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 5 a ltac:(lia)).
Qed.
