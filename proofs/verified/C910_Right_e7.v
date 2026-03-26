(** * Octonion right alternativity for basis element e7.
    (a * e7) * e7 = a * (e7 * e7).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e7 : forall a,
  oct_mul (oct_mul a (oct_e 7)) (oct_e 7) =
  oct_mul a (oct_mul (oct_e 7) (oct_e 7)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 7 a ltac:(lia)).
Qed.
