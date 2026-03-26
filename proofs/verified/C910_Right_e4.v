(** * Octonion right alternativity for basis element e4.
    (a * e4) * e4 = a * (e4 * e4).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e4 : forall a,
  oct_mul (oct_mul a (oct_e 4)) (oct_e 4) =
  oct_mul a (oct_mul (oct_e 4) (oct_e 4)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 4 a ltac:(lia)).
Qed.
