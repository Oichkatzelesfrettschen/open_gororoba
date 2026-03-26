(** * Octonion right alternativity for basis element e6.
    (a * e6) * e6 = a * (e6 * e6).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e6 : forall a,
  oct_mul (oct_mul a (oct_e 6)) (oct_e 6) =
  oct_mul a (oct_mul (oct_e 6) (oct_e 6)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 6 a ltac:(lia)).
Qed.
