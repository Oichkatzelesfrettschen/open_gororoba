(** * Octonion right alternativity for basis element e3.
    (a * e3) * e3 = a * (e3 * e3).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e3 : forall a,
  oct_mul (oct_mul a (oct_e 3)) (oct_e 3) =
  oct_mul a (oct_mul (oct_e 3) (oct_e 3)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 3 a ltac:(lia)).
Qed.
