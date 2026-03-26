(** * Octonion right alternativity for basis element e0.
    (a * e0) * e0 = a * (e0 * e0).
    Part of the C-910 right alternativity suite. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C910_OctonionAlternative.

Theorem oct_right_alt_e0 : forall a,
  oct_mul (oct_mul a (oct_e 0)) (oct_e 0) =
  oct_mul a (oct_mul (oct_e 0) (oct_e 0)).
Proof.
  intros a.
  exact (C910_octonion_right_alt_basis 0 a ltac:(lia)).
Qed.
