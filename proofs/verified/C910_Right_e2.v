(** * Octonion right alternativity for basis element e2.
    (a * e2) * e2 = a * (e2 * e2).
    Part of the C-910 right alternativity suite. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Theorem oct_right_alt_e2 : forall a,
  oct_mul (oct_mul a (oct_e 2)) (oct_e 2) =
  oct_mul a (oct_mul (oct_e 2) (oct_e 2)).
Proof.
  intros a. destruct a as [[aa ab ac ad] [ae af ag ah]].
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.
