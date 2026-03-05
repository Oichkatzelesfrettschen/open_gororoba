(** * Octonion right alternativity for basis element e4.
    (a * e4) * e4 = a * (e4 * e4).
    Part of the C-910 right alternativity suite. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Theorem oct_right_alt_e4 : forall a,
  oct_mul (oct_mul a (oct_e 4)) (oct_e 4) =
  oct_mul a (oct_mul (oct_e 4) (oct_e 4)).
Proof.
  intros a. destruct a as [[aa ab ac ad] [ae af ag ah]].
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.
