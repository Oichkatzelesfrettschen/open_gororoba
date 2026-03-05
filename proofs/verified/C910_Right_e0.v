(** * Octonion right alternativity for basis element e0.
    (a * e0) * e0 = a * (e0 * e0).
    Part of the C-910 right alternativity suite. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Theorem oct_right_alt_e0 : forall a,
  oct_mul (oct_mul a (oct_e 0)) (oct_e 0) =
  oct_mul a (oct_mul (oct_e 0) (oct_e 0)).
Proof.
  intros a. destruct a as [[aa ab ac ad] [ae af ag ah]].
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.
