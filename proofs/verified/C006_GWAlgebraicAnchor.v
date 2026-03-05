(** * C-006: Gravitational wave algebraic anchor.

    GWTC-3 data verified via Rust; Rocq scope = algebra only.
    The algebraic anchor: octonion multiplication is left-distributive
    over addition, which is the linearity property any correct
    implementation of the CD product must satisfy.

    This bounds the algebraic content available to the GW analysis. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Octonion multiplication left-distributes over addition. *)
Theorem C006_oct_left_distr : forall a b c : CDOct,
  oct_mul a (oct_add b c) =
  oct_add (oct_mul a b) (oct_mul a c).
Proof.
  intros a b c.
  destruct a as [[a0 a1 a2 a3] [a4 a5 a6 a7]].
  destruct b as [[b0 b1 b2 b3] [b4 b5 b6 b7]].
  destruct c as [[c0 c1 c2 c3] [c4 c5 c6 c7]].
  cbv [oct_mul oct_add oct_conj
       quat_mul quat_add quat_neg quat_conj
       oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.

(** Octonion norm is multiplicative (bounds algebraic precision). *)
Theorem C006_oct_norm_mul : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq x * oct_norm_sq y.
Proof. exact oct_norm_mul. Qed.
