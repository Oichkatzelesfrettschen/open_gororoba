(** * C-894: Complex multiplication is associative.

    (x * y) * z = x * (y * z) for all x, y, z in C. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C894_complex_associativity : forall x y z,
  complex_mul (complex_mul x y) z = complex_mul x (complex_mul y z).
Proof. exact complex_mul_assoc. Qed.
