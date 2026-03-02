(** * C-893: Complex multiplication is commutative.

    z * w = w * z for all z, w in C (Cayley-Dickson dim=2). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C893_complex_commutativity : forall z w,
  complex_mul z w = complex_mul w z.
Proof. exact complex_mul_comm. Qed.
