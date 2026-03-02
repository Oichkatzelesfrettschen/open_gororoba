(** * C-918: Complex multiplication is commutative (property tower: dim=2 OK).

    The first Cayley-Dickson level (C) preserves commutativity.
    Same content as C-893 but framed as property-tower member. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C918_complex_is_commutative : forall z w : CDComplex,
  complex_mul z w = complex_mul w z.
Proof. exact complex_mul_comm. Qed.
