(** * C-895: Complex norm is multiplicative (Brahmagupta-Fibonacci at dim=2).

    |z*w|^2 = |z|^2 * |w|^2 for all z, w in C. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C895_complex_norm_multiplicative : forall z w,
  complex_norm_sq (complex_mul z w) = complex_norm_sq z * complex_norm_sq w.
Proof. exact complex_norm_mul. Qed.
