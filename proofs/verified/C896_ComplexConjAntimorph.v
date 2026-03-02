(** * C-896: Complex conjugate is an anti-automorphism.

    conj(z*w) = conj(w)*conj(z) for all z, w in C. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C896_complex_conj_antimorph : forall z w,
  complex_conj (complex_mul z w) = complex_mul (complex_conj w) (complex_conj z).
Proof. exact complex_conj_antimorphism. Qed.
