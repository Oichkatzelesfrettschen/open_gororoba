(** * C_NormConjugate: Brahmagupta-Fibonacci norm-conjugate identity.

    Proves z * conj(z) = |z|^2 for both complex numbers and quaternions.
    This is the multiplicativity of the norm (Brahmagupta-Fibonacci identity)
    specialized to self-conjugate products. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem claim_complex_norm_conjugate :
  forall z, complex_mul z (complex_conj z) = mkComplex (complex_norm_sq z) 0.
Proof. exact complex_norm_conjugate. Qed.

Theorem claim_quat_norm_conjugate :
  forall q, quat_mul q (quat_conj q) = mkQuat (quat_norm_sq q) 0 0 0.
Proof. exact quat_norm_conjugate. Qed.
