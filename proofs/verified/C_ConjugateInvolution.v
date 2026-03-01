(** * C_ConjugateInvolution: Cayley-Dickson conjugate is an involution.

    Proves conj(conj(z)) = z for both complex numbers and quaternions
    in the Cayley-Dickson construction. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem claim_complex_conj_involution :
  forall z, complex_conj (complex_conj z) = z.
Proof. exact complex_conj_involution. Qed.

Theorem claim_quat_conj_involution :
  forall q, quat_conj (quat_conj q) = q.
Proof. exact quat_conj_involution. Qed.
