(** * C-899: Quaternion conjugate is an anti-automorphism.

    conj(p*q) = conj(q)*conj(p) for all p, q in H. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C899_quat_conj_antimorph : forall p q,
  quat_conj (quat_mul p q) = quat_mul (quat_conj q) (quat_conj p).
Proof. exact quat_conj_antimorphism. Qed.
