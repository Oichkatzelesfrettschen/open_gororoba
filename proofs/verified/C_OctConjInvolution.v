(** * Octonion conjugate is an involution.
    oct_conj(oct_conj(x)) = x for all x : CDOct.
    Extends C_ConjugateInvolution (which covers complex and quaternion dims). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Lemma quat_neg_neg : forall q : CDQuat, quat_neg (quat_neg q) = q.
Proof.
  intros q. destruct q as [a b c d].
  cbv [quat_neg qa qb qc qd]. f_equal; ring.
Qed.

Theorem oct_conj_involution : forall x : CDOct, oct_conj (oct_conj x) = x.
Proof.
  intros x. destruct x as [lo hi].
  cbv [oct_conj oct_lo oct_hi].
  f_equal.
  - exact (quat_conj_involution lo).
  - exact (quat_neg_neg hi).
Qed.
