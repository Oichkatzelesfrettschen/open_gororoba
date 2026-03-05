(** * Sedenion conjugate is an involution.
    sed_conj(sed_conj(x)) = x for all x : CDSed.

    The proof decomposes along the CD doubling structure:
    - lo part: oct_conj(oct_conj(lo)) = lo  (from C_OctConjInvolution)
    - hi part: the sedenion conjugate negates each quaternion component
      of hi, so applying it twice gives quat_neg(quat_neg(q)) = q. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororobaVerified Require Import C_OctConjInvolution.

Theorem sed_conj_involution : forall x : CDSed, sed_conj (sed_conj x) = x.
Proof.
  intros x. destruct x as [lo [hlo hhi]].
  cbv [sed_conj sed_lo sed_hi oct_lo oct_hi].
  f_equal.
  - exact (oct_conj_involution lo).
  - destruct hlo as [a1 b1 c1 d1]. destruct hhi as [a2 b2 c2 d2].
    cbv [quat_neg qa qb qc qd]. f_equal; f_equal; ring.
Qed.
