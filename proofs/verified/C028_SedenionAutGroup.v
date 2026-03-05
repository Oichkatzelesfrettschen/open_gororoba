(** * C-028: Sedenion automorphism group structure.

    Aut(S) preserves 3 octonionic subalgebras corresponding to the
    3 ways to embed O into S. Each subalgebra is the lo-half of a
    sedenion when the hi-half is zero.

    We prove: the lo-half embedding is closed under the sedenion product
    (when hi = 0). This is the algebraic prerequisite for the Aut(S)
    decomposition. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Embed an octonion into the sedenion lo-half. *)
Definition sed_embed_lo (x : CDOct) : CDSed := mkSed x oct_zero.

(** The lo-half embedding is closed: if hi=0 for both operands,
    the product also has hi=0.
    This follows from CD doubling: (a,0)(c,0) = (ac, 0). *)
Theorem C028_lo_subalgebra_closed : forall a c : CDOct,
  sed_mul (sed_embed_lo a) (sed_embed_lo c) =
  sed_embed_lo (oct_mul a c).
Proof.
  intros a c.
  destruct a as [[a0 a1 a2 a3] [a4 a5 a6 a7]].
  destruct c as [[c0 c1 c2 c3] [c4 c5 c6 c7]].
  unfold sed_embed_lo, sed_mul, oct_mul, oct_conj, oct_zero,
         quat_mul, quat_add, quat_neg, quat_conj, quat_zero, quat_one.
  simpl. unfold oct_zero, quat_zero.
  f_equal; f_equal; f_equal; ring.
Qed.
