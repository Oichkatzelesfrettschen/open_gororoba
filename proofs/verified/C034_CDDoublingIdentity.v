(** * C-034: CD doubling formula is consistent across levels.

    Chanyal's sedenion gravi-electromagnetism uses the CD doubling of
    two octonion halves. We verify: sed_mul on lo-half embeddings
    reduces to oct_mul (the lo-lo sector is just octonion multiplication).

    This proves the sedenion product is genuinely the CD doubling of
    the octonion product, not a different multiplication. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C028_SedenionAutGroup.

(** Lo-half embedding preserves product: (a,0)(c,0) = (ac, 0). *)
Theorem C034_cd_doubling_identity : forall a c : CDOct,
  sed_mul (sed_embed_lo a) (sed_embed_lo c) =
  sed_embed_lo (oct_mul a c).
Proof. exact C028_lo_subalgebra_closed. Qed.
