(** * C-012: Negative dimension parameter degeneracy (refuted).

    C-012 claimed specific (alpha, eta) values could be determined from
    cosmological observations. This is refuted: the EOS w_eff depends
    on (alpha, eta) only through the product eta*(alpha + 3/2).

    Reformulated as C-884. This file cites the kernel-checked proof. *)

From OpenGororoba Require Import Prelude.
From OpenGororobaVerified Require Import C884_NegDimDegeneracy.

(** The core degeneracy: same product => same w_eff. *)
Theorem C012_degeneracy :
  forall a1 e1 a2 e2 : R,
  neg_dim_product a1 e1 = neg_dim_product a2 e2 ->
  neg_dim_w_eff a1 e1 = neg_dim_w_eff a2 e2.
Proof. exact neg_dim_product_degeneracy. Qed.

(** Explicit witness: two distinct parameter pairs with same w_eff. *)
Theorem C012_witness :
  neg_dim_w_eff 0 1 = neg_dim_w_eff 1 (3/5).
Proof. exact neg_dim_degeneracy_witness. Qed.
