(** * C-884: Neg-dim EOS parameter degeneracy (rank-1 Jacobian).

    The negative-dimension equation of state w_eff = -1 - eta*(alpha + 3/2)
    depends on (alpha, eta) only through the product P = eta*(alpha + 3/2).
    The 2D parameterization is therefore not identifiable from w_eff alone:
    any curve in (alpha, eta) space with constant P gives the same w_eff.

    This is the positive mathematical content extracted from refuted C-012,
    which claimed specific (alpha, eta) values could be determined from
    cosmological observations. The degeneracy proves that claim impossible.

    Reformulation of refuted C-012. *)

From OpenGororoba Require Import Prelude.

(** Neg-dim effective equation of state.
    alpha: spectral index, eta: vacuum coupling strength.
    The formula comes from negative-dimension regularization of
    the vacuum energy spectral integral. *)
Definition neg_dim_w_eff (alpha eta : R) : R :=
  -1 - eta * (alpha + 3 / 2).

(** The degeneracy product: the only observable combination. *)
Definition neg_dim_product (alpha eta : R) : R :=
  eta * (alpha + 3 / 2).

(** Core theorem: same product implies same w_eff.
    This is the formal statement that the parameterization is degenerate. *)
(*<*negdimdegen>*)
Theorem neg_dim_product_degeneracy :
  forall a1 e1 a2 e2 : R,
  neg_dim_product a1 e1 = neg_dim_product a2 e2 ->
  neg_dim_w_eff a1 e1 = neg_dim_w_eff a2 e2.
Proof.
  intros a1 e1 a2 e2 Hprod.
  unfold neg_dim_w_eff, neg_dim_product in *.
  lra.
Qed.
(*</negdimdegen>*)

(** Vacuum limit: when eta = 0, w_eff = -1 regardless of alpha.
    This recovers the cosmological constant equation of state. *)
Theorem neg_dim_vacuum :
  forall alpha : R, neg_dim_w_eff alpha 0 = -1.
Proof.
  intros alpha. unfold neg_dim_w_eff. ring.
Qed.

(** Phantom divide: w_eff < -1 iff eta*(alpha + 3/2) > 0. *)
Theorem neg_dim_phantom :
  forall alpha eta : R,
  neg_dim_w_eff alpha eta < -1 <-> neg_dim_product alpha eta > 0.
Proof.
  intros alpha eta.
  unfold neg_dim_w_eff, neg_dim_product.
  split; intros H; lra.
Qed.

(** Quintessence regime: w_eff > -1 iff eta*(alpha + 3/2) < 0. *)
Theorem neg_dim_quintessence :
  forall alpha eta : R,
  neg_dim_w_eff alpha eta > -1 <-> neg_dim_product alpha eta < 0.
Proof.
  intros alpha eta.
  unfold neg_dim_w_eff, neg_dim_product.
  split; intros H; lra.
Qed.

(** Explicit degeneracy witness: two distinct parameter pairs with same w_eff. *)
Theorem neg_dim_degeneracy_witness :
  neg_dim_w_eff 0 1 = neg_dim_w_eff 1 (3/5).
Proof.
  unfold neg_dim_w_eff. field.
Qed.
