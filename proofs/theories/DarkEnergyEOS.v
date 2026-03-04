(** * Dark energy equation of state from orthoplex spectral dimension.

    Defines the thawing equation of state w(z) = -1 + beta * d_s(t(z))
    and proves key structural properties:

    - LCDM limit: beta = 0 implies w(z) = -1
    - Boundedness: 0 <= beta <= 1 implies w in [-1, -1 + beta * d_s_max]
    - Thawing direction: w(z) >= -1 when beta >= 0 and d_s >= 0

    Mirrors: orthoplex_diffusion.rs lines 173 and 182.

    References:
    - Sprint 60 claim C-932 (orthoplex thawing EOS)
    - Calcagni (2010), PRL 104, 251301 *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import HeatKernel.

Open Scope R_scope.

(** ---------- Scale mapping ---------- *)

(** Diffusion time as a function of redshift.
    t(z) = t_0 / (1+z)^alpha *)
Definition diffusion_time (z alpha t_0 : R) : R :=
  t_0 / (1 + z) ^ (Z.to_nat (up alpha)).

(** ---------- Equation of state ---------- *)

(** w(z) = -1 + beta * d_s
    where d_s is the spectral dimension at diffusion time t(z).

    For the formal proof, we abstract d_s as a real parameter
    since the full spectral dimension involves exp which makes
    algebraic proofs cleaner when separated. *)
Definition w_from_ds (beta ds : R) : R := -1 + beta * ds.

(** ---------- LCDM limit ---------- *)

(** When beta = 0, w = -1 regardless of spectral dimension.
    This is the Lambda-CDM recovery condition. *)
Theorem w_lcdm_limit_beta_zero : forall ds : R,
  w_from_ds 0 ds = -1.
Proof.
  intro ds. unfold w_from_ds. ring.
Qed.

(** When d_s = 0, w = -1 regardless of beta.
    This occurs at t=0 (Big Bang) and t=infinity (graph fully explored). *)
Theorem w_lcdm_limit_ds_zero : forall beta : R,
  w_from_ds beta 0 = -1.
Proof.
  intro beta. unfold w_from_ds. ring.
Qed.

(** ---------- Thawing direction ---------- *)

(** For beta >= 0 and d_s >= 0, we have w >= -1.
    The equation of state "thaws" from the cosmological constant value. *)
Theorem w_thawing : forall beta ds : R,
  beta >= 0 -> ds >= 0 ->
  w_from_ds beta ds >= -1.
Proof.
  intros beta ds Hb Hds. unfold w_from_ds. nra.
Qed.

(** ---------- Boundedness ---------- *)

(** For beta in [0,1] and d_s in [0, ds_max], w is bounded:
    -1 <= w <= -1 + beta * ds_max <= -1 + ds_max *)
Theorem w_bounded : forall beta ds ds_max : R,
  0 <= beta -> beta <= 1 ->
  0 <= ds -> ds <= ds_max ->
  -1 <= w_from_ds beta ds <= -1 + ds_max.
Proof.
  intros beta ds ds_max Hb0 Hb1 Hds0 Hdsm.
  unfold w_from_ds. split; nra.
Qed.

(** ---------- Monotonicity in beta ---------- *)

(** For fixed d_s >= 0, w is monotonically increasing in beta.
    Larger coupling means stronger departure from Lambda-CDM. *)
Theorem w_monotone_beta : forall beta1 beta2 ds : R,
  ds >= 0 -> beta1 <= beta2 ->
  w_from_ds beta1 ds <= w_from_ds beta2 ds.
Proof.
  intros beta1 beta2 ds Hds Hb. unfold w_from_ds. nra.
Qed.
