(** * C-932: Orthoplex thawing equation of state w(z).

    Verified claims:
    1. LCDM limit: beta=0 => w = -1 (cosmological constant recovery).
    2. LCDM limit: d_s=0 => w = -1 (no spectral dimension contribution).
    3. Thawing: beta >= 0 and d_s >= 0 imply w >= -1.
    4. Boundedness: w in [-1, -1 + ds_max] for beta in [0,1].
    5. Monotonicity: w increases with beta for fixed d_s >= 0.

    These properties establish that the orthoplex dark energy model
    is a well-behaved, physically motivated extension of Lambda-CDM.

    Mirrors: orthoplex_diffusion.rs lines 173, 182 (diffusion_time,
    w_orthoplex). *)

From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import HeatKernel.
From OpenGororoba Require Import DarkEnergyEOS.

Open Scope R_scope.

(** C-932a: LCDM recovery when beta = 0. *)
Theorem C932a_lcdm_limit_beta : forall ds : R,
  w_from_ds 0 ds = -1.
Proof.
  exact w_lcdm_limit_beta_zero.
Qed.

(** C-932b: LCDM recovery when spectral dimension vanishes. *)
Theorem C932b_lcdm_limit_ds : forall beta : R,
  w_from_ds beta 0 = -1.
Proof.
  exact w_lcdm_limit_ds_zero.
Qed.

(** C-932c: Thawing direction -- w departs from -1 upward. *)
Theorem C932c_thawing : forall beta ds : R,
  beta >= 0 -> ds >= 0 ->
  w_from_ds beta ds >= -1.
Proof.
  exact w_thawing.
Qed.

(** C-932d: Equation of state is bounded. *)
Theorem C932d_bounded : forall beta ds ds_max : R,
  0 <= beta -> beta <= 1 ->
  0 <= ds -> ds <= ds_max ->
  -1 <= w_from_ds beta ds <= -1 + ds_max.
Proof.
  exact w_bounded.
Qed.

(** C-932e: w is monotone increasing in beta. *)
Theorem C932e_monotone_beta : forall beta1 beta2 ds : R,
  ds >= 0 -> beta1 <= beta2 ->
  w_from_ds beta1 ds <= w_from_ds beta2 ds.
Proof.
  exact w_monotone_beta.
Qed.
