(** * C-1236: Flat band group velocity bound.

    CLAIM: For a flat band with bandwidth W < eps, the maximum group
    velocity |dE/dk| is bounded by W/dk where dk is the BZ grid
    spacing. In the limit dk -> 0, the group velocity vanishes
    identically for a perfectly flat band (W = 0).

    This bound implies that the kagome flat band (W ~ 0) has
    vanishing group velocity => infinite effective mass =>
    DOS enhancement proportional to 1/W.

    STRATEGY: From the axiom group_velocity_bandwidth_bound:
      max_group_velocity E dk <= bandwidth E / dk
    and the flat band hypothesis bandwidth E < eps, we derive:
      max_group_velocity E dk < eps / dk.

    Mirrors: crates/quantum_core/src/tight_binding.rs (detect_flat_bands)
    Binary:  magnonic-band-structure --flat-band *)

From OpenGororoba Require Import Prelude.
Require Import OpenGororoba.TightBindingTopology.

Open Scope R_scope.

(** A flat band has bounded group velocity. *)
Theorem flat_band_group_velocity_bound :
  forall (E : band_dispersion) (eps dk : R),
    is_flat_band E eps ->
    0 < dk ->
    max_group_velocity E dk < eps / dk.
Proof.
  intros E eps dk [Heps_pos Hbw] Hdk.
  assert (Hbound := group_velocity_bandwidth_bound E dk Hdk).
  assert (Hlt : bandwidth E / dk < eps / dk).
  { apply Rmult_lt_compat_r.
    - apply Rinv_0_lt_compat. exact Hdk.
    - exact Hbw. }
  lra.
Qed.

(** Corollary: A perfectly flat band (bandwidth = 0) has zero group
    velocity at any grid resolution. *)
Theorem perfectly_flat_zero_velocity :
  forall (E : band_dispersion) (dk : R),
    bandwidth E = 0 ->
    0 < dk ->
    max_group_velocity E dk <= 0.
Proof.
  intros E dk Hzero Hdk.
  assert (Hbound := group_velocity_bandwidth_bound E dk Hdk).
  rewrite Hzero in Hbound.
  assert (H0dk : 0 / dk = 0).
  { field. lra. }
  lra.
Qed.
