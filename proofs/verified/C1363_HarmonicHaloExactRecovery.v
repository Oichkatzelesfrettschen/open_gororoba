(** * C-1363: Exact alpha=0 recovery for the harmonic-halo law.

    This file formalizes the exact kernel used by
    `crates/cosmology_core/src/harmonic_halos.rs`:

    - the multiplicative harmonic-halo modulation factor is exactly 1 when
      alpha_zd = 0;
    - therefore the modulated squared circular velocity recovers the baseline
      squared circular velocity exactly.

    This is an exact model-law statement only. It does not formalize any
    observational fit or dark matter microphysics claim. *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** Abstract the Rust modulation core into a single real-valued term. *)
Definition harmonic_halo_modulation_law (alpha_zd modulation_core : R) : R :=
  1 + alpha_zd * modulation_core.

(** Squared circular velocity in the multiplicative model. *)
Definition harmonic_halo_velocity_sq
    (baseline_velocity_sq alpha_zd modulation_core : R) : R :=
  baseline_velocity_sq * harmonic_halo_modulation_law alpha_zd modulation_core.

Theorem C1363_alpha_zero_modulation_unity :
  forall modulation_core : R,
    harmonic_halo_modulation_law 0 modulation_core = 1.
Proof.
  intros modulation_core.
  unfold harmonic_halo_modulation_law.
  ring.
Qed.

Theorem C1363_alpha_zero_exact_recovery_sq :
  forall baseline_velocity_sq modulation_core : R,
    harmonic_halo_velocity_sq baseline_velocity_sq 0 modulation_core =
    baseline_velocity_sq.
Proof.
  intros baseline_velocity_sq modulation_core.
  unfold harmonic_halo_velocity_sq.
  rewrite C1363_alpha_zero_modulation_unity.
  ring.
Qed.

Theorem C1363_zero_baseline_stays_zero :
  forall alpha_zd modulation_core : R,
    harmonic_halo_velocity_sq 0 alpha_zd modulation_core = 0.
Proof.
  intros alpha_zd modulation_core.
  unfold harmonic_halo_velocity_sq.
  ring.
Qed.
