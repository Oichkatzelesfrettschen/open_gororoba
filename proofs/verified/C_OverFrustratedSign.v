(** * C_OverFrustratedSign: frustration correction sign property.

    When the sedenion frustration density F > 3/8 (over-frustrated),
    the algebraic York-time correction is positive (enhancing).
    When F < 3/8 (under-frustrated), the correction is negative.

    The correction is proportional to (F - 3/8) * theta_gr * alpha_s,
    so the sign is determined by the sign of (F - F_vac).

    Mirrors: adm_algebra_bridge.rs lines 328-343 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** Simplified algebraic York-time correction:
    delta = (F - 3/8) * theta_gr * alpha_s. *)
Definition york_correction (F theta_gr alpha_s : R) : R :=
  (F - 3 / 8) * theta_gr * alpha_s.

(** Over-frustrated (F > 3/8) with positive theta_gr and alpha_s
    gives a positive correction. *)
Theorem over_frustrated_positive :
  forall F theta_gr alpha_s,
  F > 3 / 8 -> theta_gr > 0 -> alpha_s > 0 ->
  york_correction F theta_gr alpha_s > 0.
Proof.
  intros F theta_gr alpha_s HF Hth Ha.
  unfold york_correction.
  assert (Hdiff : F - 3 / 8 > 0) by lra.
  assert (Hprod : (F - 3 / 8) * theta_gr > 0) by nra.
  nra.
Qed.

(** Under-frustrated (F < 3/8) with positive theta_gr and alpha_s
    gives a negative correction. *)
Theorem under_frustrated_negative :
  forall F theta_gr alpha_s,
  F < 3 / 8 -> theta_gr > 0 -> alpha_s > 0 ->
  york_correction F theta_gr alpha_s < 0.
Proof.
  intros F theta_gr alpha_s HF Hth Ha.
  unfold york_correction.
  assert (Hdiff : F - 3 / 8 < 0) by lra.
  assert (Hprod : (F - 3 / 8) * theta_gr < 0) by nra.
  nra.
Qed.

(** At the attractor F = 3/8, the correction vanishes exactly. *)
Theorem at_attractor_vanishes :
  forall theta_gr alpha_s,
  york_correction (3 / 8) theta_gr alpha_s = 0.
Proof.
  intros. unfold york_correction. ring.
Qed.
