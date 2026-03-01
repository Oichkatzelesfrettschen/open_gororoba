(** * C-877: Zero coupling gives zero algebraic correction.

    Formal proof that when alpha_s = 0 (zero coupling constant),
    the algebraic York time correction vanishes identically for
    all frustration values and all York times.

    Mirrors: algebraic_york_time_correction() in adm_algebra_bridge.rs:174-180.
    Rust test: test_zero_coupling_gives_zero_correction (line 264-268). *)

From OpenGororoba Require Import Prelude ADMAlgebraBridge.

(** CLAIM C-877: Zero coupling => zero correction, universally. *)
Theorem C877_zero_coupling :
  forall F theta : R,
    algebraic_york_time_correction F theta 0 = 0.
Proof.
  intros F theta.
  unfold algebraic_york_time_correction.
  ring.
Qed.

(** Strengthening: the correction is linear in the coupling constant. *)
Theorem C877_correction_linear_in_alpha :
  forall F theta alpha1 alpha2 : R,
    algebraic_york_time_correction F theta (alpha1 + alpha2) =
    algebraic_york_time_correction F theta alpha1 +
    algebraic_york_time_correction F theta alpha2.
Proof.
  intros. unfold algebraic_york_time_correction. ring.
Qed.

(** The correction is also linear in theta. *)
Theorem C877_correction_linear_in_theta :
  forall F theta1 theta2 alpha_s : R,
    algebraic_york_time_correction F (theta1 + theta2) alpha_s =
    algebraic_york_time_correction F theta1 alpha_s +
    algebraic_york_time_correction F theta2 alpha_s.
Proof.
  intros. unfold algebraic_york_time_correction. ring.
Qed.
