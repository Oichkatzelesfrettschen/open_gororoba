(** Scalar oscillator and cutoff consequences with explicit model parameters.
    These statements concern declared spectra and harmonic response. Source
    attribution, density-of-states validity and measured coupling remain
    separate evidence obligations. *)
From Stdlib Require Import Reals Psatz.
Open Scope R_scope.

Definition absorption_line coefficient occupation := coefficient * (occupation + 1).
Definition emission_line coefficient occupation := coefficient * occupation.

Theorem vacuum_ordered_line_asymmetry : forall coefficient,
  0 < coefficient -> emission_line coefficient 0 = 0 /\ absorption_line coefficient 0 > 0.
Proof. intros; unfold emission_line, absorption_line; split; nra. Qed.

Theorem oscillator_variance_survives_zero_occupation : forall coefficient,
  0 < coefficient ->
  absorption_line coefficient 0 + emission_line coefficient 0 = coefficient /\ coefficient > 0.
Proof. intros; unfold absorption_line, emission_line; split; nra. Qed.

Theorem ordered_lines_obey_bose_balance : forall coefficient inverse_temperature_frequency,
  0 < inverse_temperature_frequency ->
  absorption_line coefficient (/ (exp inverse_temperature_frequency - 1)) =
  exp inverse_temperature_frequency *
    emission_line coefficient (/ (exp inverse_temperature_frequency - 1)).
Proof.
  intros coefficient argument positive.
  pose proof (exp_ineq1 argument) as lower.
  assert (exp argument - 1 <> 0) by (assert (argument <> 0) by lra; specialize (lower H); lra).
  unfold absorption_line, emission_line. field. assumption.
Qed.

Definition harmonic_zero_point_variance hbar mass stiffness :=
  hbar / (2 * sqrt (mass * stiffness)).

Theorem harmonic_variance_positive : forall hbar mass stiffness,
  0 < hbar -> 0 < mass -> 0 < stiffness ->
  0 < harmonic_zero_point_variance hbar mass stiffness.
Proof.
  intros hbar mass stiffness hbar_positive mass_positive stiffness_positive.
  assert (0 < sqrt (mass * stiffness)).
  { pose proof (sqrt_lt_1 0 (mass * stiffness) ltac:(lra) ltac:(nra) ltac:(nra)). rewrite sqrt_0 in H. exact H. }
  unfold harmonic_zero_point_variance, Rdiv.
  apply Rmult_lt_0_compat; [assumption|apply Rinv_0_lt_compat; lra].
Qed.

Theorem stable_softening_increases_zero_point_variance : forall hbar mass softer stiffer,
  0 < hbar -> 0 < mass -> 0 < softer -> softer < stiffer ->
  harmonic_zero_point_variance hbar mass stiffer <
  harmonic_zero_point_variance hbar mass softer.
Proof.
  intros hbar mass softer stiffer hbar_positive mass_positive softer_positive ordered.
  assert (root_positive : 0 < sqrt (mass * softer)).
  { pose proof (sqrt_lt_1 0 (mass * softer) ltac:(lra) ltac:(nra) ltac:(nra)). rewrite sqrt_0 in H. exact H. }
  assert (root_ordered : sqrt (mass * softer) < sqrt (mass * stiffer)).
  { apply sqrt_lt_1; nra. }
  unfold harmonic_zero_point_variance, Rdiv.
  apply Rmult_lt_compat_l; [assumption|].
  apply Rinv_lt_contravar; nra.
Qed.

Theorem stable_softening_increases_classical_thermal_variance : forall temperature_scale softer stiffer,
  0 < temperature_scale -> 0 < softer -> softer < stiffer ->
  temperature_scale / stiffer < temperature_scale / softer.
Proof.
  intros temperature_scale softer stiffer positive softer_positive ordered.
  unfold Rdiv. apply Rmult_lt_compat_l; [assumption|]. apply Rinv_lt_contravar; nra.
Qed.

Definition cutoff_density coefficient cutoff := coefficient * cutoff^4.
Theorem ultraviolet_cutoff_quartic_scaling : forall coefficient cutoff scale,
  cutoff_density coefficient (scale * cutoff) = scale^4 * cutoff_density coefficient cutoff.
Proof. intros; unfold cutoff_density; ring. Qed.

Theorem equal_units_do_not_determine_a_map :
  exists (first second : R -> R),
    (forall speed, first speed = speed) /\
    (forall speed, second speed = 2*speed) /\ first 1 <> second 1.
Proof. exists (fun speed => speed), (fun speed => 2*speed). repeat split; try reflexivity; lra. Qed.

Theorem input_independent_nonzero_map_is_not_additive : forall constant,
  constant <> 0 -> ~ (forall first second : R,
    (fun _ : R => constant) (first + second) =
    (fun _ : R => constant) first + (fun _ : R => constant) second).
Proof. intros constant nonzero additive. specialize (additive 0 0). simpl in additive. lra. Qed.
