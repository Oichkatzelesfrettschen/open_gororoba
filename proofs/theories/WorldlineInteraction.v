(** Two-body scalar interaction subtraction in the worldline representation.
    The attenuation factors are supplied by nonnegative body potentials.
    The identities establish subtraction, signs and dimensional scaling;
    Brownian sampling, continuum convergence and real-material response are
    separate obligations. Gies, Langfeld and Moyaerts, Eqs. 8, 13 and 44. *)
From Stdlib Require Import Reals Psatz.
Open Scope R_scope.

Definition two_body_weight first_attenuation second_attenuation :=
  1 - first_attenuation - second_attenuation + first_attenuation * second_attenuation.

Theorem interaction_subtraction_factorizes : forall first second,
  two_body_weight first second = (1-first) * (1-second).
Proof. intros; unfold two_body_weight; ring. Qed.

Theorem interaction_weight_range : forall first second,
  0 <= first <= 1 -> 0 <= second <= 1 -> 0 <= two_body_weight first second <= 1.
Proof.
  intros first second [first_lower first_upper] [second_lower second_upper].
  rewrite interaction_subtraction_factorizes. split; nra.
Qed.

Theorem transparent_body_has_zero_interaction : forall attenuation,
  two_body_weight 1 attenuation = 0 /\ two_body_weight attenuation 1 = 0.
Proof. intros; unfold two_body_weight; split; ring. Qed.

Definition dirichlet_interaction (first_hit second_hit : bool) : R :=
  if andb first_hit second_hit then 1 else 0.

Theorem dirichlet_requires_both_bodies : forall first second,
  dirichlet_interaction first second = 1 <-> first = true /\ second = true.
Proof. destruct first, second; simpl; unfold dirichlet_interaction; simpl; split; intros; try tauto; try lra; intuition discriminate. Qed.

Definition scalar_local_prefactor : R := -1 / (32 * PI^2).
Definition scalar_plate_energy (separation : R) := -PI^2 / (1440 * separation^3).
Definition electromagnetic_plate_energy (separation : R) := -PI^2 / (720 * separation^3).

Theorem scalar_prefactor_negative : scalar_local_prefactor < 0.
Proof.
  unfold scalar_local_prefactor, Rdiv.
  assert (0 < 32 * PI^2) by (pose proof PI_RGT_0; nra).
  pose proof (Rinv_0_lt_compat _ H). nra.
Qed.

Theorem electromagnetic_plate_has_two_scalar_polarizations : forall separation,
  0 < separation -> electromagnetic_plate_energy separation = 2 * scalar_plate_energy separation.
Proof. intros; unfold electromagnetic_plate_energy, scalar_plate_energy. field. nra. Qed.

Theorem scalar_plate_dilation : forall separation scale,
  0 < separation -> 0 < scale ->
  scalar_plate_energy (scale * separation) = scalar_plate_energy separation / scale^3.
Proof. intros; unfold scalar_plate_energy. field; nra. Qed.

Theorem proper_time_weight_dilation : forall proper_time scale,
  0 < proper_time -> 0 < scale ->
  (/ (scale^2 * proper_time)^3) = (/ proper_time^3) / scale^6.
Proof. intros. field; nra. Qed.

Theorem prefactor_uncertainty_nonnegative : forall prefactor variance,
  0 <= variance -> 0 <= Rabs prefactor * sqrt variance.
Proof. intros; apply Rmult_le_pos; [apply Rabs_pos|apply sqrt_pos]. Qed.
