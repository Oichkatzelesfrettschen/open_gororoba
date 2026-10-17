(** Sampled classical Gaussian phase noise under an explicitly PSD covariance.
    Weights include pulse sign, quadrature duration and calibrated coupling.
    The Gaussian characteristic-function formula is the declared model;
    ordered quantum detector spectra retain a separate contract. *)
From Stdlib Require Import Reals Psatz.
From OpenGororoba Require Import PassiveMechanicalMemory.
Open Scope R_scope.

Definition gaussian_phase_variance dimension covariance weights :=
  quadratic_form dimension covariance weights weights.
Definition gaussian_coherence (relaxation_exponent phase_variance : R) :=
  exp (-relaxation_exponent - phase_variance / 2).

Theorem psd_phase_variance_nonnegative : forall dimension covariance weights,
  form_psd dimension covariance ->
  0 <= gaussian_phase_variance dimension covariance weights.
Proof. intros; apply H. Qed.

Lemma exp_nonpositive_range : forall exponent,
  exponent <= 0 -> 0 < exp exponent <= 1.
Proof.
  intros exponent bounded. split; [apply exp_pos|].
  destruct (Rle_lt_or_eq_dec exponent 0 bounded) as [strict|equal].
  - pose proof (exp_increasing exponent 0 strict). rewrite exp_0 in H. lra.
  - rewrite equal, exp_0. lra.
Qed.

Theorem gaussian_coherence_range : forall dimension covariance weights relaxation,
  form_psd dimension covariance -> 0 <= relaxation ->
  0 < gaussian_coherence relaxation (gaussian_phase_variance dimension covariance weights) <= 1.
Proof.
  intros dimension covariance weights relaxation covariance_positive relaxation_positive.
  pose proof (psd_phase_variance_nonnegative dimension covariance weights covariance_positive).
  unfold gaussian_coherence. apply exp_nonpositive_range. lra.
Qed.

Theorem gaussian_variance_monotone_coherence : forall relaxation first_variance second_variance,
  first_variance <= second_variance ->
  gaussian_coherence relaxation second_variance <= gaussian_coherence relaxation first_variance.
Proof.
  intros relaxation first_variance second_variance ordered.
  destruct (Rle_lt_or_eq_dec first_variance second_variance ordered) as [strict|equal].
  - unfold gaussian_coherence.
    assert (-relaxation - second_variance / 2 < -relaxation - first_variance / 2) by lra.
    left. apply exp_increasing. assumption.
  - rewrite equal. lra.
Qed.

Lemma form_zero_weights : forall dimension covariance,
  quadratic_form dimension covariance (fun _ => 0) (fun _ => 0) = 0.
Proof.
  intros dimension covariance. unfold quadratic_form.
  assert (zero_sum : forall count, finite_sum count (fun _ => 0) = 0).
  { intro count. induction count; simpl; lra. }
  transitivity (finite_sum dimension (fun _ => 0)).
  - apply finite_sum_ext. intro row.
    transitivity (finite_sum dimension (fun _ => 0)).
    + apply finite_sum_ext. intro column. ring.
    + apply zero_sum.
  - apply zero_sum.
Qed.

Theorem zero_coupling_relaxation_only : forall dimension covariance relaxation,
  gaussian_coherence relaxation (gaussian_phase_variance dimension covariance (fun _ => 0)) = exp (-relaxation).
Proof. intros. unfold gaussian_phase_variance, gaussian_coherence. rewrite form_zero_weights. apply f_equal. lra. Qed.

Definition quasistatic_covariance (variance : R) : nat -> nat -> R := fun _ _ => variance.

Theorem quasistatic_phase_variance : forall dimension variance weights,
  gaussian_phase_variance dimension (quasistatic_covariance variance) weights =
  variance * (finite_sum dimension weights)^2.
Proof.
  intros dimension variance weights.
  unfold gaussian_phase_variance, quadratic_form, quasistatic_covariance.
  transitivity (finite_sum dimension (fun row => weights row * variance * finite_sum dimension weights)).
  - apply finite_sum_ext. intro row. apply finite_sum_scale.
  - transitivity (finite_sum dimension (fun row =>
      (variance * finite_sum dimension weights) * weights row)).
    + apply finite_sum_ext. intro row. ring.
    + rewrite finite_sum_scale. ring.
Qed.

Theorem quasistatic_covariance_psd : forall dimension variance,
  0 <= variance -> form_psd dimension (quasistatic_covariance variance).
Proof.
  intros dimension variance positive weights.
  change (0 <= gaussian_phase_variance dimension (quasistatic_covariance variance) weights).
  rewrite quasistatic_phase_variance. apply Rmult_le_pos; [assumption|nra].
Qed.

Theorem ideal_echo_cancels_quasistatic_phase : forall dimension variance weights relaxation,
  finite_sum dimension weights = 0 ->
  gaussian_coherence relaxation
    (gaussian_phase_variance dimension (quasistatic_covariance variance) weights) = exp (-relaxation).
Proof.
  intros dimension variance weights relaxation balanced.
  rewrite quasistatic_phase_variance, balanced.
  unfold gaussian_coherence. apply f_equal. nra.
Qed.

Theorem positive_variance_ramsey_has_extra_decay : forall dimension variance weights relaxation,
  0 < variance -> finite_sum dimension weights <> 0 ->
  gaussian_coherence relaxation
    (gaussian_phase_variance dimension (quasistatic_covariance variance) weights) < exp (-relaxation).
Proof.
  intros dimension variance weights relaxation positive nonzero.
  rewrite quasistatic_phase_variance. unfold gaussian_coherence.
  apply exp_increasing. assert (0 < (finite_sum dimension weights)^2) by nra. nra.
Qed.

(** A finite qubit phase channel has a normalized Choi matrix supported on
    |00> and |11>. The two middle computational directions have eigenvalue
    zero. The supplied matrix and its Bell decomposition are proved here;
    general Choi equivalence and a quantum-bath derivation remain separate. *)
From OpenGororoba Require Import CayleyDicksonAlgebra FiniteComplexPassivity VacuumCounterexamples.

Definition dephasing_identity_weight factor := (1 + factor) / 2.
Definition dephasing_phase_weight factor := (1 - factor) / 2.

Theorem dephasing_weights_are_probabilities : forall factor,
  -1 <= factor <= 1 ->
  0 <= dephasing_identity_weight factor /\
  0 <= dephasing_phase_weight factor /\
  dephasing_identity_weight factor + dephasing_phase_weight factor = 1.
Proof. intros; unfold dephasing_identity_weight, dephasing_phase_weight; repeat split; lra. Qed.

Definition normalized_dephasing_choi_form factor
  (first second third fourth : CDComplex) :=
  (complex_norm_sq first + complex_norm_sq fourth +
   2 * factor * complex_real_pair first fourth) / 2.

Theorem normalized_dephasing_choi_bell_decomposition : forall factor first second third fourth,
  normalized_dephasing_choi_form factor first second third fourth =
  dephasing_identity_weight factor / 2 * complex_norm_sq (complex_add first fourth) +
  dephasing_phase_weight factor / 2 * complex_norm_sq (complex_difference first fourth).
Proof.
  intros factor [first_real first_imaginary] [second_real second_imaginary]
    [third_real third_imaginary] [fourth_real fourth_imaginary].
  unfold normalized_dephasing_choi_form, dephasing_identity_weight,
    dephasing_phase_weight, complex_real_pair, complex_norm_sq,
    complex_difference, complex_add, complex_scale; simpl; field; ring.
Qed.

Theorem normalized_dephasing_choi_psd : forall factor,
  -1 <= factor <= 1 -> forall first second third fourth,
  0 <= normalized_dephasing_choi_form factor first second third fourth.
Proof.
  intros factor interval first second third fourth.
  pose proof (dephasing_weights_are_probabilities factor interval) as [identity_positive [phase_positive total]].
  rewrite normalized_dephasing_choi_bell_decomposition.
  pose proof (complex_probability_nonnegative (fun _ => complex_add first fourth) 0) as plus_positive.
  pose proof (complex_probability_nonnegative (fun _ => complex_difference first fourth) 0) as minus_positive.
  unfold state_probability in *.
  assert (0 <= dephasing_identity_weight factor / 2 * complex_norm_sq (complex_add first fourth)) by
    (apply Rmult_le_pos; [lra|assumption]).
  assert (0 <= dephasing_phase_weight factor / 2 * complex_norm_sq (complex_difference first fourth)) by
    (apply Rmult_le_pos; [lra|assumption]). lra.
Qed.

Theorem normalized_dephasing_choi_middle_directions_zero : forall factor second third,
  normalized_dephasing_choi_form factor complex_zero second third complex_zero = 0.
Proof.
  intros; unfold normalized_dephasing_choi_form, complex_norm_sq, complex_real_pair, complex_zero;
    simpl; field; ring.
Qed.

Theorem gaussian_phase_weights_are_probabilities : forall variance,
  0 <= variance ->
  0 <= dephasing_identity_weight (exp (-variance/2)) /\
  0 <= dephasing_phase_weight (exp (-variance/2)) /\
  dephasing_identity_weight (exp (-variance/2)) +
  dephasing_phase_weight (exp (-variance/2)) = 1.
Proof.
  intros variance positive. apply dephasing_weights_are_probabilities.
  pose proof (exp_nonpositive_range (-variance/2) ltac:(lra)); lra.
Qed.

Theorem gaussian_phase_normalized_choi_psd : forall variance,
  0 <= variance -> forall first second third fourth,
  0 <= normalized_dephasing_choi_form (exp (-variance/2)) first second third fourth.
Proof.
  intros variance positive. apply normalized_dephasing_choi_psd.
  pose proof (exp_nonpositive_range (-variance/2) ltac:(lra)); lra.
Qed.
