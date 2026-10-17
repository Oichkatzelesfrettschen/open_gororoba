(** Finite complex amplitudes and diagonal Hamiltonian work bounds.
    Spectral energies are supplied in an orthonormal eigenbasis. A map's
    norm-preservation premise is explicit; physical realizability, spectral
    diagonalization, and floating-point correspondence remain separate. *)
From Stdlib Require Import Reals Psatz Lia.
From OpenGororoba Require Import CayleyDicksonAlgebra PassiveMechanicalMemory.
Open Scope R_scope.

Definition complex_state := nat -> CDComplex.
Definition state_probability (state : complex_state) index := complex_norm_sq (state index).
Definition state_norm dimension state := finite_sum dimension (state_probability state).
Definition normalized_state dimension state := state_norm dimension state = 1.
Definition spectral_energy dimension (energies : nat -> R) state :=
  finite_sum dimension (fun index => energies index * state_probability state index).
Definition preserves_norm source_dimension target_dimension (transform : complex_state -> complex_state) :=
  forall state, state_norm target_dimension (transform state) = state_norm source_dimension state.

Lemma complex_probability_nonnegative : forall state index,
  0 <= state_probability state index.
Proof. intros; unfold state_probability, complex_norm_sq; nra. Qed.

Theorem complex_state_norm_nonnegative : forall dimension state,
  0 <= state_norm dimension state.
Proof. intros; apply finite_sum_nonnegative; intros; apply complex_probability_nonnegative. Qed.

Theorem spectral_energy_lower_bound : forall dimension energies state ground,
  (forall index, (index < dimension)%nat -> ground <= energies index) ->
  ground * state_norm dimension state <= spectral_energy dimension energies state.
Proof.
  induction dimension; intros energies state ground lower; simpl; [unfold state_norm, spectral_energy; simpl; lra|].
  assert (prior : ground * state_norm dimension state <= spectral_energy dimension energies state).
  { apply IHdimension. intros; apply lower; lia. }
  assert (last : ground <= energies dimension) by (apply lower; lia).
  pose proof (complex_probability_nonnegative state dimension).
  unfold state_norm, spectral_energy in *; simpl; nra.
Qed.

Corollary normalized_spectral_lower_bound : forall dimension energies state ground,
  normalized_state dimension state ->
  (forall index, (index < dimension)%nat -> ground <= energies index) ->
  ground <= spectral_energy dimension energies state.
Proof.
  intros dimension energies state ground normalized lower.
  pose proof (spectral_energy_lower_bound dimension energies state ground lower).
  unfold normalized_state in normalized. rewrite normalized in H. lra.
Qed.

Theorem isometry_preserves_normalization : forall source_dimension target_dimension transform state,
  preserves_norm source_dimension target_dimension transform ->
  normalized_state source_dimension state -> normalized_state target_dimension (transform state).
Proof. intros; unfold normalized_state; rewrite H; assumption. Qed.

Theorem isometry_composition : forall first_dimension middle_dimension last_dimension first second,
  preserves_norm first_dimension middle_dimension first ->
  preserves_norm middle_dimension last_dimension second ->
  preserves_norm first_dimension last_dimension (fun state => second (first state)).
Proof. intros first_dimension middle_dimension last_dimension first second first_norm second_norm state.
  rewrite second_norm, first_norm. reflexivity. Qed.

Theorem cyclic_available_work_bound : forall dimension energies state transform ground,
  normalized_state dimension state -> preserves_norm dimension dimension transform ->
  (forall index, (index < dimension)%nat -> ground <= energies index) ->
  spectral_energy dimension energies state - spectral_energy dimension energies (transform state)
  <= spectral_energy dimension energies state - ground.
Proof.
  intros dimension energies state transform ground normalized isometry lower.
  assert (ground <= spectral_energy dimension energies (transform state)).
  { apply normalized_spectral_lower_bound; [eapply isometry_preserves_normalization; eauto|assumption]. }
  lra.
Qed.

Corollary cyclic_ground_state_passive : forall dimension energies state transform ground,
  normalized_state dimension state -> preserves_norm dimension dimension transform ->
  (forall index, (index < dimension)%nat -> ground <= energies index) ->
  spectral_energy dimension energies state = ground ->
  spectral_energy dimension energies state - spectral_energy dimension energies (transform state) <= 0.
Proof. intros; pose proof (cyclic_available_work_bound dimension energies state transform ground H H0 H1); lra. Qed.

Theorem spectral_common_offset : forall dimension energies state offset,
  spectral_energy dimension (fun index => energies index + offset) state =
  spectral_energy dimension energies state + offset * state_norm dimension state.
Proof.
  intros; unfold spectral_energy, state_norm. rewrite <- finite_sum_scale, <- finite_sum_add.
  apply finite_sum_ext. intro index. ring.
Qed.

Theorem cyclic_work_offset_invariant : forall dimension energies state transform offset,
  preserves_norm dimension dimension transform ->
  spectral_energy dimension (fun index => energies index + offset) state -
  spectral_energy dimension (fun index => energies index + offset) (transform state) =
  spectral_energy dimension energies state - spectral_energy dimension energies (transform state).
Proof. intros; repeat rewrite spectral_common_offset. rewrite H. ring. Qed.

Lemma finite_sum_scale_right : forall dimension scalar term,
  finite_sum dimension (fun index => term index * scalar) = finite_sum dimension term * scalar.
Proof. induction dimension; intros; simpl; [ring|rewrite IHdimension; ring]. Qed.

(** A tensor product uses two bounded indices, avoiding flattened-index
    arithmetic. Both subsystem dimensions remain arbitrary natural numbers. *)
Definition product_norm left_dimension right_dimension left right :=
  finite_sum left_dimension (fun left_index => finite_sum right_dimension
    (fun right_index => complex_norm_sq (complex_mul (left left_index) (right right_index)))).
Definition product_energy left_dimension right_dimension left_energies right_energies left right :=
  finite_sum left_dimension (fun left_index => finite_sum right_dimension
    (fun right_index => (left_energies left_index + right_energies right_index) *
      complex_norm_sq (complex_mul (left left_index) (right right_index)))).

Theorem tensor_norm_factorization : forall left_dimension right_dimension left right,
  product_norm left_dimension right_dimension left right =
  state_norm left_dimension left * state_norm right_dimension right.
Proof.
  intros; unfold product_norm, state_norm, state_probability.
  transitivity (finite_sum left_dimension (fun left_index =>
    complex_norm_sq (left left_index) * finite_sum right_dimension
      (fun right_index => complex_norm_sq (right right_index)))).
  - apply finite_sum_ext. intro left_index. rewrite <- finite_sum_scale.
    apply finite_sum_ext. intro right_index. apply complex_norm_mul.
  - apply finite_sum_scale_right.
Qed.

Theorem tensor_additive_energy : forall left_dimension right_dimension left_energies right_energies left right,
  product_energy left_dimension right_dimension left_energies right_energies left right =
  spectral_energy left_dimension left_energies left * state_norm right_dimension right +
  state_norm left_dimension left * spectral_energy right_dimension right_energies right.
Proof.
  intros; unfold product_energy, spectral_energy, state_norm, state_probability.
  transitivity (finite_sum left_dimension (fun left_index =>
    (left_energies left_index * complex_norm_sq (left left_index)) *
      finite_sum right_dimension (fun right_index => complex_norm_sq (right right_index)) +
    complex_norm_sq (left left_index) *
      finite_sum right_dimension (fun right_index => right_energies right_index * complex_norm_sq (right right_index)))).
  - apply finite_sum_ext. intro left_index.
    rewrite <- finite_sum_scale, <- finite_sum_scale, <- finite_sum_add.
    apply finite_sum_ext. intro right_index. rewrite complex_norm_mul. ring.
  - rewrite finite_sum_add, finite_sum_scale_right, finite_sum_scale_right. ring.
Qed.

Corollary tensor_ground_energy_lower_bound : forall left_dimension right_dimension left_energies right_energies left right left_ground right_ground,
  normalized_state left_dimension left -> normalized_state right_dimension right ->
  (forall index, (index < left_dimension)%nat -> left_ground <= left_energies index) ->
  (forall index, (index < right_dimension)%nat -> right_ground <= right_energies index) ->
  left_ground + right_ground <=
  product_energy left_dimension right_dimension left_energies right_energies left right.
Proof.
  intros left_dimension right_dimension left_energies right_energies left right left_ground right_ground left_normal right_normal left_bound right_bound.
  rewrite tensor_additive_energy. unfold normalized_state in *.
  rewrite left_normal, right_normal.
  pose proof (normalized_spectral_lower_bound left_dimension left_energies left left_ground left_normal left_bound).
  pose proof (normalized_spectral_lower_bound right_dimension right_energies right right_ground right_normal right_bound). lra.
Qed.

Lemma finite_sum_split : forall retained tail term,
  finite_sum (retained + tail) term = finite_sum retained term +
  finite_sum tail (fun index => term (retained + index)%nat).
Proof. intros retained tail; induction tail; intros; [rewrite Nat.add_0_r; simpl; ring|].
  rewrite Nat.add_succ_r. simpl. rewrite IHtail. ring. Qed.

Theorem truncation_probability_and_energy : forall retained tail energies state tail_ceiling,
  normalized_state (retained + tail) state ->
  0 <= tail_ceiling ->
  (forall index, (index < tail)%nat -> 0 <= energies (retained + index)%nat <= tail_ceiling) ->
  0 <= 1 - state_norm retained state /\
  0 <= spectral_energy (retained + tail) energies state - spectral_energy retained energies state <=
    tail_ceiling * (1 - state_norm retained state).
Proof.
  intros retained tail energies state tail_ceiling normalized ceiling_positive tail_bounds.
  assert (mass_split : state_norm (retained + tail) state = state_norm retained state +
    finite_sum tail (fun index => state_probability state (retained + index)%nat)).
  { unfold state_norm. apply finite_sum_split. }
  assert (tail_mass_nonnegative : 0 <= finite_sum tail (fun index => state_probability state (retained + index)%nat)).
  { apply finite_sum_nonnegative. intros. apply complex_probability_nonnegative. }
  assert (energy_split : spectral_energy (retained + tail) energies state = spectral_energy retained energies state +
    finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat)).
  { unfold spectral_energy. apply finite_sum_split. }
  assert (energy_nonnegative : 0 <= finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat)).
  { apply finite_sum_nonnegative. intros index bounded. apply Rmult_le_pos; [apply tail_bounds; assumption|apply complex_probability_nonnegative]. }
  assert (tail_upper : finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat) <=
    tail_ceiling * finite_sum tail (fun index => state_probability state (retained + index)%nat)).
  { assert (difference_positive : 0 <= finite_sum tail (fun index =>
       (tail_ceiling - energies (retained + index)%nat) * state_probability state (retained + index)%nat)).
    { apply finite_sum_nonnegative. intros index bounded. apply Rmult_le_pos; [pose proof (tail_bounds index bounded); lra|apply complex_probability_nonnegative]. }
    replace (finite_sum tail (fun index => (tail_ceiling - energies (retained + index)%nat) * state_probability state (retained + index)%nat)) with
      (tail_ceiling * finite_sum tail (fun index => state_probability state (retained + index)%nat) -
       finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat)) in difference_positive.
    - lra.
    - rewrite <- finite_sum_scale. unfold Rminus.
      replace (- finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat)) with
        ((-1) * finite_sum tail (fun index => energies (retained + index)%nat * state_probability state (retained + index)%nat)) by ring.
      rewrite <- finite_sum_scale, <- finite_sum_add. apply finite_sum_ext. intro index. ring. }
  unfold normalized_state in normalized. rewrite normalized in mass_split. split; [lra|]. split; nra.
Qed.

(** A concrete diagonal complex phase operation witnesses the isometry
    premise without introducing matrix or physical implementation axioms. *)
Theorem unit_modulus_phase_isometry : forall dimension phase,
  complex_norm_sq phase = 1 ->
  preserves_norm dimension dimension
    (fun state index => complex_mul phase (state index)).
Proof.
  intros dimension phase unit_phase state; unfold state_norm, state_probability.
  apply finite_sum_ext. intro index. rewrite complex_norm_mul, unit_phase. ring.
Qed.

Corollary trigonometric_phase_isometry : forall dimension angle,
  preserves_norm dimension dimension
    (fun state index => complex_mul (mkComplex (cos angle) (sin angle)) (state index)).
Proof.
  intros dimension angle. apply unit_modulus_phase_isometry.
  unfold complex_norm_sq; simpl.
  pose proof (sin2_cos2 angle). unfold Rsqr in H. nra.
Qed.
