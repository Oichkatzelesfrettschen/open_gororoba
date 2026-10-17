(** Joint complex amplitudes permit arbitrary finite entanglement.
    No factorization of the joint state is assumed. The Hamiltonian spectrum
    is explicitly additive in two orthonormal subsystem eigenbases. *)
From Stdlib Require Import Reals Psatz Lia.
From OpenGororoba Require Import CayleyDicksonAlgebra PassiveMechanicalMemory FiniteComplexPassivity.
Open Scope R_scope.

Definition joint_complex_state := nat -> complex_state.
Definition joint_norm first_dimension second_dimension (state : joint_complex_state) :=
  finite_sum first_dimension (fun first => state_norm second_dimension (state first)).
Definition joint_energy first_dimension second_dimension first_energies second_energies
  (state : joint_complex_state) :=
  finite_sum first_dimension (fun first => spectral_energy second_dimension
    (fun second => first_energies first + second_energies second) (state first)).

Theorem joint_spectral_lower_bound : forall first_dimension second_dimension first_energies second_energies state first_ground second_ground,
  (forall index, (index < first_dimension)%nat -> first_ground <= first_energies index) ->
  (forall index, (index < second_dimension)%nat -> second_ground <= second_energies index) ->
  (first_ground + second_ground) * joint_norm first_dimension second_dimension state <=
  joint_energy first_dimension second_dimension first_energies second_energies state.
Proof.
  induction first_dimension; intros second_dimension first_energies second_energies state first_ground second_ground first_lower second_lower.
  - unfold joint_norm, joint_energy. simpl. lra.
  - assert (previous : (first_ground + second_ground) * joint_norm first_dimension second_dimension state <=
      joint_energy first_dimension second_dimension first_energies second_energies state).
    { apply IHfirst_dimension; [intros; apply first_lower; lia|assumption]. }
    assert (last : (first_ground + second_ground) * state_norm second_dimension (state first_dimension) <=
      spectral_energy second_dimension (fun second => first_energies first_dimension + second_energies second) (state first_dimension)).
    { apply spectral_energy_lower_bound. intros index bounded.
      pose proof (first_lower first_dimension ltac:(lia)).
      pose proof (second_lower index bounded). lra. }
    unfold joint_norm, joint_energy in *; simpl; lra.
Qed.

Theorem normalized_joint_ground_bound : forall first_dimension second_dimension first_energies second_energies state first_ground second_ground,
  joint_norm first_dimension second_dimension state = 1 ->
  (forall index, (index < first_dimension)%nat -> first_ground <= first_energies index) ->
  (forall index, (index < second_dimension)%nat -> second_ground <= second_energies index) ->
  first_ground + second_ground <= joint_energy first_dimension second_dimension first_energies second_energies state.
Proof.
  intros first_dimension second_dimension first_energies second_energies state first_ground second_ground normalized first_lower second_lower.
  pose proof (joint_spectral_lower_bound first_dimension second_dimension first_energies second_energies state first_ground second_ground first_lower second_lower).
  rewrite normalized in H. lra.
Qed.

Theorem joint_ground_state_cyclic_passive : forall first_dimension second_dimension first_energies second_energies initial final first_ground second_ground,
  joint_norm first_dimension second_dimension initial = 1 ->
  joint_norm first_dimension second_dimension final = joint_norm first_dimension second_dimension initial ->
  (forall index, (index < first_dimension)%nat -> first_ground <= first_energies index) ->
  (forall index, (index < second_dimension)%nat -> second_ground <= second_energies index) ->
  joint_energy first_dimension second_dimension first_energies second_energies initial = first_ground + second_ground ->
  joint_energy first_dimension second_dimension first_energies second_energies initial -
    joint_energy first_dimension second_dimension first_energies second_energies final <= 0.
Proof.
  intros first_dimension second_dimension first_energies second_energies initial final first_ground second_ground normalized preserved first_lower second_lower initial_ground.
  rewrite normalized in preserved.
  pose proof (normalized_joint_ground_bound first_dimension second_dimension first_energies second_energies final first_ground second_ground preserved first_lower second_lower). lra.
Qed.

Definition correlated_two_level_state (first second : nat) : CDComplex :=
  match first, second with
  | O, O => {| cre := 3/5; cim := 0 |}
  | S O, S O => {| cre := 4/5; cim := 0 |}
  | _, _ => complex_zero
  end.

Theorem correlated_state_normalized : joint_norm 2 2 correlated_two_level_state = 1.
Proof.
  unfold joint_norm, state_norm, state_probability, correlated_two_level_state, complex_norm_sq, complex_zero.
  simpl. field.
Qed.

Theorem correlated_state_nonzero_minor :
  cre (correlated_two_level_state 0 0) * cre (correlated_two_level_state 1 1) -
  cre (correlated_two_level_state 0 1) * cre (correlated_two_level_state 1 0) = 12/25.
Proof. unfold correlated_two_level_state, complex_zero; simpl; field; ring. Qed.

Lemma complex_product_minor_vanishes : forall first second third fourth,
  complex_mul (complex_mul first second) (complex_mul third fourth) =
  complex_mul (complex_mul first fourth) (complex_mul third second).
Proof.
  intros first second third fourth.
  repeat rewrite complex_mul_assoc. apply f_equal.
  rewrite (complex_mul_comm second (complex_mul third fourth)).
  rewrite complex_mul_assoc.
  rewrite (complex_mul_comm third (complex_mul fourth second)).
  rewrite complex_mul_assoc, (complex_mul_comm second third). reflexivity.
Qed.

Theorem correlated_state_cannot_factor : forall left right : complex_state,
  ~ (forall first second, (first < 2)%nat -> (second < 2)%nat ->
    correlated_two_level_state first second = complex_mul (left first) (right second)).
Proof.
  intros left right factorization.
  pose proof (complex_product_minor_vanishes (left 0%nat) (right 0%nat) (left 1%nat) (right 1%nat)) as minor.
  rewrite <- (factorization 0%nat 0%nat ltac:(lia) ltac:(lia)),
    <- (factorization 1%nat 1%nat ltac:(lia) ltac:(lia)),
    <- (factorization 0%nat 1%nat ltac:(lia) ltac:(lia)),
    <- (factorization 1%nat 0%nat ltac:(lia) ltac:(lia)) in minor.
  apply (f_equal cre) in minor.
  unfold correlated_two_level_state, complex_mul, complex_zero in minor; simpl in minor. lra.
Qed.
