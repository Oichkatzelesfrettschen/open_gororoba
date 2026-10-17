(** Finite-dimensional quadratic storage with coupled passive memory.
    Real arithmetic and exact bilinear symmetry/positivity are premises.
    The directional storage coefficient follows from the exact quadratic
    increment. Numerical matrix validation and trajectory integration are
    separate obligations. *)
From Stdlib Require Import Reals Psatz Lia.
Open Scope R_scope.

Fixpoint finite_sum (dimension : nat) (term : nat -> R) : R :=
  match dimension with
  | O => 0
  | S previous => finite_sum previous term + term previous
  end.

Lemma finite_sum_ext : forall dimension first second,
  (forall index, first index = second index) ->
  finite_sum dimension first = finite_sum dimension second.
Proof. induction dimension; intros first second equal_terms; simpl; [reflexivity|rewrite (IHdimension first second equal_terms), equal_terms; reflexivity]. Qed.

Lemma finite_sum_add : forall dimension first second,
  finite_sum dimension (fun index => first index + second index) =
  finite_sum dimension first + finite_sum dimension second.
Proof. induction dimension; intros; simpl; [ring|rewrite IHdimension; ring]. Qed.

Lemma finite_sum_scale : forall dimension scalar term,
  finite_sum dimension (fun index => scalar * term index) = scalar * finite_sum dimension term.
Proof. induction dimension; intros; simpl; [ring|rewrite IHdimension; ring]. Qed.

Lemma finite_sum_nonnegative : forall dimension term,
  (forall index, (index < dimension)%nat -> 0 <= term index) ->
  0 <= finite_sum dimension term.
Proof.
  induction dimension; intros term nonnegative; simpl; [lra|].
  assert (0 <= finite_sum dimension term) by (apply IHdimension; intros; apply nonnegative; lia).
  assert (0 <= term dimension) by (apply nonnegative; lia). lra.
Qed.

Definition quadratic_form dimension (matrix : nat -> nat -> R) (left right : nat -> R) :=
  finite_sum dimension (fun row => finite_sum dimension
    (fun column => left row * matrix row column * right column)).
Definition form_symmetric dimension matrix := forall left right,
  quadratic_form dimension matrix left right = quadratic_form dimension matrix right left.
Definition form_psd dimension matrix := forall vector,
  0 <= quadratic_form dimension matrix vector vector.

Lemma form_right_add : forall dimension matrix left first second,
  quadratic_form dimension matrix left (fun index => first index + second index) =
  quadratic_form dimension matrix left first + quadratic_form dimension matrix left second.
Proof.
  intros. unfold quadratic_form.
  transitivity (finite_sum dimension (fun row =>
    finite_sum dimension (fun column => left row * matrix row column * first column) +
    finite_sum dimension (fun column => left row * matrix row column * second column))).
  - apply finite_sum_ext. intro row. rewrite <- finite_sum_add.
    apply finite_sum_ext. intro column. ring.
  - apply finite_sum_add.
Qed.

Lemma form_right_scale : forall dimension matrix left scalar right,
  quadratic_form dimension matrix left (fun index => scalar * right index) =
  scalar * quadratic_form dimension matrix left right.
Proof.
  intros. unfold quadratic_form. rewrite <- finite_sum_scale.
  apply finite_sum_ext. intro row. rewrite <- finite_sum_scale.
  apply finite_sum_ext. intro column. ring.
Qed.

Lemma form_left_add : forall dimension matrix first second right,
  quadratic_form dimension matrix (fun index => first index + second index) right =
  quadratic_form dimension matrix first right + quadratic_form dimension matrix second right.
Proof.
  intros. unfold quadratic_form. rewrite <- finite_sum_add.
  apply finite_sum_ext. intro row. rewrite <- finite_sum_add.
  apply finite_sum_ext. intro column. ring.
Qed.

Lemma form_left_scale : forall dimension matrix scalar left right,
  quadratic_form dimension matrix (fun index => scalar * left index) right =
  scalar * quadratic_form dimension matrix left right.
Proof.
  intros. unfold quadratic_form. rewrite <- finite_sum_scale.
  apply finite_sum_ext. intro row. rewrite <- finite_sum_scale.
  apply finite_sum_ext. intro column. ring.
Qed.

Theorem symmetric_quadratic_increment : forall dimension matrix state direction step,
  form_symmetric dimension matrix ->
  quadratic_form dimension matrix
    (fun index => state index + step * direction index)
    (fun index => state index + step * direction index) -
  quadratic_form dimension matrix state state =
  2 * step * quadratic_form dimension matrix state direction +
  step^2 * quadratic_form dimension matrix direction direction.
Proof.
  intros dimension matrix state direction step symmetric.
  rewrite form_left_add, form_left_scale.
  repeat rewrite form_right_add. repeat rewrite form_right_scale.
  rewrite (symmetric direction state). ring.
Qed.

Theorem auxiliary_memory_balance : forall dimension matrix state velocity time_constant,
  form_symmetric dimension matrix -> 0 < time_constant ->
  quadratic_form dimension matrix state
    (fun index => velocity index + (- /time_constant) * state index) =
  quadratic_form dimension matrix velocity state -
  quadratic_form dimension matrix state state / time_constant.
Proof.
  intros dimension matrix state velocity time_constant symmetric positive_time.
  rewrite form_right_add, form_right_scale, (symmetric state velocity).
  unfold Rdiv. ring.
Qed.

Definition memory_dissipation dimension branches matrices states times :=
  finite_sum branches (fun branch =>
    quadratic_form dimension (matrices branch) (states branch) (states branch) / times branch).

Theorem memory_dissipation_nonnegative : forall dimension branches matrices states times,
  (forall branch, (branch < branches)%nat -> form_psd dimension (matrices branch)) ->
  (forall branch, (branch < branches)%nat -> 0 < times branch) ->
  0 <= memory_dissipation dimension branches matrices states times.
Proof.
  intros dimension branches matrices states times positive_forms positive_times.
  unfold memory_dissipation. apply finite_sum_nonnegative. intros branch bounded.
  unfold Rdiv. apply Rmult_le_pos.
  - apply (positive_forms branch bounded).
  - left. apply Rinv_0_lt_compat. apply positive_times; assumption.
Qed.

Theorem coupled_mechanical_storage_balance : forall dimension branches mass support damping
  matrices states times position velocity acceleration supplied_power,
  form_symmetric dimension mass -> form_symmetric dimension support ->
  (forall branch, form_symmetric dimension (matrices branch)) ->
  (forall branch, 0 < times branch) ->
  supplied_power =
    quadratic_form dimension mass velocity acceleration +
    quadratic_form dimension support velocity position +
    quadratic_form dimension damping velocity velocity +
    finite_sum branches (fun branch =>
      quadratic_form dimension (matrices branch) velocity (states branch)) ->
  quadratic_form dimension mass velocity acceleration +
  quadratic_form dimension support position velocity +
  finite_sum branches (fun branch =>
    quadratic_form dimension (matrices branch) (states branch)
      (fun index => velocity index + (- /times branch) * states branch index)) =
  supplied_power - quadratic_form dimension damping velocity velocity -
  memory_dissipation dimension branches matrices states times.
Proof.
  intros dimension branches mass support damping matrices states times position velocity
    acceleration supplied_power mass_symmetric support_symmetric memory_symmetric positive_times balance.
  assert (memory_balance :
    finite_sum branches (fun branch =>
      quadratic_form dimension (matrices branch) (states branch)
        (fun index => velocity index + (- /times branch) * states branch index)) =
    finite_sum branches (fun branch =>
      quadratic_form dimension (matrices branch) velocity (states branch)) -
    memory_dissipation dimension branches matrices states times).
  { unfold memory_dissipation. unfold Rminus. replace (- finite_sum branches (fun branch => quadratic_form dimension (matrices branch) (states branch) (states branch) / times branch)) with ((-1) * finite_sum branches (fun branch => quadratic_form dimension (matrices branch) (states branch) (states branch) / times branch)) by ring.
    rewrite <- finite_sum_scale, <- finite_sum_add.
    apply finite_sum_ext. intro branch.
    rewrite auxiliary_memory_balance; [ring|apply memory_symmetric|apply positive_times]. }
  rewrite memory_balance, (support_symmetric position velocity). lra.
Qed.

Theorem resource_closed_supply_bound : forall initial_storage final_storage supplied dissipated,
  0 <= dissipated -> final_storage - initial_storage = supplied - dissipated ->
  final_storage - initial_storage <= supplied.
Proof. intros; lra. Qed.

Theorem conservative_cycle_cancels : forall initial_potential final_potential,
  (initial_potential - final_potential) + (final_potential - initial_potential) = 0.
Proof. intros; ring. Qed.

Definition coupled_storage dimension branches mass support matrices position velocity states :=
  quadratic_form dimension mass velocity velocity / 2 +
  quadratic_form dimension support position position / 2 +
  finite_sum branches (fun branch =>
    quadratic_form dimension (matrices branch) (states branch) (states branch) / 2).

Theorem coupled_storage_nonnegative : forall dimension branches mass support matrices position velocity states,
  form_psd dimension mass -> form_psd dimension support ->
  (forall branch, (branch < branches)%nat -> form_psd dimension (matrices branch)) ->
  0 <= coupled_storage dimension branches mass support matrices position velocity states.
Proof.
  intros dimension branches mass support matrices position velocity states mass_positive support_positive memory_positive.
  unfold coupled_storage.
  pose proof (mass_positive velocity). pose proof (support_positive position).
  assert (0 <= finite_sum branches (fun branch =>
    quadratic_form dimension (matrices branch) (states branch) (states branch) / 2)).
  { apply finite_sum_nonnegative. intros branch bounded.
    pose proof (memory_positive branch bounded (states branch)). lra. }
  lra.
Qed.

Theorem dissipative_storage_coefficient_is_supply_bounded : forall dimension branches damping matrices states times velocity supplied coefficient,
  form_psd dimension damping ->
  (forall branch, (branch < branches)%nat -> form_psd dimension (matrices branch)) ->
  (forall branch, (branch < branches)%nat -> 0 < times branch) ->
  coefficient = supplied - quadratic_form dimension damping velocity velocity -
    memory_dissipation dimension branches matrices states times ->
  coefficient <= supplied.
Proof.
  intros dimension branches damping matrices states times velocity supplied coefficient damping_positive memory_positive positive_times balance.
  pose proof (damping_positive velocity).
  pose proof (memory_dissipation_nonnegative dimension branches matrices states times memory_positive positive_times). lra.
Qed.
