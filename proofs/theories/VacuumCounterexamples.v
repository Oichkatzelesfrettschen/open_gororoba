(** Explicit finite Choi quadratic-form test for the qubit depolarizing map.
    The matrix is the normalized Choi state of rho |-> parameter*rho +
    (1-parameter)*I*trace(rho)/2 in the computational basis. The theorem
    proves positive semidefiniteness of that supplied matrix. Constructing
    the channel from a memory equation and the general Choi equivalence
    theorem remain separate obligations. *)
From Stdlib Require Import Reals Psatz.
From OpenGororoba Require Import CayleyDicksonAlgebra FiniteComplexPassivity.
Open Scope R_scope.

Definition complex_difference first second :=
  complex_add first (complex_scale (-1) second).
Definition complex_real_pair first second :=
  cre first * cre second + cim first * cim second.

Definition depolarizing_choi_form parameter first second third fourth :=
  ((1 + parameter) * (complex_norm_sq first + complex_norm_sq fourth) +
   (1 - parameter) * (complex_norm_sq second + complex_norm_sq third) +
   4 * parameter * complex_real_pair first fourth) / 4.
Definition depolarizing_choi_psd parameter := forall first second third fourth,
  0 <= depolarizing_choi_form parameter first second third fourth.

Theorem depolarizing_choi_bell_decomposition : forall parameter first second third fourth,
  depolarizing_choi_form parameter first second third fourth =
  (1 + 3 * parameter) / 8 * complex_norm_sq (complex_add first fourth) +
  (1 - parameter) / 8 * complex_norm_sq (complex_difference first fourth) +
  (1 - parameter) / 4 * (complex_norm_sq second + complex_norm_sq third).
Proof.
  intros parameter [first_real first_imaginary] [second_real second_imaginary]
    [third_real third_imaginary] [fourth_real fourth_imaginary].
  unfold depolarizing_choi_form, complex_real_pair, complex_norm_sq,
    complex_difference, complex_add, complex_scale; simpl; field; ring.
Qed.

Theorem depolarizing_choi_psd_interval : forall parameter,
  depolarizing_choi_psd parameter <-> -1/3 <= parameter <= 1.
Proof.
  intro parameter; split.
  - intro positive.
    pose proof (positive complex_zero complex_one complex_zero complex_zero) as upper.
    pose proof (positive complex_one complex_zero complex_zero complex_one) as lower.
    unfold depolarizing_choi_form, complex_norm_sq, complex_real_pair,
      complex_zero, complex_one in upper, lower; simpl in upper, lower; split; nra.
  - intros [lower upper] first second third fourth.
    rewrite depolarizing_choi_bell_decomposition.
    pose proof (complex_probability_nonnegative (fun _ => complex_add first fourth) 0).
    pose proof (complex_probability_nonnegative (fun _ => complex_difference first fourth) 0).
    pose proof (complex_probability_nonnegative (fun _ => second) 0).
    pose proof (complex_probability_nonnegative (fun _ => third) 0).
    unfold state_probability in *.
    assert (0 <= (1 + 3 * parameter) / 8 * complex_norm_sq (complex_add first fourth)) by
      (apply Rmult_le_pos; [lra|assumption]).
    assert (0 <= (1 - parameter) / 8 * complex_norm_sq (complex_difference first fourth)) by
      (apply Rmult_le_pos; [lra|assumption]).
    assert (0 <= (1 - parameter) / 4 * (complex_norm_sq second + complex_norm_sq third)) by
      (apply Rmult_le_pos; lra). lra.
Qed.

Theorem depolarizing_negative_half_witness :
  depolarizing_choi_form (-1/2) complex_one complex_zero complex_zero complex_one = -1/4.
Proof.
  unfold depolarizing_choi_form, complex_norm_sq, complex_real_pair, complex_zero, complex_one; simpl; field; ring.
Qed.
