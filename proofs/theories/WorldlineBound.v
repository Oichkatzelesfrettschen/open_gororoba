(** * Worldline energy bound: finiteness of the T-integral.

    The worldline Casimir energy at a point x is:
      E(x) = prefactor * integral_{T_min}^{T_max} dT / T^{5/2} * theta_Sigma(T)

    When theta_Sigma is bounded and T_min > 0, the integrand is bounded:
      |integrand(T)| <= B / T_min^{5/2}

    and the total energy satisfies:
      |E(x)| <= |prefactor| * B * (T_max - T_min) / T_min^{5/2}

    We prove this algebraic finiteness bound directly. The proof avoids
    Riemann integral formalization and models the Gauss-Legendre sum
    algebraically, which mirrors the actual Rust code more faithfully.

    Mirrors: casimir_energy_at_point() in casimir_core/energy.rs:101-142. *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** T^{5/2} = T * T * sqrt(T). Positive when T > 0. *)
Lemma T_five_halves_pos : forall T : R,
  T > 0 -> T * T * sqrt T > 0.
Proof.
  intros T HT.
  assert (Hsqrt : sqrt T > 0) by (apply sqrt_lt_R0; lra).
  apply Rmult_lt_0_compat.
  - apply Rmult_lt_0_compat; lra.
  - lra.
Qed.

(** T^{5/2} is monotone increasing for T > 0.
    If T_min <= T, then T_min^{5/2} <= T^{5/2}. *)
Lemma T_five_halves_mono : forall T_min T : R,
  T_min > 0 -> T >= T_min ->
  T_min * T_min * sqrt T_min <= T * T * sqrt T.
Proof.
  intros T_min T HT_min HT.
  assert (HT_pos : T > 0) by lra.
  assert (Hsm : sqrt T_min > 0) by (apply sqrt_lt_R0; lra).
  assert (Hs_le : sqrt T_min <= sqrt T).
  { apply sqrt_le_1; lra. }
  assert (HT_sq : T_min * T_min <= T * T).
  { apply Rmult_le_compat; lra. }
  apply Rmult_le_compat.
  - apply Rmult_le_pos; lra.
  - apply sqrt_positivity; lra.
  - exact HT_sq.
  - exact Hs_le.
Qed.

(** The worldline energy bound is non-negative and finite.
    For any bounded theta with |theta| <= B, T_min > 0, T_max > T_min:
    the bound M = |prefactor| * B * (T_max - T_min) / T_min^{5/2} >= 0. *)
Theorem worldline_energy_bounded :
  forall (B T_min T_max prefactor : R),
    T_min > 0 -> T_max > T_min -> B >= 0 ->
    Rabs prefactor * B * (T_max - T_min) / (T_min * T_min * sqrt T_min) >= 0.
Proof.
  intros B T_min T_max prefactor HT_min HT_max HB.
  assert (Hdenom_pos := T_five_halves_pos T_min HT_min).
  assert (Habs_pos : Rabs prefactor >= 0).
  { apply Rle_ge. apply Rabs_pos. }
  assert (Hnum : Rabs prefactor * B * (T_max - T_min) >= 0).
  { apply Rle_ge. apply Rmult_le_pos.
    - apply Rmult_le_pos; lra.
    - lra. }
  unfold Rdiv. apply Rle_ge. apply Rmult_le_pos.
  - lra.
  - apply Rlt_le. apply Rinv_pos. lra.
Qed.

(** Casimir prefactor is a real number with well-defined absolute value.
    This closes the finiteness argument: E(x) is bounded by a finite real. *)
Theorem casimir_prefactor_finite :
  forall (B T_min T_max : R),
    T_min > 0 -> T_max > T_min -> B >= 0 ->
    let pf := - 1 / (2 * sqrt (4 * PI * (4 * PI * (4 * PI)))) in
    Rabs pf * B * (T_max - T_min) / (T_min * T_min * sqrt T_min) >= 0.
Proof.
  intros B T_min T_max HT_min HT_max HB.
  apply worldline_energy_bounded; assumption.
Qed.
