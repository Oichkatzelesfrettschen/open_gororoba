(** Uniform bounds for weighted sums do not control unweighted energy.
    The constant residue sequence has uniformly bounded weighted partial
    sums and arbitrarily large unweighted squared partial sums. The result
    concerns finite sums over exact classical reals. An infinite-series
    convergence theorem requires a separate limit construction. *)
From Stdlib Require Import Reals Psatz Lia ZArith.
From OpenGororoba Require Import PassiveMechanicalMemory.
Open Scope R_scope.

Definition weighted_constant_residues count :=
  finite_sum count (fun index => 1 / INR (S index)^2).
Definition unweighted_constant_energy count :=
  finite_sum count (fun _ => 1^2).

Lemma reciprocal_square_telescoping_bound : forall denominator,
  1 <= denominator ->
  1 / denominator^2 <= 2 / denominator - 2 / (denominator + 1).
Proof.
  intros denominator positive.
  assert (identity : 2 / denominator - 2 / (denominator + 1) - 1 / denominator^2 =
    (denominator - 1) / (denominator^2 * (denominator + 1))) by (field; nra).
  assert (0 <= (denominator - 1) / (denominator^2 * (denominator + 1))).
  { unfold Rdiv. apply Rmult_le_pos; [lra|]. left. apply Rinv_0_lt_compat. nra. }
  lra.
Qed.

Theorem weighted_constant_partial_sums_bound : forall count,
  0 <= weighted_constant_residues count <= 2 - 2 / INR (S count).
Proof.
  intro count; split.
  - unfold weighted_constant_residues. apply finite_sum_nonnegative.
    intros index bounded. unfold Rdiv. apply Rmult_le_pos; [lra|].
    left. apply Rinv_0_lt_compat. pose proof (pos_INR (S index)).
    assert (0 < INR (S index)) by (apply lt_0_INR; lia). nra.
  - induction count as [| previous previous_bound].
    + unfold weighted_constant_residues; simpl. lra.
    + unfold weighted_constant_residues in *.
      change (finite_sum previous (fun index => 1 / INR (S index)^2) + 1 / INR (S previous)^2 <= 2 - 2 / INR (S (S previous))).
      pose proof (reciprocal_square_telescoping_bound (INR (S previous))) as step_bound.
      assert (1 <= INR (S previous)).
      { rewrite S_INR. pose proof (pos_INR previous). lra. }
      specialize (step_bound H). rewrite (S_INR (S previous)). lra.
Qed.

Theorem unweighted_constant_energy_exact : forall count,
  unweighted_constant_energy count = INR count.
Proof.
  induction count; unfold unweighted_constant_energy in *; [simpl; ring|].
  change (finite_sum count (fun _ => 1^2) + 1^2 = INR (S count)).
  rewrite IHcount, S_INR. ring.
Qed.

Lemma real_bound_has_larger_natural : forall bound,
  exists count, bound < INR count.
Proof.
  intro bound. exists (Z.to_nat (Z.max 0 (up bound))).
  rewrite INR_IZR_INZ, Z2Nat.id by lia.
  pose proof (proj1 (archimed bound)).
  pose proof (IZR_le (up bound) (Z.max 0 (up bound)) (Z.le_max_r 0 (up bound))). lra.
Qed.

Theorem weighted_bound_does_not_bound_unweighted_energy : forall bound,
  exists count,
  0 <= weighted_constant_residues count <= 2 /\
  bound < unweighted_constant_energy count.
Proof.
  intro bound. destruct (real_bound_has_larger_natural bound) as [count larger].
  exists count. split.
  - pose proof (weighted_constant_partial_sums_bound count) as [nonnegative upper].
    assert (0 <= 2 / INR (S count)).
    { unfold Rdiv. apply Rmult_le_pos; [lra|]. left. apply Rinv_0_lt_compat.
      apply lt_0_INR. lia. }
    split; lra.
  - rewrite unweighted_constant_energy_exact. assumption.
Qed.
