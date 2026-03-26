(** * C1540_MorHaQuaternion: Moreno (1997) Theorem 1.13 -- canonical H_a copy.

    Moreno's Theorem 1.13 says that for a doubly-pure unit-norm element a,
    the span H_a = {e0, a, tilde_a, tilde_e0} is a quaternion subalgebra.

    This file formalizes the canonical concrete instance in the octonions:
      a         := e1
      tilde_a   := e5
      tilde_e0  := e4

    in the source order {e0, a, tilde_a, tilde_e0}.  The resulting
    multiplication table is the opposite-handed quaternion table used by the
    Rust Moreno companion:

      1 * 1 = 1           1 * a = a           1 * tilde_a = tilde_a
      1 * tilde_e0 = tilde_e0

      a * 1 = a           a * a = -1          a * tilde_a = -tilde_e0
      a * tilde_e0 = tilde_a

      tilde_a * 1 = tilde_a
      tilde_a * a = tilde_e0
      tilde_a * tilde_a = -1
      tilde_a * tilde_e0 = -a

      tilde_e0 * 1 = tilde_e0
      tilde_e0 * a = -tilde_a
      tilde_e0 * tilde_a = a
      tilde_e0 * tilde_e0 = -1

    This is a paper-scoped concrete Rocq companion for C-1540.  It certifies
    the standard octonion witness used by the executable Moreno lane, while
    the full arbitrary-dimensional lift remains separate. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDSignBridge.

Definition moreno_a : CDOct := oct_e 1.
Definition moreno_tilde_a : CDOct := oct_e 5.
Definition moreno_tilde_e0 : CDOct := oct_e 4.

Definition moreno_ha_basis (i : nat) : CDOct :=
  match i with
  | 0 => oct_e 0
  | 1 => moreno_a
  | 2 => moreno_tilde_a
  | 3 => moreno_tilde_e0
  | _ => oct_zero
  end.

Definition moreno_ha_expected (i j : nat) : CDOct :=
  match i, j with
  | 0, 0 => oct_e 0
  | 0, 1 => moreno_a
  | 0, 2 => moreno_tilde_a
  | 0, 3 => moreno_tilde_e0
  | 1, 0 => moreno_a
  | 1, 1 => oct_neg (oct_e 0)
  | 1, 2 => oct_neg moreno_tilde_e0
  | 1, 3 => moreno_tilde_a
  | 2, 0 => moreno_tilde_a
  | 2, 1 => moreno_tilde_e0
  | 2, 2 => oct_neg (oct_e 0)
  | 2, 3 => oct_neg moreno_a
  | 3, 0 => moreno_tilde_e0
  | 3, 1 => oct_neg moreno_tilde_a
  | 3, 2 => moreno_a
  | 3, 3 => oct_neg (oct_e 0)
  | _, _ => oct_zero
  end.

Theorem moreno_ha_basis_unit_norm : forall i : nat,
  (i < 4)%nat ->
  oct_norm_sq (moreno_ha_basis i) = 1.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|]]]]; try lia;
  unfold moreno_ha_basis, moreno_a, moreno_tilde_a, moreno_tilde_e0;
  apply oct_basis_unit_norm; lia.
Qed.

Lemma moreno_a_mul_e0 : oct_mul moreno_a (oct_e 0) = moreno_a.
Proof.
  unfold moreno_a.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_e0_mul_a : oct_mul (oct_e 0) moreno_a = moreno_a.
Proof.
  unfold moreno_a.
  apply oct_mul_e0_left; lia.
Qed.

Lemma moreno_tilde_a_mul_e0 : oct_mul moreno_tilde_a (oct_e 0) = moreno_tilde_a.
Proof.
  unfold moreno_tilde_a.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_e0_mul_tilde_a : oct_mul (oct_e 0) moreno_tilde_a = moreno_tilde_a.
Proof.
  unfold moreno_tilde_a.
  apply oct_mul_e0_left; lia.
Qed.

Lemma moreno_tilde_e0_mul_e0 : oct_mul moreno_tilde_e0 (oct_e 0) = moreno_tilde_e0.
Proof.
  unfold moreno_tilde_e0.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_e0_mul_tilde_e0 : oct_mul (oct_e 0) moreno_tilde_e0 = moreno_tilde_e0.
Proof.
  unfold moreno_tilde_e0.
  apply oct_mul_e0_left; lia.
Qed.

Lemma moreno_a_square : oct_mul moreno_a moreno_a = oct_neg (oct_e 0).
Proof.
  unfold moreno_a.
  apply oct_mul_self_neg; lia.
Qed.

Lemma moreno_tilde_a_square : oct_mul moreno_tilde_a moreno_tilde_a = oct_neg (oct_e 0).
Proof.
  unfold moreno_tilde_a.
  apply oct_mul_self_neg; lia.
Qed.

Lemma moreno_a_mul_tilde_a : oct_mul moreno_a moreno_tilde_a = oct_neg moreno_tilde_e0.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_neg oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_tilde_a_mul_a : oct_mul moreno_tilde_a moreno_a = moreno_tilde_e0.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_a_mul_tilde_e0 : oct_mul moreno_a moreno_tilde_e0 = moreno_tilde_a.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_tilde_e0_mul_a : oct_mul moreno_tilde_e0 moreno_a = oct_neg moreno_tilde_a.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_neg oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_tilde_a_mul_tilde_e0 : oct_mul moreno_tilde_a moreno_tilde_e0 = oct_neg moreno_a.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_neg oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_tilde_e0_mul_tilde_a : oct_mul moreno_tilde_e0 moreno_tilde_a = moreno_a.
Proof.
  unfold moreno_a, moreno_tilde_a, moreno_tilde_e0.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_tilde_e0_square_local :
  oct_mul moreno_tilde_e0 moreno_tilde_e0 = oct_neg (oct_e 0).
Proof.
  unfold moreno_tilde_e0.
  apply oct_mul_self_neg; lia.
Qed.

Theorem moreno_ha_mul_table : forall i j : nat,
  (i < 4)%nat ->
  (j < 4)%nat ->
  oct_mul (moreno_ha_basis i) (moreno_ha_basis j) = moreno_ha_expected i j.
Proof.
  intros i j Hi Hj.
  destruct i as [|[|[|[|]]]]; try lia.
  - destruct j as [|[|[|[|]]]]; try lia.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
           quat_zero quat_one oct_lo oct_hi qa qb qc qd].
      repeat f_equal; ring.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_e0_mul_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_e0_mul_tilde_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_e0_mul_tilde_e0.
  - destruct j as [|[|[|[|]]]]; try lia.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_a_mul_e0.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_a_square.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_a_mul_tilde_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_a_mul_tilde_e0.
  - destruct j as [|[|[|[|]]]]; try lia.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_a_mul_e0.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_a_mul_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_a_square.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_a_mul_tilde_e0.
  - destruct j as [|[|[|[|]]]]; try lia.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_e0_mul_e0.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_e0_mul_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_e0_mul_tilde_a.
    + unfold moreno_ha_basis, moreno_ha_expected; simpl.
      exact moreno_tilde_e0_square_local.
Qed.

Corollary moreno_a_tilde_a_eq_neg_tilde_e0 :
  oct_mul moreno_a moreno_tilde_a = oct_neg moreno_tilde_e0.
Proof. exact (moreno_ha_mul_table 1 2 ltac:(lia) ltac:(lia)). Qed.

Corollary moreno_tilde_a_a_eq_tilde_e0 :
  oct_mul moreno_tilde_a moreno_a = moreno_tilde_e0.
Proof. exact (moreno_ha_mul_table 2 1 ltac:(lia) ltac:(lia)). Qed.

Corollary moreno_tilde_e0_square :
  oct_mul moreno_tilde_e0 moreno_tilde_e0 = oct_neg (oct_e 0).
Proof. exact moreno_tilde_e0_square_local. Qed.
