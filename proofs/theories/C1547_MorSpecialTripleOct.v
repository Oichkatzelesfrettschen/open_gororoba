(** * C1547_MorSpecialTripleOct: Moreno (1997) Theorem 2.13 -- canonical special triple witness.

    Moreno's Theorem 2.13 identifies, for a special triple {a, y, b}, the
    8-dimensional span

      V(a;y;b) = {e0, a, b, a*b, (a*y)*b, y*b, a*y, y}

    with a copy of the octonions.

    This file formalizes the canonical octonion witness used by the Rust
    Moreno lane:

      a := e1,  y := e7,  b := e2

    In the source order above, the basis collapses exactly to the standard
    octonion basis:

      {e0, e1, e2, e3, e4, e5, e6, e7}.

    The proof is staged in the repo's modern finite-basis style:
    1. certify the four nontrivial basis products concretely;
    2. prove the V(a;y;b) basis agrees with the standard octonion basis;
    3. transport the full 8x8 multiplication table through CDSignBridge.

    This gives a paper-scoped concrete Rocq companion for C-1547 while the
    fully abstract "iff special triple then octonion copy in A_n" direction
    remains separate. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDSignBridge CDNegLemmas.

Definition moreno_special_a : CDOct := oct_e 1.
Definition moreno_special_y : CDOct := oct_e 7.
Definition moreno_special_b : CDOct := oct_e 2.

Definition moreno_v_basis (i : nat) : CDOct :=
  match i with
  | 0 => oct_e 0
  | 1 => moreno_special_a
  | 2 => moreno_special_b
  | 3 => oct_mul moreno_special_a moreno_special_b
  | 4 => oct_mul (oct_mul moreno_special_a moreno_special_y) moreno_special_b
  | 5 => oct_mul moreno_special_y moreno_special_b
  | 6 => oct_mul moreno_special_a moreno_special_y
  | 7 => moreno_special_y
  | _ => oct_zero
  end.

Lemma moreno_special_ab :
  oct_mul moreno_special_a moreno_special_b = oct_e 3.
Proof.
  unfold moreno_special_a, moreno_special_b.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_special_ay :
  oct_mul moreno_special_a moreno_special_y = oct_e 6.
Proof.
  unfold moreno_special_a, moreno_special_y.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_special_yb :
  oct_mul moreno_special_y moreno_special_b = oct_e 5.
Proof.
  unfold moreno_special_y, moreno_special_b.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_special_ayb :
  oct_mul (oct_mul moreno_special_a moreno_special_y) moreno_special_b = oct_e 4.
Proof.
  rewrite moreno_special_ay.
  unfold moreno_special_b.
  cbv [oct_mul oct_e oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Lemma moreno_special_a_yb :
  oct_mul moreno_special_a (oct_mul moreno_special_y moreno_special_b) = oct_neg (oct_e 4).
Proof.
  rewrite moreno_special_yb.
  unfold moreno_special_a.
  cbv [oct_mul oct_e oct_neg oct_conj quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  repeat f_equal; ring.
Qed.

Theorem moreno_special_triple_relation :
  oct_mul (oct_mul moreno_special_a moreno_special_y) moreno_special_b =
  oct_neg (oct_mul moreno_special_a (oct_mul moreno_special_y moreno_special_b)).
Proof.
  rewrite moreno_special_ayb.
  rewrite moreno_special_a_yb.
  rewrite oct_neg_neg.
  reflexivity.
Qed.

Theorem moreno_v_basis_eq_oct_e : forall i : nat,
  (i < 8)%nat ->
  moreno_v_basis i = oct_e i.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - exact moreno_special_ab.
  - exact moreno_special_ayb.
  - exact moreno_special_yb.
  - exact moreno_special_ay.
  - reflexivity.
Qed.

Lemma moreno_v_basis_lxor_lt8 : forall i j : nat,
  (i < 8)%nat ->
  (j < 8)%nat ->
  (Nat.lxor i j < 8)%nat.
Proof.
  intros i j Hi Hj.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  vm_compute; lia.
Qed.

Theorem moreno_v_basis_unit_norm : forall i : nat,
  (i < 8)%nat ->
  oct_norm_sq (moreno_v_basis i) = 1.
Proof.
  intros i Hi.
  rewrite moreno_v_basis_eq_oct_e by exact Hi.
  apply oct_e_norm; exact Hi.
Qed.

Theorem moreno_v_mul_table : forall i j : nat,
  (i < 8)%nat ->
  (j < 8)%nat ->
  oct_mul (moreno_v_basis i) (moreno_v_basis j) =
    oct_scale (sign_to_R (oct_sign i j)) (moreno_v_basis (Nat.lxor i j)).
Proof.
  intros i j Hi Hj.
  rewrite moreno_v_basis_eq_oct_e by exact Hi.
  rewrite moreno_v_basis_eq_oct_e by exact Hj.
  rewrite moreno_v_basis_eq_oct_e.
  - apply oct_basis_mul_xor; assumption.
  - apply moreno_v_basis_lxor_lt8; assumption.
Qed.
