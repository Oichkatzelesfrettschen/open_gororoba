(** * Hyperbolic tangent and secant: supplementary lemmas.

    Stdlib 9.1 provides tanh, sinh, cosh in Rtrigo_def, plus sinh_lt
    in Ranalysis4. We define sech (NOT in stdlib) and prove additional
    lemmas needed by WarpShapeFunction.v and the verified claim files.

    Key results beyond stdlib:
    - cosh_pos: cosh is strictly positive everywhere
    - tanh_neg: tanh is an odd function
    - sech / sech_neg: sech is an even function
    - cosh_sq_minus_sinh_sq: cosh^2 - sinh^2 = 1
    - tanh_bounded: -1 < tanh x < 1
    - tanh_lt: tanh is strictly increasing

    Mirrors: tanh() calls in warp_metric.rs shape functions. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** ---------- Definitions ---------- *)

(** sech is NOT provided by stdlib 9.1. *)
Definition sech (x : R) : R := / cosh x.

(** ---------- Positivity of cosh ---------- *)

Lemma cosh_pos : forall x : R, 0 < cosh x.
Proof.
  intro x. unfold cosh.
  assert (He1 : 0 < exp x) by apply exp_pos.
  assert (He2 : 0 < exp (- x)) by apply exp_pos.
  lra.
Qed.

Lemma cosh_neq_0 : forall x : R, cosh x <> 0.
Proof.
  intro x. generalize (cosh_pos x). lra.
Qed.

(** ---------- Symmetry of sinh and cosh ---------- *)

Lemma sinh_neg : forall x : R, sinh (- x) = - sinh x.
Proof.
  intro x. unfold sinh.
  rewrite Ropp_involutive.
  lra.
Qed.

Lemma cosh_neg : forall x : R, cosh (- x) = cosh x.
Proof.
  intro x. unfold cosh.
  rewrite Ropp_involutive.
  lra.
Qed.

(** ---------- tanh and sech symmetry ---------- *)

Lemma tanh_0 : tanh 0 = 0.
Proof.
  unfold tanh. rewrite sinh_0.
  unfold Rdiv. rewrite Rmult_0_l. reflexivity.
Qed.

Lemma tanh_neg : forall x : R, tanh (- x) = - tanh x.
Proof.
  intro x. unfold tanh.
  rewrite sinh_neg, cosh_neg.
  field. apply cosh_neq_0.
Qed.

Lemma sech_pos : forall x : R, 0 < sech x.
Proof.
  intro x. unfold sech.
  apply Rinv_pos. apply cosh_pos.
Qed.

Lemma sech_neg : forall x : R, sech (- x) = sech x.
Proof.
  intro x. unfold sech. rewrite cosh_neg. reflexivity.
Qed.

(** ---------- Antisymmetry identity ---------- *)

Lemma tanh_antisym : forall a : R,
  tanh a - tanh (- a) = 2 * tanh a.
Proof.
  intro a. rewrite tanh_neg. lra.
Qed.

(** ---------- sech squared symmetry ---------- *)

Lemma sech_sq_sym : forall a : R,
  sech a ^ 2 - sech (- a) ^ 2 = 0.
Proof.
  intro a. rewrite sech_neg. lra.
Qed.

(** ---------- The fundamental identity: cosh^2 - sinh^2 = 1 ---------- *)

Lemma cosh_sq_minus_sinh_sq : forall x : R,
  (cosh x) ^ 2 - (sinh x) ^ 2 = 1.
Proof.
  intro x. unfold cosh, sinh.
  assert (Hep : exp x * exp (- x) = 1).
  { rewrite <- exp_plus. rewrite Rplus_opp_r. apply exp_0. }
  assert (He1 : 0 < exp x) by apply exp_pos.
  assert (He2 : 0 < exp (- x)) by apply exp_pos.
  nra.
Qed.

(** ---------- Positivity of sinh for positive argument ---------- *)

Lemma sinh_pos_of_pos : forall x : R, x > 0 -> sinh x > 0.
Proof.
  intros x Hx.
  assert (Hs : sinh 0 < sinh x) by (apply sinh_lt; lra).
  rewrite sinh_0 in Hs.
  lra.
Qed.

(** ---------- Positivity of tanh for positive argument ---------- *)

Lemma tanh_pos_pos : forall x : R, x > 0 -> tanh x > 0.
Proof.
  intros x Hx.
  unfold tanh.
  apply Rdiv_pos_pos.
  - apply sinh_pos_of_pos. assumption.
  - apply cosh_pos.
Qed.

(** ---------- tanh is bounded between -1 and 1 ---------- *)

Lemma sinh_sq_lt_cosh_sq : forall x : R,
  (sinh x) ^ 2 < (cosh x) ^ 2.
Proof.
  intro x.
  assert (Hid := cosh_sq_minus_sinh_sq x).
  lra.
Qed.

Lemma tanh_bounded : forall x : R, -1 < tanh x < 1.
Proof.
  intro x.
  unfold tanh.
  assert (Hcp := cosh_pos x).
  assert (Hcsq := sinh_sq_lt_cosh_sq x).
  split.
  - unfold Rdiv.
    apply (Rmult_lt_reg_r (cosh x) (-1) (sinh x * / cosh x) Hcp).
    rewrite Rmult_assoc, Rinv_l, Rmult_1_r.
    + unfold sinh, cosh.
      assert (He1 : 0 < exp x) by apply exp_pos.
      assert (He2 : 0 < exp (- x)) by apply exp_pos.
      lra.
    + apply Rgt_not_eq. exact Hcp.
  - unfold Rdiv.
    apply (Rmult_lt_reg_r (cosh x) (sinh x * / cosh x) 1 Hcp).
    rewrite Rmult_assoc, Rinv_l, Rmult_1_r, Rmult_1_l.
    + unfold sinh, cosh.
      assert (He : 0 < exp (- x)) by apply exp_pos.
      lra.
    + apply Rgt_not_eq. exact Hcp.
Qed.

(** ---------- Monotonicity of tanh ---------- *)

(** tanh is strictly increasing: x < y -> tanh x < tanh y.
    Proof via sinh_lt (stdlib) and denominator clearing. *)
Lemma tanh_lt : forall x y : R, x < y -> tanh x < tanh y.
Proof.
  intros x y Hxy.
  unfold tanh.
  assert (Hcx := cosh_pos x).
  assert (Hcy := cosh_pos y).
  unfold Rdiv.
  apply (Rmult_lt_reg_r (cosh x * cosh y)).
  { nra. }
  replace (sinh x * / cosh x * (cosh x * cosh y))
    with (sinh x * cosh y).
  2:{ field. lra. }
  replace (sinh y * / cosh y * (cosh x * cosh y))
    with (sinh y * cosh x).
  2:{ field. lra. }
  assert (Hdiff : y - x > 0) by lra.
  assert (Hsinh_diff : sinh (y - x) > 0).
  { apply sinh_pos_of_pos. exact Hdiff. }
  unfold sinh, cosh in *.
  assert (Hex : 0 < exp x) by apply exp_pos.
  assert (Heny : 0 < exp (- y)) by apply exp_pos.
  assert (Hey : 0 < exp y) by apply exp_pos.
  assert (Henx : 0 < exp (- x)) by apply exp_pos.
  assert (Hediff : 0 < exp (y - x)) by apply exp_pos.
  assert (Hendiff : 0 < exp (- (y - x))) by apply exp_pos.
  assert (He1 : exp (y - x) = exp y * exp (- x)).
  { rewrite <- exp_plus. f_equal. }
  assert (He2 : exp (- (y - x)) = exp (- y) * exp x).
  { rewrite <- exp_plus. f_equal. ring. }
  rewrite He1, He2 in Hsinh_diff.
  nra.
Qed.
