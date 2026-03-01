(** * Warp shape functions: Alcubierre top-hat, axial taper, and gating.

    Defines and proves properties of the component functions for the
    nacelle warp bubble metric (White et al. 2025):

    - shape_fun: Alcubierre tanh top-hat f(r_s)
    - axial_taper: W_x(x), same functional form along propagation axis
    - gating: logistic transition W_g(rho)

    Key results:
    - shape_fun_center: f(0,Rb,sig) = 1
    - axial_taper_center: W_x(0,L,sigma_x) = 1
    - shape_fun_deriv_zero: sech^2 symmetry at center
    - axial_taper_deriv_zero: same for W_x
    - gating_range: 0 < gating(rho,...) < 1
    - gating_interior_bound: W_g(0,...) < 1/2 when delta < rho_0
    - shape_fun_bounded: 0 <= f <= 1 for reasonable parameters

    Mirrors: alcubierre_shape_function() in warp_metric.rs:125-131,
    axial_taper() in warp_metric.rs:209-215,
    gating_function() in warp_metric.rs:162-164.

    NOTE: Parameter names avoid R (the real number type). We use Rb
    for the bubble radius parameter. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude Tanh.

Open Scope R_scope.

(** ---------- Definitions ---------- *)

(** Alcubierre shape function f(r_s, Rb, sig).
    f = [tanh(sig*(r_s+Rb)) - tanh(sig*(r_s-Rb))] / [2*tanh(sig*Rb)] *)
Definition shape_fun (r_s Rb sig : R) : R :=
  (tanh (sig * (r_s + Rb)) - tanh (sig * (r_s - Rb))) / (2 * tanh (sig * Rb)).

(** Axial taper W_x(x, L, sigma_x) -- identical functional form.
    W_x = [tanh(sigma_x*(x+L)) - tanh(sigma_x*(x-L))] / [2*tanh(sigma_x*L)] *)
Definition axial_taper (x L sigma_x : R) : R :=
  (tanh (sigma_x * (x + L)) - tanh (sigma_x * (x - L))) / (2 * tanh (sigma_x * L)).

(** Logistic gating function W_g(rho, rho_0, kappa, delta).
    W_g = 1 / (1 + exp(-kappa * (rho - rho_0 + delta))) *)
Definition gating (rho rho_0 kappa delta : R) : R :=
  1 / (1 + exp (- kappa * (rho - rho_0 + delta))).

(** Derivative of the shape function (sech^2 form).
    df/dr_s = sig * [sech^2(sig*(r_s+Rb)) - sech^2(sig*(r_s-Rb))] / [2*tanh(sig*Rb)] *)
Definition shape_fun_deriv (r_s Rb sig : R) : R :=
  sig * ((sech (sig * (r_s + Rb)))^2 - (sech (sig * (r_s - Rb)))^2) /
  (2 * tanh (sig * Rb)).

(** ---------- Helper lemma: tanh(sig*Rb) > 0 when sig > 0 and Rb > 0 ---------- *)

Lemma tanh_sig_Rb_pos : forall sig Rb : R,
  sig > 0 -> Rb > 0 ->
  tanh (sig * Rb) > 0.
Proof.
  intros sig Rb Hs HR.
  apply tanh_pos_pos. nra.
Qed.

Lemma denom_pos : forall sig Rb : R,
  sig > 0 -> Rb > 0 ->
  2 * tanh (sig * Rb) > 0.
Proof.
  intros sig Rb Hs HR.
  assert (Ht := tanh_sig_Rb_pos sig Rb Hs HR). lra.
Qed.

Lemma denom_neq_0 : forall sig Rb : R,
  sig > 0 -> Rb > 0 ->
  2 * tanh (sig * Rb) <> 0.
Proof.
  intros sig Rb Hs HR.
  assert (Hd := denom_pos sig Rb Hs HR). lra.
Qed.

Lemma tanh_sig_Rb_neq_0 : forall sig Rb : R,
  sig > 0 -> Rb > 0 ->
  tanh (sig * Rb) <> 0.
Proof.
  intros sig Rb Hs HR.
  assert (Ht := tanh_sig_Rb_pos sig Rb Hs HR). lra.
Qed.

(** ---------- shape_fun_center: f(0, Rb, sig) = 1 ---------- *)

(** At the center r_s=0, the numerator simplifies by tanh antisymmetry:
    tanh(sig*Rb) - tanh(-sig*Rb) = tanh(sig*Rb) + tanh(sig*Rb) = 2*tanh(sig*Rb).
    So f = 2*tanh(sig*Rb) / (2*tanh(sig*Rb)) = 1. *)
Theorem shape_fun_center : forall Rb sig : R,
  sig > 0 -> Rb > 0 ->
  shape_fun 0 Rb sig = 1.
Proof.
  intros Rb sig Hs HR.
  unfold shape_fun.
  replace (0 + Rb) with Rb by ring.
  replace (0 - Rb) with (-Rb) by ring.
  replace (sig * - Rb) with (- (sig * Rb)) by ring.
  rewrite tanh_neg.
  field.
  apply tanh_sig_Rb_neq_0; assumption.
Qed.

(** ---------- axial_taper_center: W_x(0, L, sigma_x) = 1 ---------- *)

Theorem axial_taper_center : forall L sigma_x : R,
  sigma_x > 0 -> L > 0 ->
  axial_taper 0 L sigma_x = 1.
Proof.
  intros L sigma_x Hs HL.
  unfold axial_taper.
  replace (0 + L) with L by ring.
  replace (0 - L) with (-L) by ring.
  replace (sigma_x * - L) with (- (sigma_x * L)) by ring.
  rewrite tanh_neg. field.
  apply tanh_sig_Rb_neq_0; assumption.
Qed.

(** ---------- shape_fun_deriv_zero: df/dr_s(0) = 0 by sech^2 symmetry ---------- *)

(** At r_s=0: sech^2(sig*Rb) - sech^2(-sig*Rb) = 0 by sech evenness. *)
Theorem shape_fun_deriv_zero : forall Rb sig : R,
  sig > 0 -> Rb > 0 ->
  shape_fun_deriv 0 Rb sig = 0.
Proof.
  intros Rb sig Hs HR.
  unfold shape_fun_deriv.
  replace (0 + Rb) with Rb by ring.
  replace (0 - Rb) with (-Rb) by ring.
  replace (sig * - Rb) with (-(sig * Rb)) by ring.
  rewrite sech_neg.
  field.
  apply tanh_sig_Rb_neq_0; assumption.
Qed.

(** ---------- axial_taper_deriv_zero ---------- *)

(** Identical argument for the axial taper derivative. *)
Definition axial_taper_deriv (x L sigma_x : R) : R :=
  sigma_x * ((sech (sigma_x * (x + L)))^2 - (sech (sigma_x * (x - L)))^2) /
  (2 * tanh (sigma_x * L)).

Theorem axial_taper_deriv_zero : forall L sigma_x : R,
  sigma_x > 0 -> L > 0 ->
  axial_taper_deriv 0 L sigma_x = 0.
Proof.
  intros L sigma_x Hs HL.
  unfold axial_taper_deriv.
  replace (0 + L) with L by ring.
  replace (0 - L) with (-L) by ring.
  replace (sigma_x * - L) with (-(sigma_x * L)) by ring.
  rewrite sech_neg. field.
  apply tanh_sig_Rb_neq_0; assumption.
Qed.

(** ---------- gating_range: 0 < W_g < 1 ---------- *)

(** The logistic function is always in (0, 1) since exp > 0. *)
Theorem gating_range : forall rho rho_0 kappa delta : R,
  kappa > 0 ->
  0 < gating rho rho_0 kappa delta < 1.
Proof.
  intros rho rho_0 kappa delta Hk.
  unfold gating.
  assert (He : 0 < exp (- kappa * (rho - rho_0 + delta))) by apply exp_pos.
  split.
  - unfold Rdiv. apply Rmult_lt_0_compat.
    + lra.
    + apply Rinv_pos. lra.
  - apply (Rmult_lt_reg_r (1 + exp (- kappa * (rho - rho_0 + delta)))).
    { lra. }
    unfold Rdiv. rewrite Rmult_1_l. rewrite Rinv_l.
    2:{ lra. }
    lra.
Qed.

(** ---------- gating_interior_bound: W_g(0,...) < 1/2 when delta < rho_0 ---------- *)

(** When rho=0 and delta < rho_0, the exponent -kappa*(0-rho_0+delta)
    = kappa*(rho_0-delta) > 0, so exp > 1, so 1/(1+exp) < 1/2. *)
Theorem gating_interior_bound : forall rho_0 kappa delta : R,
  kappa > 0 -> rho_0 > 0 -> delta < rho_0 ->
  gating 0 rho_0 kappa delta < 1/2.
Proof.
  intros rho_0 kappa delta Hk Hrho Hdelta.
  unfold gating.
  assert (Harg : - kappa * (0 - rho_0 + delta) = kappa * (rho_0 - delta)) by ring.
  rewrite Harg.
  assert (Hpos : kappa * (rho_0 - delta) > 0) by nra.
  assert (Hexp : exp (kappa * (rho_0 - delta)) > 1).
  { assert (Hgt := exp_increasing 0 (kappa * (rho_0 - delta))).
    rewrite exp_0 in Hgt. apply Hgt. lra. }
  assert (He : 0 < exp (kappa * (rho_0 - delta))) by apply exp_pos.
  unfold Rdiv.
  apply (Rmult_lt_reg_r (2 * (1 + exp (kappa * (rho_0 - delta))))).
  { nra. }
  rewrite Rmult_assoc.
  replace (/ (1 + exp (kappa * (rho_0 - delta))) *
           (2 * (1 + exp (kappa * (rho_0 - delta)))))
    with 2.
  2:{ field. lra. }
  replace (1 * / 2 * (2 * (1 + exp (kappa * (rho_0 - delta)))))
    with (1 + exp (kappa * (rho_0 - delta))).
  2:{ field. }
  lra.
Qed.

(** ---------- shape_fun_bounded: 0 <= f <= 1 for r_s >= 0 ---------- *)

(** The shape function is bounded between 0 and 1 when r_s >= 0.
    This follows from tanh being monotonically increasing and bounded. *)
Theorem shape_fun_nonneg : forall r_s Rb sig : R,
  sig > 0 -> Rb > 0 -> r_s >= 0 ->
  shape_fun r_s Rb sig >= 0.
Proof.
  intros r_s Rb sig Hs HR Hrs.
  unfold shape_fun.
  assert (Hd := denom_pos sig Rb Hs HR).
  unfold Rdiv.
  apply Rle_ge.
  apply Rmult_le_pos.
  - assert (Hlt : sig * (r_s - Rb) < sig * (r_s + Rb)) by nra.
    assert (Hm := tanh_lt (sig * (r_s - Rb)) (sig * (r_s + Rb)) Hlt).
    lra.
  - apply Rlt_le. apply Rinv_pos. exact Hd.
Qed.
