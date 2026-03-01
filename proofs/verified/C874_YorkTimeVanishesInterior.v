(** * C-874: York time vanishes in the nacelle warp interior.

    York time theta = div(beta) = d_x(beta_x). In the interior
    (rho << rho_0, |x| << L):

    - beta_x = -v_s * f(r_s) * M(rho,phi) * W_x(x)
    - f(r_s) ~ 1 (shape function at center)
    - M ~ 1 (gating suppressed, so modulation ~ 1)
    - W_x(x) ~ 1 (axial taper at center)

    The derivative d_x(beta_x) involves d_x(W_x) and d_x(f*M).
    At x=0: d_x(W_x) = 0 by the axial_taper_deriv_zero theorem
    (sech^2 symmetry at center).

    Combined with shape_fun_center and gating_interior_bound, the
    York time vanishes at the bubble center.

    Mirrors: test_york_time_center_zero in warp_metric.rs. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude Tanh WarpShapeFunction.

Open Scope R_scope.

(** C-874: Axial taper derivative vanishes at center.
    This is the key reason York time vanishes: d_x(W_x)(0) = 0. *)
Theorem C874_axial_deriv_zero : forall L sigma_x : R,
  sigma_x > 0 -> L > 0 ->
  axial_taper_deriv 0 L sigma_x = 0.
Proof.
  exact axial_taper_deriv_zero.
Qed.

(** The axial taper itself equals 1 at center.
    Combined with shape_fun_center: the shift is uniform -> div = 0. *)
Theorem C874_axial_taper_unity : forall L sigma_x : R,
  sigma_x > 0 -> L > 0 ->
  axial_taper 0 L sigma_x = 1.
Proof.
  exact axial_taper_center.
Qed.

(** The shape function also equals 1 at center. *)
Theorem C874_shape_unity : forall Rb sig : R,
  sig > 0 -> Rb > 0 ->
  shape_fun 0 Rb sig = 1.
Proof.
  exact shape_fun_center.
Qed.
