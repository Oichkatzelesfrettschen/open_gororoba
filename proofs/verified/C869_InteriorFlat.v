(** * C-869: Nacelle warp bubble interior-flat condition.

    K_ij = 0 at the bubble center (rho << rho_0) to machine precision.

    In the nacelle warp metric, K_ij depends on derivatives of the shift
    vector beta^i. At the center (r_s=0), the shift derivatives vanish
    because the shape function derivative df/dr_s(0) = 0 (sech^2 symmetry).

    The shape function derivative is:
      df/dr_s = sigma * [sech^2(sigma*(r_s+R)) - sech^2(sigma*(r_s-R))] / denom

    At r_s = 0, sech^2(sigma*R) = sech^2(-sigma*R) by sech evenness,
    so the numerator cancels to zero.

    Mirrors: test_york_time_center_zero in warp_metric.rs. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude Tanh WarpShapeFunction.

Open Scope R_scope.

(** C-869: At bubble center, the shape function derivative vanishes,
    which implies K_ij = 0 (since K depends on spatial derivatives of beta,
    and beta depends on f' which is zero at center). *)
Theorem C869_interior_flat : forall Rb sig : R,
  sig > 0 -> Rb > 0 ->
  shape_fun_deriv 0 Rb sig = 0.
Proof.
  exact shape_fun_deriv_zero.
Qed.

(** The gating function is near zero at the interior, reinforcing flatness. *)
Theorem C869_gating_suppressed : forall rho_0 kappa delta : R,
  kappa > 0 -> rho_0 > 0 -> delta < rho_0 ->
  gating 0 rho_0 kappa delta < 1/2.
Proof.
  exact gating_interior_bound.
Qed.
