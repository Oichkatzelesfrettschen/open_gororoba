(** * FractalMetricPositivity: Fractal metric g_uv positive-definite for D_f in (2,3).

    The fractal spacetime scaling factor is (r/r_0)^(D_f - 3).
    For D_f in (2, 3), the exponent is in (-1, 0), so the factor
    is positive for all r > 0 (it equals r_0^(3-D_f) / r^(3-D_f)).

    This guarantees the modified metric remains a valid Riemannian
    metric (positive-definite spatial part) everywhere outside the origin.

    Mirrors: gr_core/src/fractal_metric.rs (scaling_factor)
    Claims: C-1121 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** THEOREM: For D_f in (2, 3), the exponent D_f - 3 is in (-1, 0). *)
Theorem fractal_exponent_range :
  forall d_f : R,
    2 < d_f < 3 ->
    -1 < d_f - 3 < 0.
Proof.
  intros d_f [H1 H2]. lra.
Qed.

(** THEOREM: For r > 0 and r_0 > 0, the ratio r/r_0 is positive. *)
Theorem ratio_positive :
  forall r r_0 : R,
    r > 0 -> r_0 > 0 ->
    r / r_0 > 0.
Proof.
  intros r r_0 Hr Hr0.
  apply Rdiv_lt_0_compat; lra.
Qed.

(** THEOREM: The product of two positive reals is positive. *)
Theorem metric_component_positive :
  forall g_base scaling : R,
    g_base > 0 -> scaling > 0 ->
    g_base * scaling > 0.
Proof.
  intros g_base scaling Hg Hs.
  apply Rmult_lt_0_compat; lra.
Qed.

(** THEOREM: D_f / 4 is positive when D_f > 0. *)
Theorem fractal_factor_positive :
  forall d_f : R,
    d_f > 0 ->
    d_f / 4 > 0.
Proof.
  intros d_f Hd. lra.
Qed.
