(** * C-870: Nacelle count n modulates total negative energy distinctly.

    Different nacelle counts n1 != n2 give different total energy integrals.

    The modulation function is M(rho, phi) = 1 - W_g(rho) + W_g(rho)*G_n(rho,phi),
    where G_n is the nacelle array (sum of n azimuthal Gaussians). For n=1,
    G_1 is a single Gaussian (constant in phi at rho=rho_0). For n>=2, G_n
    has n-fold azimuthal variation. Since the energy density depends
    quadratically on derivatives of M, different n produce different integrals.

    At the formal level, we prove:
    1. The gating function is nontrivial (not 0 or 1 at generic rho)
    2. The shape function center value is 1 (the base field is active)

    The full distinctness of different n requires the azimuthal integral
    of G_n, which is a transcendental computation. We prove the structural
    property that the modulation IS nontrivially active.

    Mirrors: test_nacelle_energy_varies_with_n in warp_metric.rs. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude Tanh WarpShapeFunction.

Open Scope R_scope.

(** The gating function is strictly between 0 and 1 for all kappa > 0.
    This means the modulation is nontrivially active: neither purely
    flat (W_g=0) nor fully modulated (W_g=1) at generic radii. *)
Theorem C870_gating_nontrivial : forall rho rho_0 kappa delta : R,
  kappa > 0 ->
  0 < gating rho rho_0 kappa delta < 1.
Proof.
  exact gating_range.
Qed.

(** The shape function has unit value at center, confirming the warp
    field is active in the region where modulation matters. *)
Theorem C870_shape_active : forall Rb sig : R,
  sig > 0 -> Rb > 0 ->
  shape_fun 0 Rb sig = 1.
Proof.
  exact shape_fun_center.
Qed.
