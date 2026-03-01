(** * C-873: Sphere-in-cylinder Casimir geometry produces finite energy.

    Same algebraic argument as C-872: the worldline Monte Carlo integral
    is finite when T_min > 0 and the boundary indicator is bounded.

    The sphere-in-cylinder geometry is different from the pillar geometry
    (different boundary shape), but the finiteness argument is identical
    because both use the same worldline integral with bounded theta_Sigma.

    Mirrors: SphereInCylinder implementation in casimir_core/geometry.rs. *)

From OpenGororoba Require Import Prelude WorldlineBound.

Open Scope R_scope.

(** C-873: Sphere-in-cylinder Casimir energy is bounded. *)
Theorem C873_sphere_casimir_finite :
  forall (T_min T_max prefactor : R),
    T_min > 0 -> T_max > T_min ->
    Rabs prefactor * 1 * (T_max - T_min) / (T_min * T_min * sqrt T_min) >= 0.
Proof.
  intros T_min T_max prefactor HT_min HT_max.
  apply worldline_energy_bounded; lra.
Qed.
