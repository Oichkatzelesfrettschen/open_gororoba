(** * C-1364: Linear homotopy bridge law under explicit assumptions.

    This file mirrors the exact bridge-law layer implemented in
    `crates/cosmology_core/src/homotopy_bridge.rs`.

    Formalized facts:
    - lambda = coupling * obstruction_norm;
    - zero coupling gives zero lambda;
    - zero coupling recovers isotropic tangential pressure;
    - positive obstruction, coupling, density, and radial pressure imply a
      strictly positive anisotropy correction.

    These are bridge-law consequences under explicit assumptions. They do not
    derive a physical stress-energy tensor from the algebra. *)

From Stdlib Require Import Lra.
From OpenGororoba Require Import Prelude.

Open Scope R_scope.

Definition homotopy_lambda_law (obstruction_norm coupling : R) : R :=
  coupling * obstruction_norm.

Definition anisotropy_correction
    (rho p_r obstruction_norm coupling : R) : R :=
  homotopy_lambda_law obstruction_norm coupling * rho * p_r.

Definition tangential_pressure
    (rho p_r obstruction_norm coupling : R) : R :=
  p_r + anisotropy_correction rho p_r obstruction_norm coupling.

Theorem C1364_zero_coupling_lambda_zero :
  forall obstruction_norm : R,
    homotopy_lambda_law obstruction_norm 0 = 0.
Proof.
  intros obstruction_norm.
  unfold homotopy_lambda_law.
  ring.
Qed.

Theorem C1364_zero_coupling_isotropic :
  forall rho p_r obstruction_norm : R,
    tangential_pressure rho p_r obstruction_norm 0 = p_r.
Proof.
  intros rho p_r obstruction_norm.
  unfold tangential_pressure, anisotropy_correction, homotopy_lambda_law.
  ring.
Qed.

Theorem C1364_positive_anisotropy_correction :
  forall rho p_r obstruction_norm coupling : R,
    rho > 0 ->
    p_r > 0 ->
    obstruction_norm > 0 ->
    coupling > 0 ->
    anisotropy_correction rho p_r obstruction_norm coupling > 0.
Proof.
  intros rho p_r obstruction_norm coupling Hrho Hpr Hobs Hc.
  unfold anisotropy_correction, homotopy_lambda_law.
  repeat apply Rmult_lt_0_compat; assumption.
Qed.

Theorem C1364_positive_coupling_increases_tangential_pressure :
  forall rho p_r obstruction_norm coupling : R,
    rho > 0 ->
    p_r > 0 ->
    obstruction_norm > 0 ->
    coupling > 0 ->
    tangential_pressure rho p_r obstruction_norm coupling > p_r.
Proof.
  intros rho p_r obstruction_norm coupling Hrho Hpr Hobs Hc.
  unfold tangential_pressure.
  pose proof
    (C1364_positive_anisotropy_correction rho p_r obstruction_norm coupling
      Hrho Hpr Hobs Hc) as Hsigma.
  lra.
Qed.
