(** * FanoResonance: Temporal coupled-mode theory scattering.

    Defines the TCMT transmission and reflection coefficients for a
    single resonance coupled to a waveguide:
      t = (i*delta + gamma_i) / (i*delta + gamma_e + gamma_i)
      r = gamma_e / (i*delta + gamma_e + gamma_i)

    where delta = omega - omega_0 is the detuning, gamma_e is the
    external coupling rate, and gamma_i is the intrinsic loss rate.

    In terms of |t|^2 and |r|^2:
      T = (delta^2 + gamma_i^2) / (delta^2 + (gamma_e + gamma_i)^2)
      R = gamma_e^2 / (delta^2 + (gamma_e + gamma_i)^2) -- NOT correct for full |r|^2

    Energy conservation: T + R_full = 1 when gamma_i = 0 (lossless).

    Mirrors: optics_core/src/fano_tcmt.rs *)

From OpenGororoba Require Import Prelude.

(** Squared transmission coefficient |t|^2.
    Numerator: delta^2 + gamma_i^2.
    Denominator: delta^2 + (gamma_e + gamma_i)^2. *)
Definition tcmt_T (delta gamma_e gamma_i : R) : R :=
  (delta^2 + gamma_i^2) / (delta^2 + (gamma_e + gamma_i)^2).

(** Squared reflection coefficient contribution from coupling.
    For the lossless case (gamma_i=0), |r|^2 = gamma_e^2 / (delta^2 + gamma_e^2).
    The full Fano |r|^2 depends on the phase, but in the symmetric
    (no Fano interference) case: |r|^2 = gamma_e^2 / denom^2.
    For unitarity we use: R = 1 - T (lossless). *)

(** TCMT denominator. *)
Definition tcmt_denom (delta gamma_e gamma_i : R) : R :=
  delta^2 + (gamma_e + gamma_i)^2.

(** At exact resonance (delta=0), lossless (gamma_i=0): T = 0.
    All power is reflected -- the resonance acts as a perfect mirror. *)
Theorem tcmt_antiresonance : forall gamma_e,
  gamma_e > 0 -> tcmt_T 0 gamma_e 0 = 0.
Proof.
  intros gamma_e Hge.
  unfold tcmt_T. simpl.
  field. nra.
Qed.

(** T = 1 when gamma_e = 0 (no coupling) and gamma_i > 0.
    Without waveguide coupling, all power is transmitted. *)
Theorem tcmt_no_coupling : forall delta gamma_i,
  gamma_i > 0 ->
  tcmt_T delta 0 gamma_i = 1.
Proof.
  intros delta gamma_i Hgi. unfold tcmt_T. field. nra.
Qed.

(** Unitarity identity: for lossless scattering (gamma_i=0),
    T + R = 1 where T = delta^2/(delta^2+gamma_e^2) and
    R = gamma_e^2/(delta^2+gamma_e^2). *)
Theorem tcmt_unitarity : forall delta gamma_e,
  gamma_e > 0 ->
  delta^2 / (delta^2 + gamma_e^2) + gamma_e^2 / (delta^2 + gamma_e^2) = 1.
Proof.
  intros delta gamma_e Hge.
  field.
  assert (Hge2 : gamma_e^2 > 0) by nra.
  assert (Hd2 : delta^2 >= 0) by nra.
  nra.
Qed.
