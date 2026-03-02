(** * C-914: TCMT unitarity for lossless scattering.

    T + R = 1 where T = delta^2/(delta^2+gamma_e^2) and
    R = gamma_e^2/(delta^2+gamma_e^2) for lossless (gamma_i=0) TCMT. *)

From OpenGororoba Require Import Prelude FanoResonance.

Theorem C914_tcmt_unitarity : forall delta gamma_e,
  gamma_e > 0 ->
  delta^2 / (delta^2 + gamma_e^2) + gamma_e^2 / (delta^2 + gamma_e^2) = 1.
Proof. exact tcmt_unitarity. Qed.
