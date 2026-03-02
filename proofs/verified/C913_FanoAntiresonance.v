(** * C-913: Fano antiresonance -- perfect reflection at resonance.

    At delta=0 with lossless coupling (gamma_i=0), TCMT transmission
    vanishes: T(0, gamma_e, 0) = 0 for gamma_e > 0. *)

From OpenGororoba Require Import Prelude FanoResonance.

Theorem C913_fano_antiresonance : forall gamma_e,
  gamma_e > 0 -> tcmt_T 0 gamma_e 0 = 0.
Proof. exact tcmt_antiresonance. Qed.
