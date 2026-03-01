(** * C_QIBoundNegative: Ford-Roman QI bound is strictly negative.

    The 4D quantum inequality bound -3/(32 pi^2 tau^4) is always negative
    for positive sampling time tau. This is the foundational sign property
    that constrains how negative time-averaged energy densities can be. *)

From OpenGororoba Require Import Prelude QuantumInequalities.

Theorem claim_qi_bound_negative :
  forall tau, tau > 0 -> qi_bound_4d tau < 0.
Proof. exact qi_bound_negative. Qed.
