(** * C_QITauScaling: QI bound scales as tau^{-4}.

    Doubling the sampling time reduces the bound magnitude by 16x.
    This is the quartic scaling that makes QI bounds less restrictive
    at longer timescales. *)

From OpenGororoba Require Import Prelude QuantumInequalities.

Theorem claim_qi_tau_scaling :
  forall tau, tau > 0 ->
  qi_bound_4d (2 * tau) = qi_bound_4d tau / 16.
Proof. exact qi_bound_tau_scaling. Qed.
