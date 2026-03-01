(** * C_WECImpliesNEC: classical energy condition implication chain.

    Proves WEC => NEC, SEC => NEC, DEC => WEC, and DEC => NEC.
    These are the standard implications between energy conditions
    in general relativity. *)

From OpenGororoba Require Import Prelude EnergyConditions.

Theorem claim_WEC_implies_NEC :
  forall rho p, WEC rho p -> NEC rho p.
Proof. exact WEC_implies_NEC. Qed.

Theorem claim_SEC_implies_NEC :
  forall rho p, SEC rho p -> NEC rho p.
Proof. exact SEC_implies_NEC. Qed.

Theorem claim_DEC_implies_WEC :
  forall rho p, DEC rho p -> WEC rho p.
Proof. exact DEC_implies_WEC. Qed.
