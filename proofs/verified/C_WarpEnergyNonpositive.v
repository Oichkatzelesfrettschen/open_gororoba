(** * C_WarpEnergyNonpositive: Alcubierre warp energy is nonpositive.

    Both the local energy density T_00 and the total warp bubble energy
    are nonpositive, confirming WEC violation by the Alcubierre metric. *)

From OpenGororoba Require Import Prelude QuantumInequalities.

Theorem claim_alcubierre_T00_nonpositive :
  forall v_s df_dr y z r_s,
  r_s > 0 ->
  alcubierre_T00_factor v_s df_dr y z r_s <= 0.
Proof. exact alcubierre_T00_nonpositive. Qed.

Theorem claim_warp_energy_nonpositive :
  forall beta r_bubble delta,
  delta > 0 ->
  warp_energy_factor beta r_bubble delta <= 0.
Proof. exact warp_energy_nonpositive. Qed.
