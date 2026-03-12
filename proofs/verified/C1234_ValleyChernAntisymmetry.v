(** * C-1234: Valley Chern number antisymmetry VCN(K) = -VCN(K').

    CLAIM: When the total Chern number is zero (as guaranteed by TRS
    via C-1233), the valley Chern numbers at K and K' are equal in
    magnitude and opposite in sign: VCN(K) = -VCN(K').

    STRATEGY: From the BZ splitting axiom
      C_total = VCN(K) + VCN(K')
    and the TRS valley-swap axiom
      VCN(K') = -VCN(K)
    we get the result directly.

    Mirrors: crates/quantum_core/src/tight_binding.rs (valley_chern_number)
    Binary:  magnonic-band-structure --chern *)

From OpenGororoba Require Import Prelude.
Require Import OpenGororoba.TightBindingTopology.

Open Scope R_scope.

(** MAIN THEOREM: Under TRS, VCN(K') = -VCN(K). *)
Theorem valley_chern_antisymmetry :
  forall (F : berry_curvature),
    has_trs F ->
    valley_integral_Kp F = - valley_integral_K F.
Proof.
  intros F Htrs.
  exact (trs_swaps_valleys F Htrs).
Qed.

(** Corollary: The sum of valley Chern numbers equals the total
    Chern number (which is zero under TRS). *)
Theorem valley_chern_sum_zero :
  forall (F : berry_curvature),
    has_trs F ->
    valley_integral_K F + valley_integral_Kp F = 0.
Proof.
  intros F Htrs.
  rewrite (valley_chern_antisymmetry F Htrs).
  lra.
Qed.

(** Corollary: The total Chern number splits correctly. *)
Theorem chern_valley_split_consistent :
  forall (F : berry_curvature),
    has_trs F ->
    chern_integral F = valley_integral_K F + valley_integral_Kp F.
Proof.
  intros F Htrs.
  exact (chern_splits_into_valleys F).
Qed.
