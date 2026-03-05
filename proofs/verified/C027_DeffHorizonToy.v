(** * C-027: D_eff horizon toy model.

    D_eff(rho) = 3 - k * ln(rho / rho_vac) is strictly decreasing
    in rho for k > 0.

    We formalize: for fixed k > 0, the function f(rho) = -k * ln(rho)
    is strictly decreasing (since ln is strictly increasing and k > 0).

    The D_eff = 2 horizon occurs when rho = rho_vac * exp(1/k). *)

From Stdlib Require Import Reals Lra Rpower.
Open Scope R_scope.

Definition D_eff (k rho rho_vac : R) : R :=
  3 - k * ln (rho / rho_vac).

(** D_eff = 3 at rho = rho_vac. *)
Theorem C027_vacuum_value : forall k rho_vac : R,
  rho_vac > 0 ->
  D_eff k rho_vac rho_vac = 3.
Proof.
  intros k rho_vac Hrv.
  unfold D_eff. unfold Rdiv.
  rewrite Rinv_r by lra. rewrite ln_1. ring.
Qed.

(** D_eff is strictly decreasing in rho for k > 0. *)
Theorem C027_decreasing : forall k rho1 rho2 rho_vac : R,
  k > 0 -> rho_vac > 0 -> rho1 > 0 -> rho2 > 0 ->
  rho1 < rho2 ->
  D_eff k rho2 rho_vac < D_eff k rho1 rho_vac.
Proof.
  intros k rho1 rho2 rho_vac Hk Hrv Hr1 Hr2 Hlt.
  unfold D_eff.
  assert (Hpos1 : rho1 / rho_vac > 0).
  { unfold Rdiv. apply Rmult_lt_0_compat; [lra | apply Rinv_0_lt_compat; lra]. }
  assert (Hord : rho1 / rho_vac < rho2 / rho_vac).
  { unfold Rdiv. apply Rmult_lt_compat_r; [apply Rinv_0_lt_compat; lra | lra]. }
  assert (Hln : ln (rho1 / rho_vac) < ln (rho2 / rho_vac)).
  { apply ln_increasing; lra. }
  nra.
Qed.
