(** * Calcagni spectral dimension: range and monotonicity.

    Formalizes d_S(s) = 4 - 2/(1+s) from Calcagni PRL 104 (2010) 251301.
    The variable s = (k/k_star)^{-alpha} > 0 parametrizes the momentum scale.

    Key properties:
    - d_S(s) in (2, 4) for all s > 0
    - d_S is strictly increasing: s1 < s2 => d_S(s1) < d_S(s2)

    Mirrors: cosmology_core::spectral::calcagni_spectral_dimension *)

From OpenGororoba Require Import Prelude.

(** Calcagni's running spectral dimension.
    s > 0 is the dimensionless scale variable (k/k_star)^{-alpha}. *)
Definition calcagni_d_s (s : R) : R := 4 - 2 / (1 + s).

(** Range theorem: d_S(s) in (2, 4) for all s > 0. *)
Theorem calcagni_range : forall s : R, s > 0 -> 2 < calcagni_d_s s < 4.
Proof.
  intros s Hs.
  unfold calcagni_d_s.
  assert (H1s : 1 + s > 0) by lra.
  assert (H1s_ne : 1 + s <> 0) by lra.
  split.
  - (* 4 - 2/(1+s) > 2  <=>  2 > 2/(1+s)  <=>  2*(1+s) > 2  <=>  s > 0 *)
    cut (2 / (1 + s) < 2).
    { lra. }
    rewrite <- (Rmult_1_r 2) at 2.
    unfold Rdiv.
    apply Rmult_lt_compat_l; [lra |].
    rewrite <- Rinv_1.
    apply Rinv_lt_contravar; lra.
  - (* 4 - 2/(1+s) < 4  <=>  2/(1+s) > 0 *)
    cut (2 / (1 + s) > 0).
    { lra. }
    unfold Rdiv.
    apply Rmult_lt_0_compat; [lra |].
    apply Rinv_0_lt_compat. lra.
Qed.

(** Monotonicity: d_S is strictly increasing for s > 0.
    Larger s (deeper IR) gives larger d_S (approaching 4). *)
Theorem calcagni_decreasing :
  forall s1 s2 : R, 0 < s1 -> s1 < s2 ->
  calcagni_d_s s1 < calcagni_d_s s2.
Proof.
  intros s1 s2 Hs1 Hs12.
  unfold calcagni_d_s.
  assert (H1 : 1 + s1 > 0) by lra.
  assert (H2 : 1 + s2 > 0) by lra.
  assert (H3 : 1 + s1 < 1 + s2) by lra.
  (* 4 - 2/(1+s1) < 4 - 2/(1+s2)  <=>  2/(1+s2) < 2/(1+s1) *)
  cut (2 / (1 + s2) < 2 / (1 + s1)).
  { lra. }
  unfold Rdiv.
  apply Rmult_lt_compat_l; [lra |].
  apply Rinv_lt_contravar.
  - apply Rmult_lt_0_compat; lra.
  - lra.
Qed.
