(** * C-883: Calcagni spectral dimension range and monotonicity.

    Proves that Calcagni's d_S(s) = 4 - 2/(1+s) has range (2, 4) and is
    strictly increasing for s > 0. This makes claims about d_S values
    outside (2, 4) provably false under the Calcagni ansatz.

    In particular, refuted C-040 claimed D_eff ~ 2.8 implies n_s ~ 0.965.
    The monotonicity theorem shows this would require a UNIQUE s value
    (s ~ 2/3), not a range of possibilities as C-040 assumed.

    Reformulation of refuted C-040. *)

From OpenGororoba Require Import Prelude SpectralDimension.

(** C-883: d_S(s) in (2, 4) for all s > 0. *)
Theorem C883_range : forall s : R, s > 0 -> 2 < calcagni_d_s s < 4.
Proof. exact calcagni_range. Qed.

(** C-883: d_S is strictly increasing (larger s => larger d_S). *)
Theorem C883_monotone : forall s1 s2 : R,
  0 < s1 -> s1 < s2 -> calcagni_d_s s1 < calcagni_d_s s2.
Proof. exact calcagni_decreasing. Qed.

(** Corollary: d_S is injective on (0, infinity).
    If d_S(s1) = d_S(s2) and both s1, s2 > 0, then s1 = s2. *)
Theorem C883_injective : forall s1 s2 : R,
  0 < s1 -> 0 < s2 ->
  calcagni_d_s s1 = calcagni_d_s s2 -> s1 = s2.
Proof.
  intros s1 s2 Hs1 Hs2 Heq.
  destruct (Rtotal_order s1 s2) as [Hlt | [Heq12 | Hgt]].
  - (* s1 < s2 => d_S(s1) < d_S(s2), contradicts equality *)
    exfalso.
    assert (calcagni_d_s s1 < calcagni_d_s s2) by (apply calcagni_decreasing; lra).
    lra.
  - exact Heq12.
  - (* s2 < s1 => d_S(s2) < d_S(s1), contradicts equality *)
    exfalso.
    assert (calcagni_d_s s2 < calcagni_d_s s1) by (apply calcagni_decreasing; lra).
    lra.
Qed.
