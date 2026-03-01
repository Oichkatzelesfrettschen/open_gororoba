(** * C-868: Painleve-Gullstrand ADM decomposition.

    Formal proofs about the PG coordinate system for Schwarzschild:
    1. Lapse = 1 (alpha^2 = 1 for all M > 0, r > 0)
    2. Shift is non-negative (beta^r = sqrt(2M/r) >= 0)
    3. Spatial metric radial component is flat (gamma_{rr} = 1)

    Mirrors: PainleveGullstrand in adm.rs:527-573.
    Rust tests: test_painleve_lapse_is_unity (line 710-721),
    test_painleve_shift_is_infall_velocity (line 724-743),
    test_painleve_spatial_metric_is_flat (line 746-771). *)

From OpenGororoba Require Import Prelude ADM.

(** CLAIM C-868a: PG lapse is unity.

    alpha^2 = sqrt(2M/r)^2 + 1 - 2M/r = 2M/r + 1 - 2M/r = 1.

    Uses sqrt_sqrt: for x >= 0, sqrt(x) * sqrt(x) = x. *)
Theorem C868_pg_lapse_unity :
  forall M r : R, M > 0 -> r > 0 ->
    pg_lapse_sq M r = 1.
Proof.
  intros M r HM Hr.
  unfold pg_lapse_sq.
  assert (H : 0 <= 2 * M / r).
  { unfold Rdiv. apply Rle_mult_inv_pos. lra. lra. }
  rewrite sqrt_sqrt by exact H.
  lra.
Qed.

(** CLAIM C-868b: PG shift is non-negative. *)
Theorem C868_pg_shift_nonneg :
  forall M r : R, M > 0 -> r > 0 ->
    pg_shift_r M r >= 0.
Proof.
  intros M r HM Hr.
  unfold pg_shift_r.
  apply Rle_ge.
  apply sqrt_pos.
Qed.

(** CLAIM C-868c: PG spatial metric radial component is 1 (flat). *)
Theorem C868_pg_spatial_flat :
  pg_gamma_rr = 1.
Proof.
  unfold pg_gamma_rr. reflexivity.
Qed.

(** Corollary: the lapse is exactly 1 (not just alpha^2 = 1). *)
Theorem C868_pg_lapse_is_one :
  forall M r : R, M > 0 -> r > 0 ->
    sqrt (pg_lapse_sq M r) = 1.
Proof.
  intros M r HM Hr.
  rewrite C868_pg_lapse_unity by assumption.
  exact sqrt_1.
Qed.
