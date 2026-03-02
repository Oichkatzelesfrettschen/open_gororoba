(** * C-891: Gravity Probe B geodetic bound requires omega > 177.

    |gamma_BD - 1| < 5.6e-3 when omega > 177.
    gamma_BD - 1 = -1/(2+omega), so |gamma_BD - 1| = 1/(2+omega).
    Cross-multiply: 10000 < 56*(2+omega) iff omega > (10000/56 - 2) = 176.57... *)

From OpenGororoba Require Import Prelude BransDicke.

(** Cross-multiplication for positive-denominator fractions. *)
Lemma frac_lt : forall a b c d,
  b > 0 -> d > 0 -> a * d < c * b -> a / b < c / d.
Proof.
  intros a b c d Hb Hd Hlt.
  unfold Rdiv.
  apply (Rmult_lt_reg_r b); [lra |].
  replace (a * / b * b) with a by (field; lra).
  apply (Rmult_lt_reg_r d); [lra |].
  replace (c * / d * b * d) with (c * b) by (field; lra).
  lra.
Qed.

Theorem C891_gpb_geodetic : forall omega,
  omega > 177 -> Rabs (ppn_gamma_bd omega - 1) < 56 / 10000.
Proof.
  intros omega H.
  unfold ppn_gamma_bd.
  assert (Hd : 2 + omega > 0) by lra.
  replace ((1 + omega) / (2 + omega) - 1) with (- (1 / (2 + omega))) by (field; lra).
  rewrite Rabs_Ropp.
  rewrite Rabs_pos_eq.
  - apply frac_lt; [lra | lra |]. nra.
  - unfold Rdiv. apply Rmult_le_pos; [lra |]. left. apply Rinv_pos. lra.
Qed.
