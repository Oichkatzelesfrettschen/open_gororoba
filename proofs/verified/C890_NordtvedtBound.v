(** * C-890: Nordtvedt parameter bound requires omega > 1998.

    nordtvedt_bd(omega) = 1/(2+omega) < 5e-4 when omega > 1998.
    Cross-multiply: 10000 < 5*(2+omega) = 10 + 5*omega, so omega > 1998. *)

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

Theorem C890_nordtvedt_bound : forall omega,
  omega > 1998 -> nordtvedt_bd omega < 5 / 10000.
Proof.
  intros omega H.
  unfold nordtvedt_bd.
  assert (Hd : 2 + omega > 0) by lra.
  apply frac_lt; [lra | lra |]. nra.
Qed.
