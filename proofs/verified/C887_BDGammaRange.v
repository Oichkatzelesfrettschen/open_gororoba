(** * C-887: BD gamma is in (1/2, 1) for omega > 0.

    ppn_gamma_bd(omega) = (1+omega)/(2+omega) lies strictly between
    1/2 (at omega=0) and 1 (the GR limit at omega=inf). *)

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

Theorem C887_bd_gamma_range : forall omega,
  omega > 0 -> 1/2 < ppn_gamma_bd omega < 1.
Proof.
  intros omega H. unfold ppn_gamma_bd.
  assert (Hd : 2 + omega > 0) by lra.
  split.
  - (* 1/2 < (1+omega)/(2+omega) *)
    apply frac_lt; [lra | lra |]. nra.
  - (* (1+omega)/(2+omega) < 1 *)
    apply (Rmult_lt_reg_r (2 + omega)); [lra |].
    replace ((1 + omega) / (2 + omega) * (2 + omega)) with (1 + omega) by (field; lra).
    lra.
Qed.
