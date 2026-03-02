(** * C-888: BD gamma is strictly increasing for omega > 0.

    d/d(omega) ppn_gamma_bd = 1/(2+omega)^2 > 0.
    Cross-multiplication reduces to (1+w1)*(2+w2) < (1+w2)*(2+w1),
    which simplifies to omega1 < omega2 (linear). *)

From OpenGororoba Require Import Prelude BransDicke.

(** Helper: a/b < c/d iff a*d < c*b for positive b, d. *)
Lemma div_lt_div : forall a b c d,
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

Theorem C888_bd_gamma_monotone : forall omega1 omega2,
  omega1 > 0 -> omega2 > omega1 ->
  ppn_gamma_bd omega1 < ppn_gamma_bd omega2.
Proof.
  intros omega1 omega2 H1 H12.
  unfold ppn_gamma_bd.
  apply div_lt_div; lra.
Qed.
