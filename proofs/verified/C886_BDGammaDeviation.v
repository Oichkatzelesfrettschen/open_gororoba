(** * C-886: BD gamma deviation equals negative Nordtvedt parameter.

    ppn_gamma_bd(omega) - 1 = -1/(2+omega) = -nordtvedt_bd(omega).
    Mirrors C-824 in registry. *)

From OpenGororoba Require Import Prelude BransDicke.

Theorem C886_bd_gamma_deviation : forall omega,
  omega > 0 -> ppn_gamma_deviation omega = - nordtvedt_bd omega.
Proof.
  intros omega H. unfold ppn_gamma_deviation, ppn_gamma_bd, nordtvedt_bd.
  field. lra.
Qed.
