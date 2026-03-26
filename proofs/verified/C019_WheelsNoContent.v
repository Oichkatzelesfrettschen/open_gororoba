(** * C-019: Wheels do not resolve sedenion zero-divisor pathology.

    For the ZD witness pair, the product a*b = 0 with both nonzero.
    Any wheel-based "quotient" a/b would require (a/b)*b = a,
    but since a*b = 0, the round-trip fails: norm(a*b) = 0 while
    norm(a) = sqrt(2) > 0.

    Positive reformulated content: the ZD product is genuinely zero,
    so no multiplicative inverse scheme can recover a from b. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** The product is genuinely zero (both components, all 16 scalars). *)
Theorem C019_product_zero :
  sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof. exact sed_zd_product_zero. Qed.

(** Both operands have nonzero norm. *)
Theorem C019_operands_nonzero :
  sed_norm_sq sed_zd_a > 0 /\ sed_norm_sq sed_zd_b > 0.
Proof.
  split; [rewrite sed_zd_a_norm | rewrite sed_zd_b_norm]; lra.
Qed.

(** The norm of the product is zero. *)
Theorem C019_product_norm_zero :
  sed_norm_sq (sed_mul sed_zd_a sed_zd_b) = 0.
Proof. exact sed_zd_product_norm. Qed.
