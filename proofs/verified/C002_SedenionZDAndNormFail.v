(** * C-002: Sedenions have zero divisors AND norm multiplicativity fails.

    Two independent pathologies at dim 16:
    1. Zero divisors exist: sed_zd_a * sed_zd_b = 0 with both nonzero.
    2. Hurwitz fails: |x*y|^2 <> |x|^2 * |y|^2 for some x, y.

    These are logically independent facts, but the zero divisors
    provide the witness for the Hurwitz failure:
    |a*b|^2 = 0 but |a|^2 * |b|^2 = 4.

    Reformulated positive content of C-002. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C908_SedenionZeroDivisor.

(** Zero divisors exist in the sedenions. *)
Theorem C002_zero_divisors :
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_zd_a <> sed_zero /\
  sed_zd_b <> sed_zero.
Proof.
  split; [| split].
  - exact C908_sedenion_zero_divisor.
  - exact sed_zd_a_nonzero.
  - exact sed_zd_b_nonzero.
Qed.

(** Norm multiplicativity fails at dim 16. *)
Theorem C002_norm_fails :
  exists x y : CDSed,
    sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y.
Proof. exact sed_norm_fails. Qed.
