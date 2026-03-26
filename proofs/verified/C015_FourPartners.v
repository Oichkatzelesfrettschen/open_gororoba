(** * C-015: Each zero-divisor has exactly 4 annihilating partners.

    The annihilator of a sedenion ZD element is a 4-dimensional
    linear subspace of the 16-dimensional sedenion space.
    Here we verify: the partner count is at least 1 (sed_zd_b for sed_zd_a)
    and at most 15 (not all basis elements annihilate).

    The exact dimension=4 is verified computationally in Rust via SVD.
    Rocq scope: algebraic structure of the annihilator set. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** sed_zd_a * sed_zd_a is NOT zero (not a self-annihilator). *)
Theorem C015_not_self_annihilating :
  sed_mul sed_zd_a sed_zd_a <> sed_zero.
Proof.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  assert (Hq := f_equal oct_lo Hlo).
  assert (Ha := f_equal qa Hq).
  cbv [sed_mul sed_zd_a oct_mul oct_conj
       quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero quat_one
       sed_zero qa qb qc qd] in Ha.
  lra.
Qed.

(** sed_zd_b is a valid partner (product = 0). *)
Theorem C015_partner_exists :
  sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof. exact sed_zd_product_zero. Qed.
