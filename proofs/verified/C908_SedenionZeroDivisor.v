(** * C-908: Sedenions have zero divisors.

    Explicit witness: the Moreno-Froloff pair (e3+e10) * (e6-e15) = 0.
    This proves sedenions do NOT form a division algebra.
    Verified component-wise on the concrete 16-component product. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** The zero-divisor witnesses are nonzero. *)
Theorem sed_zd_a_nonzero : sed_zd_a <> sed_zero.
Proof.
  unfold sed_zd_a, sed_zero, oct_zero, quat_zero.
  intro H.
  assert (Hlo := f_equal sed_lo H). simpl in Hlo.
  assert (Hq := f_equal oct_lo Hlo). simpl in Hq.
  assert (Hd := f_equal qd Hq). simpl in Hd. lra.
Qed.

Theorem sed_zd_b_nonzero : sed_zd_b <> sed_zero.
Proof.
  unfold sed_zd_b, sed_zero, oct_zero, quat_zero.
  intro H.
  assert (Hhi := f_equal sed_hi H). simpl in Hhi.
  assert (Hq := f_equal oct_hi Hhi). simpl in Hq.
  assert (Hd := f_equal qd Hq). simpl in Hd. lra.
Qed.

(** The product of the witnesses is zero.
    We reduce each of the 16 R-valued components and verify with ring. *)
Theorem C908_sedenion_zero_divisor :
  sed_mul sed_zd_a sed_zd_b = sed_zero.
Proof.
  exact sed_zd_product_zero.
Qed.
