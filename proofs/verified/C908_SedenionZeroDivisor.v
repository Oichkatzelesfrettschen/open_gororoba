(** * C-908: Sedenions have zero divisors.

    Explicit witness: the Moreno-Froloff pair (e3+e10) * (e6-e15) = 0.
    This proves sedenions do NOT form a division algebra.
    Verified component-wise on the concrete 16-component product. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

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
  unfold sed_zd_a, sed_zd_b, sed_mul, oct_mul, oct_conj,
         quat_mul, quat_add, quat_neg, quat_conj,
         oct_zero, quat_zero, quat_one, sed_zero.
  (* simpl reduces to mkSed(mkOct,mkOct) form but re-folds oct_zero/quat_zero;
     unfold them again so f_equal can decompose all 4 constructor layers. *)
  simpl. unfold oct_zero, quat_zero.
  f_equal;       (* mkSed: 2 CDOct goals *)
    f_equal;     (* mkOct: 4 CDQuat goals *)
      f_equal;   (* mkQuat: 16 R goals *)
        ring.    (* each 0*0-0*1+... = 0 *)
Qed.
