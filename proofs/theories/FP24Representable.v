(* FP24 numeric refinement: any integer strictly inside the exact window 2^17 is
   exactly representable in the FP24 significand, so the DP4 accumulator that the
   IDCT lane keeps inside 2^17 (dp8_int7_within_2pow17) round-trips through FP24
   without rounding.  This is the numeric half of the exactness claim; the integer
   window arithmetic is in IDCT8DP4ExactBound.v.

   FP24 on RS480 is s1e7m16 = 17 significand bits (16 stored + 1 implicit).  FLX
   radix2 precision 17 is the fixed-significand model: it drops the exponent range,
   which is sound for this window because an integer with |n| < 2^17 stays far
   inside FP24's normal range (bias 63) and never reaches the overflow or
   subnormal boundary.  ASCII only. *)
From Stdlib Require Import ZArith Reals Lia.
From Flocq Require Import Core.
Open Scope Z_scope.

(* FP24 significand precision: 16 stored + 1 implicit. *)
Definition fp24_prec : Z := 17.

(* An integer strictly inside the FP24 exact-integer window is in FLX(17). *)
Theorem fp24_int_exact :
  forall n : Z,
    Z.abs n < 2 ^ fp24_prec ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros n Hn.
  destruct (Z.eq_dec n 0) as [->|Hnz].
  - apply generic_format_0.
  - assert (Hb : (Rabs (IZR n) < bpow radix2 fp24_prec)%R).
    { rewrite <- abs_IZR.
      rewrite <- IZR_Zpower by (unfold fp24_prec; lia).
      apply IZR_lt. exact Hn. }
    replace (IZR n) with (F2R (Float radix2 n 0)) by
      (unfold F2R; simpl; ring).
    apply generic_format_F2R.
    intros _.
    unfold cexp, FLX_exp.
    assert (Hmag : (mag radix2 (F2R (Float radix2 n 0)) <= fp24_prec)%Z).
    { apply mag_le_bpow.
      - replace (F2R (Float radix2 n 0)) with (IZR n) by
          (unfold F2R; simpl; ring).
        apply not_0_IZR. exact Hnz.
      - replace (F2R (Float radix2 n 0)) with (IZR n) by
          (unfold F2R; simpl; ring).
        exact Hb. }
    lia.
Qed.

(* The FP24 admission gate value bound: the r300 classifier admits a carry value
   n exactly when |n| < R300_MP_FP24_EXACT_INT, and every admitted value is
   FP24-exact by fp24_int_exact.  (r300_numeric_domain.c: exact_int_bound
   131072 = 2^17.) *)
Definition fp24_value_admit (n : Z) : bool := (Z.abs n <? 2 ^ fp24_prec)%Z.

Theorem fp24_value_admit_exact :
  forall n : Z,
    fp24_value_admit n = true ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros n H. apply fp24_int_exact.
  unfold fp24_value_admit in H. apply Z.ltb_lt in H. exact H.
Qed.
