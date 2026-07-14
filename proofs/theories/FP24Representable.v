(* FP24 numeric refinement: any integer inside the exact window 2^17 is exactly
   representable in the FP24 significand, so the DP4 accumulator that the IDCT lane
   keeps inside 2^17 round-trips through FP24 without rounding.  This is the numeric
   half of the exactness claim; the integer window arithmetic is in
   IDCT8DP4ExactBound.v.

   FP24 on RS480 is s1e7m16 = 17 significand bits (16 stored + 1 implicit), a 7-bit
   exponent with bias 62 and normal range [2^-61, 2^65] (r300_numeric_domain.h).
   FLX radix2 precision 17 is the fixed-significand model: it drops the exponent
   range, which is sound for this window because an integer of magnitude at most
   2^17 stays far inside the normal range and never reaches the overflow or
   subnormal boundary.  ASCII only. *)
From Stdlib Require Import ZArith Reals Lia Bool.
From Flocq Require Import Core.
Open Scope Z_scope.

(* FP24 significand precision: 16 stored + 1 implicit. *)
Definition fp24_prec : Z := 17.

(* FP24 exact-integer window, inclusive: R300_MP_FP24_EXACT_INT = 131072 = 2^17. *)
Definition fp24_exact_int : Z := 2 ^ fp24_prec.

(* An integer strictly inside the window is in FLX(17). *)
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

(* The boundary 2^17 is itself exact (2^17 = 1.0 * 2^17), so the window is
   inclusive: every integer with |n| <= 2^17 is in FLX(17).  The equality case
   needs its own representation (bpow 17), one bit beyond the strict interior. *)
Theorem fp24_int_exact_inclusive :
  forall n : Z,
    Z.abs n <= 2 ^ fp24_prec ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros n Hn.
  destruct (Z.lt_ge_cases (Z.abs n) (2 ^ fp24_prec)) as [Hlt|Hge].
  - apply fp24_int_exact; exact Hlt.
  - assert (Habs : Z.abs n = 2 ^ fp24_prec) by lia.
    assert (Hgf : generic_format radix2 (FLX_exp fp24_prec) (bpow radix2 fp24_prec)).
    { apply generic_format_bpow. unfold FLX_exp, fp24_prec; lia. }
    assert (Hbp : bpow radix2 fp24_prec = IZR (2 ^ fp24_prec)).
    { rewrite <- IZR_Zpower by (unfold fp24_prec; lia). reflexivity. }
    destruct (Z_lt_le_dec n 0) as [Hneg|Hpos].
    + assert (Hn2 : n = - (2 ^ fp24_prec)).
      { rewrite Z.abs_neq in Habs by lia. lia. }
      rewrite Hn2, opp_IZR, <- Hbp.
      apply generic_format_opp. exact Hgf.
    + assert (Hn2 : n = 2 ^ fp24_prec).
      { rewrite Z.abs_eq in Habs by lia. lia. }
      rewrite Hn2, <- Hbp. exact Hgf.
Qed.

(* Production typed-carry admission predicates, faithful to r300_nir_ssa_cut.c
   lines 433-439 (R300_MP_FP24_EXACT_INT = 131072).  The classifier admits the
   boundary: unsigned declines only when the upper bound exceeds 131072; signed
   declines only when the interval leaves [-131072, 131072].  Both are inclusive.
   The strict 8*B^2 < 2^17 (B <= 127) rule in IDCT8DP4ExactBound.v is a separate
   conservative IDCT-operand policy, not this carry-value gate. *)
Definition uint_range_admit (hi : Z) : bool := Z.leb hi fp24_exact_int.
Definition sint_range_admit (lo hi : Z) : bool :=
  andb (Z.leb (- fp24_exact_int) lo) (Z.leb hi fp24_exact_int).
Definition fp24_value_admit (n : Z) : bool := Z.leb (Z.abs n) fp24_exact_int.

(* Every value an admitted signed interval can carry is FP24-exact. *)
Theorem sint_range_admit_exact :
  forall lo hi n : Z,
    sint_range_admit lo hi = true ->
    lo <= n <= hi ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros lo hi n Ha [Hl Hh].
  apply fp24_int_exact_inclusive.
  unfold sint_range_admit, fp24_exact_int in *.
  apply andb_true_iff in Ha. destruct Ha as [Ha1 Ha2].
  apply Z.leb_le in Ha1. apply Z.leb_le in Ha2.
  apply Z.abs_le. lia.
Qed.

(* Every value an admitted unsigned interval can carry (0 <= n <= hi) is exact. *)
Theorem uint_range_admit_exact :
  forall hi n : Z,
    uint_range_admit hi = true ->
    0 <= n <= hi ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros hi n Ha [Hl Hh].
  apply fp24_int_exact_inclusive.
  unfold uint_range_admit, fp24_exact_int in *.
  apply Z.leb_le in Ha.
  apply Z.abs_le. lia.
Qed.

(* The inclusive carry-value gate: an admitted value is FP24-exact. *)
Theorem fp24_value_admit_exact :
  forall n : Z,
    fp24_value_admit n = true ->
    generic_format radix2 (FLX_exp fp24_prec) (IZR n).
Proof.
  intros n H. apply fp24_int_exact_inclusive.
  unfold fp24_value_admit, fp24_exact_int in H. apply Z.leb_le in H. exact H.
Qed.
