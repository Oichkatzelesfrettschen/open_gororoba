(** * C-1629: The r300 IDCT DP4 accumulator stays inside the FP24 exact range.

    The r300 g3dvl IDCT evaluates each output as an 8-term integer dot product on
    the fragment DP4 lane. Three facts bound its arithmetic exactness:

    - dp8_abs_bound: for operands of magnitude at most B, the 8-term accumulator
      has magnitude at most 8*B^2.
    - dp8_int7_within_2pow17: 7-bit-magnitude operands (|.| <= 127) keep the
      accumulator strictly inside 2^17 (8 * 127^2 = 129032 < 131072), hence exact
      in the FP24 24-bit mantissa.
    - fp24_ceiling_positive: the FP24 integer-exact ceiling 8*1448^2 < 2^24 is
      positive and the UINT7 lane sits far below it.

    Together these prove the DP4 IDCT lane is integer-exact on the 7-bit path and
    bounded elsewhere -- so the measured AC residual is structural addressing, not
    DP4 arithmetic (steinmarder GATE-0 verdict). Proved in IDCT8DP4ExactBound by
    nia / vm_compute over the integers, with no axioms. *)

From OpenGororoba Require Import IDCT8DP4ExactBound.

Definition C1629_dp8_abs_bound := dp8_abs_bound.
Definition C1629_dp8_int7_within_2pow17 := dp8_int7_within_2pow17.
Definition C1629_fp24_ceiling_positive := fp24_ceiling_positive.
