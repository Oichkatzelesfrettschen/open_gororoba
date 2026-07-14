(** * C-1629: The r300 IDCT DP4 accumulator stays inside the FP24 exact range.

    The r300 g3dvl IDCT evaluates each output as an 8-term integer dot product on
    the fragment DP4 lane. Three facts bound its arithmetic exactness:

    - dp8_abs_bound: for operands of magnitude at most B, the 8-term accumulator
      has magnitude at most 8*B^2.
    - dp8_int7_within_2pow17: 7-bit-magnitude operands (|.| <= 127) keep the
      accumulator strictly inside the FP24 exact-integer window 2^17
      (8 * 127^2 = 129032 < 131072), hence exact in the 17-bit FP24 significand.
    - dp8_exact_threshold: the accumulator is guaranteed exact for operand
      magnitude B <= 128, since 8 * 128^2 = 131072 = 2^17 is the boundary and
      still representable.  The r300 admission gate keeps the strict B <= 127 as a
      fail-closed choice (fp24_admit_strict_spec).

    FP24 on RS480 is s1e7m16 (17 significand bits), so the exact-integer window is
    2^17.  Together these prove the DP4 IDCT lane
    is integer-exact on the 7-bit path and bounded elsewhere -- so the measured AC
    residual is structural addressing, not DP4 arithmetic (steinmarder GATE-0
    verdict). Proved in IDCT8DP4ExactBound by nia / vm_compute over the integers,
    with no axioms. *)

From OpenGororoba Require Import IDCT8DP4ExactBound.

Definition C1629_dp8_abs_bound := dp8_abs_bound.
Definition C1629_dp8_int7_within_2pow17 := dp8_int7_within_2pow17.
Definition C1629_fp24_window := fp24_window.
Definition C1629_dp8_exact_boundary := dp8_exact_boundary.
Definition C1629_dp8_exact_threshold := dp8_exact_threshold.
Definition C1629_fp24_admit_strict_spec := fp24_admit_strict_spec.
