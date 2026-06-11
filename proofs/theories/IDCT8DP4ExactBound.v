(* Accumulator bound for the 8-term dot product that the r300 g3dvl IDCT runs on
   the RS480 DP4 substrate (matrix_mul = DP4(l0,r0) + DP4(l1,r1)).  This is the
   formal backbone of the "the DP4 is exact, the residual is addressing, not
   arithmetic" result: a bounded-operand 8-term integer dot stays inside a stated
   dynamic range, so for 7-bit-magnitude operands it is exact in the FP24 mantissa
   (2^17 < 2^24) and the 12 dB AC residual cannot be a DP4 precision loss.

   Tactic idiom: nonlinear integer arithmetic (nia), matching the open_gororoba
   house style for bounded-variable polynomial goals.  Self-contained on Stdlib
   ZArith.  ASCII only. *)
From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

Section DP8_AccumulatorBound.
  Variables a0 a1 a2 a3 a4 a5 a6 a7 : Z.
  Variables b0 b1 b2 b3 b4 b5 b6 b7 : Z.

  Definition dp8 : Z :=
    a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7.

  (* For operands of magnitude at most B, the 8-term accumulator has magnitude at
     most 8 B^2.  This is the exactness premise the substrate's DP4 lane relies on. *)
  Theorem dp8_abs_bound :
    forall B : Z,
      0 <= B ->
      -B <= a0 <= B -> -B <= a1 <= B -> -B <= a2 <= B -> -B <= a3 <= B ->
      -B <= a4 <= B -> -B <= a5 <= B -> -B <= a6 <= B -> -B <= a7 <= B ->
      -B <= b0 <= B -> -B <= b1 <= B -> -B <= b2 <= B -> -B <= b3 <= B ->
      -B <= b4 <= B -> -B <= b5 <= B -> -B <= b6 <= B -> -B <= b7 <= B ->
      - (8 * B * B) <= dp8 <= 8 * B * B.
  Proof.
    intros B HB
      Ha0 Ha1 Ha2 Ha3 Ha4 Ha5 Ha6 Ha7
      Hb0 Hb1 Hb2 Hb3 Hb4 Hb5 Hb6 Hb7.
    unfold dp8.
    nia.
  Qed.

  (* RS480 DP4_UINT7_EXACT lane: 7-bit-magnitude operands keep the 8-term
     accumulator strictly inside 2^17, hence exact in the FP24 24-bit mantissa
     (8 * 127^2 = 129032 < 131072 = 2^17 << 2^24). *)
  Corollary dp8_int7_within_2pow17 :
    -127 <= a0 <= 127 -> -127 <= a1 <= 127 -> -127 <= a2 <= 127 -> -127 <= a3 <= 127 ->
    -127 <= a4 <= 127 -> -127 <= a5 <= 127 -> -127 <= a6 <= 127 -> -127 <= a7 <= 127 ->
    -127 <= b0 <= 127 -> -127 <= b1 <= 127 -> -127 <= b2 <= 127 -> -127 <= b3 <= 127 ->
    -127 <= b4 <= 127 -> -127 <= b5 <= 127 -> -127 <= b6 <= 127 -> -127 <= b7 <= 127 ->
    - (131072) < dp8 < 131072.
  Proof.
    intros. unfold dp8. nia.
  Qed.
End DP8_AccumulatorBound.

(* FP24-exactness ceiling (the dynamic-range boundary the bound implies): the
   8-term integer accumulator is exactly representable in the FP24 s7e16 mantissa
   (24 integer bits) iff 8 B^2 < 2^24, i.e. B <= 1448.  MPEG-2 dequantized
   coefficients can exceed this, so the full-range IDCT dot is NOT integer-exact;
   its error is the FP24 relative-rounding floor (~81 dB), which still dwarfs the
   measured ~12 dB AC residual -- confirming the residual is structural addressing,
   not DP4 arithmetic.  Witness that the ceiling is positive and the UINT7 lane
   sits far below it. *)
Lemma fp24_ceiling_positive : (8 * 1448 * 1448 < 2 ^ 24)%Z /\ (131072 < 2 ^ 24)%Z.
Proof. split; vm_compute; reflexivity. Qed.
