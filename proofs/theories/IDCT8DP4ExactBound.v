(* Accumulator bound for the 8-term dot product that the r300 g3dvl IDCT runs on
   the RS480 DP4 substrate (matrix_mul = DP4(l0,r0) + DP4(l1,r1)).  This is the
   formal backbone of the "the DP4 is exact, the residual is addressing, not
   arithmetic" result: a bounded-operand 8-term integer dot stays inside the FP24
   exact-integer window, so for 7-bit-magnitude operands it round-trips exactly
   through the fragment ALU and the 12 dB AC residual cannot be a DP4 precision
   loss.

   FP24 on RS480 is s1e7m16: 1 sign, 7 exponent, 16 stored mantissa bits, hence
   17 significand bits (one implicit).  Every integer of magnitude at most
   2^17 = 131072 is exactly representable, and 2^17 + 1 is the first gap.  The
   17-bit significand sets the exact-integer window at 2^17.

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
     accumulator strictly inside the FP24 exact-integer window 2^17, hence exact
     in the 17-bit FP24 significand (8 * 127^2 = 129032 < 131072 = 2^17). *)
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

(* FP24 exact-integer window and the accumulator's exactness threshold.

   Every integer of magnitude at most 2^17 is exactly representable in the FP24
   s1e7m16 significand (17 bits), and 2^17 = 131072 itself is exact (1.0 * 2^17).
   The 8-term accumulator is guaranteed exact whenever its worst case stays inside
   the window, 8 * B^2 <= 2^17, which holds for B <= 128: 8 * 128^2 = 131072 = 2^17
   is the boundary and still exact, while 8 * 129^2 = 133128 > 2^17 leaves the
   guarantee.  The numeric-representability half (|n| <= 2^17 -> FP24-exact) is
   proved against Flocq in FP24Representable.v; here we fix the integer window. *)
Lemma fp24_window : 2 ^ 17 = 131072.
Proof. vm_compute; reflexivity. Qed.

(* The exactness boundary is tight at B = 128: the worst-case accumulator equals
   2^17 exactly, and one operand step past leaves the window. *)
Lemma dp8_exact_boundary : 8 * 128 * 128 = 2 ^ 17 /\ 2 ^ 17 < 8 * 129 * 129.
Proof. split; vm_compute; reflexivity. Qed.

(* Exactness threshold as an iff over nonnegative operand magnitude B. *)
Lemma dp8_exact_threshold :
  forall B : Z, 0 <= B -> (8 * B * B <= 2 ^ 17 <-> B <= 128).
Proof. intros B HB. change (2 ^ 17) with 131072. nia. Qed.

(* The r300 R2VB admission gate keeps the strict form 8 * B^2 < 2^17 (B <= 127):
   a fail-closed choice that declines the exact-but-boundary B = 128 rather than
   asserting it inexact.  R300_MP_FP24_EXACT_INT = 131072 in r300_numeric_domain.c. *)
Definition fp24_admit_strict (B : Z) : bool := (8 * B * B <? 131072)%Z.

Lemma fp24_admit_strict_spec :
  forall B : Z, 0 <= B -> (fp24_admit_strict B = true <-> B <= 127).
Proof. intros B HB. unfold fp24_admit_strict. rewrite Z.ltb_lt. nia. Qed.
