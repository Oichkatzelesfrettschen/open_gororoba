(* R2VB vertex-transform exactness: the producer applies a 4x4 matrix to a
   4-vector as four independent DP4 lanes, one per output component, and each
   lane is a 4-term integer dot product.  For operand magnitude at most B the
   lane accumulator has magnitude at most 4 B^2, so with 4 B^2 <= 2^17 every
   product, every partial sum, and the lane result sits inside the FP24
   exact-integer window and the transform rounds at no step.  The threshold is
   B <= 181 (4 * 181^2 = 131044 <= 131072 = 2^17; 4 * 182^2 = 132496 leaves
   the window).  The numeric half (|n| <= 2^17 -> FLX(17) exact) comes from
   FP24Representable.v; the 8-term IDCT accumulator policy lives in
   IDCT8DP4ExactBound.v and stays distinct: its strict 8 B^2 lane admits
   B <= 127, this 4-term lane admits B <= 181.  ASCII only. *)
From Stdlib Require Import ZArith Reals Lia Bool.
From Flocq Require Import Core.
From OpenGororoba Require Import FP24Representable.
Open Scope Z_scope.

(* One DP4 lane: the 4-term integer dot product. *)
Definition dp4 (a0 a1 a2 a3 b0 b1 b2 b3 : Z) : Z :=
  a0*b0 + a1*b1 + a2*b2 + a3*b3.

(* Extractable admission gate for one DP4 lane: operand magnitude B keeps the
   worst-case accumulator inside the inclusive FP24 window. *)
Definition dp4_operand_admit (B : Z) : bool :=
  andb (Z.leb 0 B) (Z.leb (4 * B * B) fp24_exact_int).

Theorem dp4_abs_bound :
  forall B a0 a1 a2 a3 b0 b1 b2 b3 : Z,
    0 <= B ->
    -B <= a0 <= B -> -B <= a1 <= B -> -B <= a2 <= B -> -B <= a3 <= B ->
    -B <= b0 <= B -> -B <= b1 <= B -> -B <= b2 <= B -> -B <= b3 <= B ->
    - (4 * B * B) <= dp4 a0 a1 a2 a3 b0 b1 b2 b3 <= 4 * B * B.
Proof. intros. unfold dp4. nia. Qed.

(* The 4-term exactness threshold: 4 B^2 stays inside the inclusive window
   exactly for B <= 181. *)
Lemma dp4_exact_threshold :
  forall B : Z, 0 <= B -> (4 * B * B <= 2 ^ 17 <-> B <= 181).
Proof. intros B HB. change (2 ^ 17) with 131072. nia. Qed.

(* An admitted lane is FP24-exact: the gate bounds the accumulator inside the
   inclusive window, and every such integer is in FLX(17). *)
Theorem dp4_operand_admit_exact :
  forall B a0 a1 a2 a3 b0 b1 b2 b3 : Z,
    dp4_operand_admit B = true ->
    -B <= a0 <= B -> -B <= a1 <= B -> -B <= a2 <= B -> -B <= a3 <= B ->
    -B <= b0 <= B -> -B <= b1 <= B -> -B <= b2 <= B -> -B <= b3 <= B ->
    generic_format radix2 (FLX_exp fp24_prec)
      (IZR (dp4 a0 a1 a2 a3 b0 b1 b2 b3)).
Proof.
  intros B a0 a1 a2 a3 b0 b1 b2 b3 Hadmit Ha0 Ha1 Ha2 Ha3 Hb0 Hb1 Hb2 Hb3.
  unfold dp4_operand_admit in Hadmit.
  apply andb_true_iff in Hadmit. destruct Hadmit as [HB Hwin].
  apply Z.leb_le in HB. apply Z.leb_le in Hwin.
  apply fp24_int_exact_inclusive.
  unfold fp24_exact_int in Hwin.
  pose proof (dp4_abs_bound B a0 a1 a2 a3 b0 b1 b2 b3 HB
                Ha0 Ha1 Ha2 Ha3 Hb0 Hb1 Hb2 Hb3) as Hbound.
  apply Z.abs_le. lia.
Qed.

(* The four-lane matrix apply: each output component is one DP4 of a matrix
   row against the vertex.  Rows share no accumulator, so lane exactness is
   the transform's exactness. *)
Definition mvp4 (m00 m01 m02 m03 m10 m11 m12 m13
                 m20 m21 m22 m23 m30 m31 m32 m33
                 v0 v1 v2 v3 : Z) : (Z * Z * Z * Z) :=
  (dp4 m00 m01 m02 m03 v0 v1 v2 v3,
   dp4 m10 m11 m12 m13 v0 v1 v2 v3,
   dp4 m20 m21 m22 m23 v0 v1 v2 v3,
   dp4 m30 m31 m32 m33 v0 v1 v2 v3).

Theorem mvp4_rows_exact :
  forall B m00 m01 m02 m03 m10 m11 m12 m13
           m20 m21 m22 m23 m30 m31 m32 m33 v0 v1 v2 v3 : Z,
    dp4_operand_admit B = true ->
    -B <= m00 <= B -> -B <= m01 <= B -> -B <= m02 <= B -> -B <= m03 <= B ->
    -B <= m10 <= B -> -B <= m11 <= B -> -B <= m12 <= B -> -B <= m13 <= B ->
    -B <= m20 <= B -> -B <= m21 <= B -> -B <= m22 <= B -> -B <= m23 <= B ->
    -B <= m30 <= B -> -B <= m31 <= B -> -B <= m32 <= B -> -B <= m33 <= B ->
    -B <= v0 <= B -> -B <= v1 <= B -> -B <= v2 <= B -> -B <= v3 <= B ->
    let r := mvp4 m00 m01 m02 m03 m10 m11 m12 m13
                  m20 m21 m22 m23 m30 m31 m32 m33 v0 v1 v2 v3 in
    generic_format radix2 (FLX_exp fp24_prec) (IZR (fst (fst (fst r)))) /\
    generic_format radix2 (FLX_exp fp24_prec) (IZR (snd (fst (fst r)))) /\
    generic_format radix2 (FLX_exp fp24_prec) (IZR (snd (fst r))) /\
    generic_format radix2 (FLX_exp fp24_prec) (IZR (snd r)).
Proof.
  intros B m00 m01 m02 m03 m10 m11 m12 m13
         m20 m21 m22 m23 m30 m31 m32 m33 v0 v1 v2 v3
         Hadmit
         H00 H01 H02 H03 H10 H11 H12 H13
         H20 H21 H22 H23 H30 H31 H32 H33 Hv0 Hv1 Hv2 Hv3.
  simpl.
  repeat split; apply (dp4_operand_admit_exact B); assumption.
Qed.

(* The tight boundary, fixed by computation: B = 181 is admitted and B = 182
   is refused. *)
Lemma dp4_admit_boundary :
  dp4_operand_admit 181 = true /\ dp4_operand_admit 182 = false.
Proof. split; vm_compute; reflexivity. Qed.
