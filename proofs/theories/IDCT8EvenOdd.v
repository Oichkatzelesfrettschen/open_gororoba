(* Even/odd butterfly factorization of the 8-point IDCT (DCT-III), proven equal
   to the dense matrix-vector definition.  This is the semantics-preserving core
   of an r300 g3dvl IDCT compaction: the dense form needs 64 multiplies, the
   butterfly form needs 32 (eight 4-term partial sums E_n/O_n shared between
   output n and its reflection 7-n), and the two are equal as a polynomial
   identity in the 32 basis constants -- no transcendental cosine identity is
   required, because the reflection C[k][7-n] = (-1)^k C[k][n] is baked into the
   dense right half (it is the defining cosine symmetry cos((2(7-n)+1)k pi/16) =
   (-1)^k cos((2n+1)k pi/16)).

   Tactic idiom matches the open_gororoba house style (destruct/unfold + ring,
   cf. CDFusedBilinear.quat_mul_fused_eq). Self-contained on Stdlib Reals. *)
From Stdlib Require Import Reals.
Open Scope R_scope.

Section IDCT8_EvenOdd.
  (* 32 DCT-III basis constants C[k][n] = a_k cos((2n+1) k pi/16) for the left
     spatial half n = 0..3, all eight frequencies k = 0..7.  Opaque reals. *)
  Variables c00 c01 c02 c03 : R.   (* k = 0 *)
  Variables c10 c11 c12 c13 : R.   (* k = 1 *)
  Variables c20 c21 c22 c23 : R.   (* k = 2 *)
  Variables c30 c31 c32 c33 : R.   (* k = 3 *)
  Variables c40 c41 c42 c43 : R.   (* k = 4 *)
  Variables c50 c51 c52 c53 : R.   (* k = 5 *)
  Variables c60 c61 c62 c63 : R.   (* k = 6 *)
  Variables c70 c71 c72 c73 : R.   (* k = 7 *)

  (* Input frequency coefficients. *)
  Variables F0 F1 F2 F3 F4 F5 F6 F7 : R.

  (* Even-frequency partial sums (k = 0,2,4,6): 4 multiplies each. *)
  Definition E0 := F0 * c00 + F2 * c20 + F4 * c40 + F6 * c60.
  Definition E1 := F0 * c01 + F2 * c21 + F4 * c41 + F6 * c61.
  Definition E2 := F0 * c02 + F2 * c22 + F4 * c42 + F6 * c62.
  Definition E3 := F0 * c03 + F2 * c23 + F4 * c43 + F6 * c63.

  (* Odd-frequency partial sums (k = 1,3,5,7): 4 multiplies each. *)
  Definition O0 := F1 * c10 + F3 * c30 + F5 * c50 + F7 * c70.
  Definition O1 := F1 * c11 + F3 * c31 + F5 * c51 + F7 * c71.
  Definition O2 := F1 * c12 + F3 * c32 + F5 * c52 + F7 * c72.
  Definition O3 := F1 * c13 + F3 * c33 + F5 * c53 + F7 * c73.

  (* Butterfly outputs: 8 add/sub on the shared partial sums (32 mults total). *)
  Definition x0_bf := E0 + O0.
  Definition x1_bf := E1 + O1.
  Definition x2_bf := E2 + O2.
  Definition x3_bf := E3 + O3.
  Definition x4_bf := E3 - O3.
  Definition x5_bf := E2 - O2.
  Definition x6_bf := E1 - O1.
  Definition x7_bf := E0 - O0.

  (* Dense DCT-III: x[n] = sum_k F_k C[k][n] (64 mults).  Right half n = 4..7
     uses the reflection C[k][7-m] = (-1)^k C[k][m]. *)
  Definition x0_dense :=
    F0*c00 + F1*c10 + F2*c20 + F3*c30 + F4*c40 + F5*c50 + F6*c60 + F7*c70.
  Definition x1_dense :=
    F0*c01 + F1*c11 + F2*c21 + F3*c31 + F4*c41 + F5*c51 + F6*c61 + F7*c71.
  Definition x2_dense :=
    F0*c02 + F1*c12 + F2*c22 + F3*c32 + F4*c42 + F5*c52 + F6*c62 + F7*c72.
  Definition x3_dense :=
    F0*c03 + F1*c13 + F2*c23 + F3*c33 + F4*c43 + F5*c53 + F6*c63 + F7*c73.
  Definition x4_dense :=
    F0*c03 - F1*c13 + F2*c23 - F3*c33 + F4*c43 - F5*c53 + F6*c63 - F7*c73.
  Definition x5_dense :=
    F0*c02 - F1*c12 + F2*c22 - F3*c32 + F4*c42 - F5*c52 + F6*c62 - F7*c72.
  Definition x6_dense :=
    F0*c01 - F1*c11 + F2*c21 - F3*c31 + F4*c41 - F5*c51 + F6*c61 - F7*c71.
  Definition x7_dense :=
    F0*c00 - F1*c10 + F2*c20 - F3*c30 + F4*c40 - F5*c50 + F6*c60 - F7*c70.

  (* The compaction is exact: the 32-multiply butterfly equals the 64-multiply
     dense transform at every output, as a ring identity in the basis constants. *)
  Theorem idct8_butterfly_eq_dense :
    x0_bf = x0_dense /\ x1_bf = x1_dense /\ x2_bf = x2_dense /\ x3_bf = x3_dense /\
    x4_bf = x4_dense /\ x5_bf = x5_dense /\ x6_bf = x6_dense /\ x7_bf = x7_dense.
  Proof.
    unfold x0_bf, x1_bf, x2_bf, x3_bf, x4_bf, x5_bf, x6_bf, x7_bf,
           x0_dense, x1_dense, x2_dense, x3_dense,
           x4_dense, x5_dense, x6_dense, x7_dense,
           E0, E1, E2, E3, O0, O1, O2, O3.
    repeat split; ring.
  Qed.
End IDCT8_EvenOdd.
