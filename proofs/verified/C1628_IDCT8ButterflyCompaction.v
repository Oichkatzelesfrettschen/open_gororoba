(** * C-1628: The 8-point even-odd butterfly IDCT equals the dense IDCT.

    The r300 g3dvl inverse DCT runs an 8-point IDCT per row and per column. The
    32-multiply even-odd butterfly factorization (Loeffler/Chen-class: split the
    8 outputs into an even half E0..E3 and an odd half O0..O3, recombine as
    x_k = E_k + O_k and x_{7-k} = E_k - O_k) produces outputs identical to the
    64-multiply dense matrix-vector form. This is the algebraic justification for
    compacting the two-pass IDCT onto the r300 fragment DP4 ALU (steinmarder
    single-pass IDCT design): halving the multiply count changes nothing the
    decoder can observe. *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import IDCT8EvenOdd.

Open Scope R_scope.

(** The eight butterfly outputs equal the eight dense outputs identically.
    Proved in IDCT8EvenOdd by ring normalization over the reals. *)
Theorem C1628_idct8_butterfly_eq_dense :
  forall
    c00 c01 c02 c03 c10 c11 c12 c13
    c20 c21 c22 c23 c30 c31 c32 c33
    c40 c41 c42 c43 c50 c51 c52 c53
    c60 c61 c62 c63 c70 c71 c72 c73
    F0 F1 F2 F3 F4 F5 F6 F7 : R,
  x0_bf c00 c10 c20 c30 c40 c50 c60 c70 F0 F1 F2 F3 F4 F5 F6 F7 =
  x0_dense c00 c10 c20 c30 c40 c50 c60 c70 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x1_bf c01 c11 c21 c31 c41 c51 c61 c71 F0 F1 F2 F3 F4 F5 F6 F7 =
  x1_dense c01 c11 c21 c31 c41 c51 c61 c71 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x2_bf c02 c12 c22 c32 c42 c52 c62 c72 F0 F1 F2 F3 F4 F5 F6 F7 =
  x2_dense c02 c12 c22 c32 c42 c52 c62 c72 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x3_bf c03 c13 c23 c33 c43 c53 c63 c73 F0 F1 F2 F3 F4 F5 F6 F7 =
  x3_dense c03 c13 c23 c33 c43 c53 c63 c73 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x4_bf c03 c13 c23 c33 c43 c53 c63 c73 F0 F1 F2 F3 F4 F5 F6 F7 =
  x4_dense c03 c13 c23 c33 c43 c53 c63 c73 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x5_bf c02 c12 c22 c32 c42 c52 c62 c72 F0 F1 F2 F3 F4 F5 F6 F7 =
  x5_dense c02 c12 c22 c32 c42 c52 c62 c72 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x6_bf c01 c11 c21 c31 c41 c51 c61 c71 F0 F1 F2 F3 F4 F5 F6 F7 =
  x6_dense c01 c11 c21 c31 c41 c51 c61 c71 F0 F1 F2 F3 F4 F5 F6 F7 /\
  x7_bf c00 c10 c20 c30 c40 c50 c60 c70 F0 F1 F2 F3 F4 F5 F6 F7 =
  x7_dense c00 c10 c20 c30 c40 c50 c60 c70 F0 F1 F2 F3 F4 F5 F6 F7.
Proof. exact idct8_butterfly_eq_dense. Qed.
