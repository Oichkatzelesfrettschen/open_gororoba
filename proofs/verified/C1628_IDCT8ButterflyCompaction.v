(** * C-1628: The 8-point even-odd butterfly IDCT equals the dense IDCT.

    The r300 g3dvl inverse DCT runs an 8-point IDCT per row and per column. The
    32-multiply even-odd butterfly factorization (Loeffler/Chen-class: split the
    8 outputs into an even half E0..E3 and an odd half O0..O3, recombine as
    x_k = E_k + O_k and x_{7-k} = E_k - O_k) produces outputs identical to the
    64-multiply dense matrix-vector form. This is the algebraic justification for
    compacting the two-pass IDCT onto the r300 fragment DP4 ALU (steinmarder
    single-pass IDCT design): halving the multiply count changes nothing the
    decoder can observe. *)

From OpenGororoba Require Import IDCT8EvenOdd.

(** The eight butterfly outputs equal the eight dense outputs identically.
    Proved in IDCT8EvenOdd by ring normalization over the reals. *)
Theorem C1628_idct8_butterfly_eq_dense :
  x0_bf = x0_dense /\ x1_bf = x1_dense /\ x2_bf = x2_dense /\ x3_bf = x3_dense /\
  x4_bf = x4_dense /\ x5_bf = x5_dense /\ x6_bf = x6_dense /\ x7_bf = x7_dense.
Proof. exact idct8_butterfly_eq_dense. Qed.
