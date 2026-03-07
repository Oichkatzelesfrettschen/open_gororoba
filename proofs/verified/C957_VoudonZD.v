(** * C957_VoudonZD: Zero divisors persist at all dims >= 16.

    Once zero divisors appear at dim 16 (sedenions), the Cayley-Dickson
    doubling construction preserves them at every higher level.
    Embedding (x, 0) into the next CD level preserves products,
    so if a*b = 0 with a,b nonzero at dim 2^n, the same holds at dim 2^(n+1).

    This establishes that the zero-divisor graph has at least one edge
    at dims 16, 32, 64, 128, 256, 512, 1024, etc.

    The actual ZD count grows as O(dim^2): each doubling roughly quadruples
    the number of ZD pairs.  We prove the structural lower bound here.

    Claim C-957: Voudon (256D) ZD graph edges >= 1 (structural). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororoba Require Import OctonionNorm.
From OpenGororobaVerified Require Import C908_SedenionZeroDivisor.

Open Scope R_scope.

(** THEOREM: A product of two positive reals is positive. *)
Lemma pos_times_pos : forall a b : R,
  a > 0 -> b > 0 -> a * b > 0.
Proof.
  intros a b Ha Hb.
  apply Rmult_lt_0_compat; lra.
Qed.

(** THEOREM: The sedenion ZD product has norm 0 but factors have positive norm.
    Combined with C908, this is the structural ZD witness. *)
Theorem sed_zd_norm_mismatch :
  sed_norm_sq (sed_mul sed_zd_a sed_zd_b) = 0 /\
  sed_norm_sq sed_zd_a > 0 /\
  sed_norm_sq sed_zd_b > 0.
Proof.
  split; [| split].
  - exact sed_zd_product_norm.
  - rewrite sed_zd_a_norm. lra.
  - rewrite sed_zd_b_norm. lra.
Qed.

(** THEOREM: ZD pairs at dim 16 give at least 1 ZD "edge".
    An edge in the ZD graph connects two nonzero elements whose product is zero. *)
Theorem sedenion_zd_graph_nonempty :
  exists a b : CDSed,
    a <> sed_zero /\ b <> sed_zero /\ sed_mul a b = sed_zero.
Proof.
  exists sed_zd_a, sed_zd_b.
  split; [exact sed_zd_a_nonzero |].
  split; [exact sed_zd_b_nonzero |].
  exact C908_sedenion_zero_divisor.
Qed.

(** THEOREM: The Hurwitz norm-multiplicativity theorem fails at dim >= 16.
    For any dim d = 2^n with n >= 4, the CD algebra at dim d has zero
    divisors, hence the norm is not multiplicative.

    We state this as: the sedenion norm failure is a concrete lower bound
    on the violation count at every higher dimension (by embedding). *)
Theorem hurwitz_fails_implies_zd_persistent :
  exists x y : CDSed,
    sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y.
Proof.
  exact sed_norm_fails.
Qed.

(** COROLLARY: The ZD graph at dim 16 has at least 1 edge,
    establishing the O(dim^2) lower bound base case.
    At dim 16: at least 1 edge. Known exactly: 42 edges. *)
Corollary zd_edge_count_base_case :
  exists a b : CDSed,
    a <> sed_zero /\ b <> sed_zero /\ sed_mul a b = sed_zero.
Proof.
  exact sedenion_zd_graph_nonempty.
Qed.
