(** * Heat kernel and spectral dimension on K_{2,2,...,2} (orthoplex graph).

    Defines the exact analytical heat kernel return probability and spectral
    dimension for the complete multipartite graph K_{2,2,...,2} with k parts
    of size 2, giving n = 2k vertices.

    Mirrors: orthoplex_diffusion.rs lines 124 and 143.

    Key results:
    - heat_kernel_k22_at_0: P(0) = 1
    - heat_kernel_k22_positive: P(t) > 0 for t >= 0
    - spectral_dim_nonneg: d_s(t) >= 0 for t > 0

    The Laplacian eigenvalues of K_{2,2,...,2} are:
    - mu_0 = 0         (multiplicity 1)
    - mu_1 = 2(k-1)    (multiplicity k)
    - mu_2 = 2k         (multiplicity k-1)
    Total: 1 + k + (k-1) = 2k vertices. *)

From Stdlib Require Import Rtrigo_def.
From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** ---------- Heat kernel return probability ---------- *)

(** P(t) = (1/(2k)) * [1 + k*exp(-2(k-1)*t) + (k-1)*exp(-2k*t)]

    We define this for real-valued k > 0 and t >= 0. *)
Definition heat_kernel_k22 (t k : R) : R :=
  (1 + k * exp (- (2 * (k - 1) * t)) + (k - 1) * exp (- (2 * k * t))) / (2 * k).

(** At t=0, P(0) = 1: walker is at starting vertex with certainty. *)
Lemma heat_kernel_k22_at_0 : forall k : R, k > 0 ->
  heat_kernel_k22 0 k = 1.
Proof.
  intros k Hk. unfold heat_kernel_k22.
  repeat rewrite Rmult_0_r.
  repeat rewrite Ropp_0.
  repeat rewrite exp_0.
  field. lra.
Qed.

(** P(t) > 0 for all t >= 0 and k >= 1. *)
Lemma heat_kernel_k22_positive : forall t k : R,
  t >= 0 -> k >= 1 ->
  heat_kernel_k22 t k > 0.
Proof.
  intros t k Ht Hk. unfold heat_kernel_k22.
  assert (He1 : exp (- (2 * (k - 1) * t)) > 0) by apply exp_pos.
  assert (He2 : exp (- (2 * k * t)) > 0) by apply exp_pos.
  apply Rdiv_lt_0_compat.
  - nra.
  - lra.
Qed.

(** ---------- Spectral dimension ---------- *)

(** d_s(t) = -2t * P'(t) / P(t)

    Numerator: 4*k*t*(k-1) * [exp(-2(k-1)*t) + exp(-2k*t)]
    Denominator: 1 + k*exp(-2(k-1)*t) + (k-1)*exp(-2k*t)

    We define only the numerator factor (unnormalized) for the
    non-negativity proof. The full spectral dimension is the ratio. *)
Definition spectral_dim_numerator (t k : R) : R :=
  4 * k * t * (k - 1) * (exp (- (2 * (k - 1) * t)) + exp (- (2 * k * t))).

Definition spectral_dim_denominator (t k : R) : R :=
  1 + k * exp (- (2 * (k - 1) * t)) + (k - 1) * exp (- (2 * k * t)).

(** The numerator is non-negative for t > 0, k >= 2. *)
Lemma spectral_dim_numerator_nonneg : forall t k : R,
  t > 0 -> k >= 2 ->
  spectral_dim_numerator t k >= 0.
Proof.
  intros t k Ht Hk. unfold spectral_dim_numerator.
  assert (He1 : exp (- (2 * (k - 1) * t)) > 0) by apply exp_pos.
  assert (He2 : exp (- (2 * k * t)) > 0) by apply exp_pos.
  assert (Hsum : exp (- (2 * (k - 1) * t)) + exp (- (2 * k * t)) > 0) by lra.
  assert (Hkt : k * t > 0) by nra.
  assert (Hkm1 : k - 1 > 0) by lra.
  assert (H4kt : 4 * (k * t) > 0) by lra.
  assert (Hprod : 4 * (k * t) * (k - 1) > 0) by nra.
  nra.
Qed.

(** The denominator is strictly positive for t >= 0, k >= 1. *)
Lemma spectral_dim_denominator_pos : forall t k : R,
  t >= 0 -> k >= 1 ->
  spectral_dim_denominator t k > 0.
Proof.
  intros t k Ht Hk. unfold spectral_dim_denominator.
  assert (He1 : exp (- (2 * (k - 1) * t)) > 0) by apply exp_pos.
  assert (He2 : exp (- (2 * k * t)) > 0) by apply exp_pos.
  nra.
Qed.

(** The spectral dimension d_s(t) = numerator / denominator >= 0
    for t > 0, k >= 2. *)
Lemma spectral_dim_nonneg : forall t k : R,
  t > 0 -> k >= 2 ->
  spectral_dim_numerator t k / spectral_dim_denominator t k >= 0.
Proof.
  intros t k Ht Hk.
  assert (Hnum : spectral_dim_numerator t k >= 0)
    by (apply spectral_dim_numerator_nonneg; lra).
  assert (Hden : spectral_dim_denominator t k > 0)
    by (apply spectral_dim_denominator_pos; lra).
  unfold Rdiv.
  apply Rle_ge.
  apply Rmult_le_pos.
  - lra.
  - left. apply Rinv_pos. exact Hden.
Qed.
