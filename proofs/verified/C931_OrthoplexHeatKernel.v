(** * C-931: Orthoplex heat kernel -- exact analytical on K_{2,2,...,2}.

    Verified claims:
    1. P(0) = 1: at t=0, the random walker is at the starting vertex.
    2. P(t) > 0: return probability is strictly positive for all t >= 0.
    3. d_s(t) >= 0: spectral dimension is non-negative for t > 0.

    These properties guarantee that the spectral dimension is well-defined
    and that the heat kernel has the correct probabilistic interpretation.

    Mirrors: orthoplex_diffusion.rs lines 124, 143 (heat_kernel_k22,
    spectral_dimension_k22). *)

From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import HeatKernel.

Open Scope R_scope.

(** C-931a: Heat kernel return probability equals 1 at t=0. *)
Theorem C931a_heat_kernel_at_zero : forall k : R, k > 0 ->
  heat_kernel_k22 0 k = 1.
Proof.
  exact heat_kernel_k22_at_0.
Qed.

(** C-931b: Heat kernel return probability is strictly positive. *)
Theorem C931b_heat_kernel_positive : forall t k : R,
  t >= 0 -> k >= 1 ->
  heat_kernel_k22 t k > 0.
Proof.
  exact heat_kernel_k22_positive.
Qed.

(** C-931c: Spectral dimension is non-negative. *)
Theorem C931c_spectral_dim_nonneg : forall t k : R,
  t > 0 -> k >= 2 ->
  spectral_dim_numerator t k / spectral_dim_denominator t k >= 0.
Proof.
  exact spectral_dim_nonneg.
Qed.
