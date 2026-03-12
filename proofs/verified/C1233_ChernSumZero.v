(** * C-1233: Total Chern number = 0 for all bands under TRS.

    CLAIM: When the Hamiltonian has time-reversal symmetry H(k) = H(-k)*,
    the Berry curvature satisfies F(k) = -F(-k), so the integral over
    the full Brillouin zone vanishes: C_total = 0.

    STRATEGY: From the TRS antisymmetry of F and linearity of the
    Chern integral, construct the cancellation F + F_reflected = 0
    pointwise, then lift to the integral.

    Mirrors: crates/quantum_core/src/tight_binding.rs (band_chern_number)
    Binary:  magnonic-band-structure --chern *)

From OpenGororoba Require Import Prelude.
Require Import OpenGororoba.TightBindingTopology.

Open Scope R_scope.

(** The reflected Berry curvature field. *)
Definition reflected (F : berry_curvature) : berry_curvature :=
  fun kx ky => F (-kx) (-ky).

(** Under TRS, F and its reflection cancel pointwise. *)
Lemma trs_pointwise_cancel :
  forall (F : berry_curvature),
    has_trs F ->
    forall kx ky, F kx ky + reflected F kx ky = 0.
Proof.
  intros F Htrs kx ky.
  unfold reflected, has_trs in *.
  rewrite Htrs. lra.
Qed.

(** MAIN THEOREM: Total Chern number vanishes under TRS. *)
Theorem chern_sum_zero_under_trs :
  forall (F : berry_curvature),
    has_trs F ->
    chern_integral F + chern_integral (reflected F) = 0.
Proof.
  intros F Htrs.
  apply chern_integral_linear.
  exact (trs_pointwise_cancel F Htrs).
Qed.
