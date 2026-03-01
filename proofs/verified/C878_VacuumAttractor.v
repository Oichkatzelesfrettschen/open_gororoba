(** * C-878: Vacuum attractor (F=3/8) gives zero correction.

    Formal proof that when F = F_vac = 3/8 (the vacuum frustration
    attractor), the algebraic York time correction vanishes for all
    coupling strengths and all York times.

    Additionally proves that the sedenion stress-energy simplifies
    to (1/2)*nu_0*K^2 when beta_coupling = 0 (at vacuum).

    Mirrors: adm_algebra_bridge.rs tests at lines 270-283. *)

From OpenGororoba Require Import Prelude ADMAlgebraBridge.

(** CLAIM C-878a: Vacuum attractor zeroes the York time correction. *)
Theorem C878_vacuum_zeroes_correction :
  forall theta alpha_s : R,
    algebraic_york_time_correction F_vac theta alpha_s = 0.
Proof.
  intros theta alpha_s.
  unfold algebraic_york_time_correction, F_vac.
  ring.
Qed.

(** C-878b: At vacuum with zero beta, stress-energy = (1/2)*nu_0*K^2.
    This mirrors the Rust test at line 280-283:
    sedenion_stress_energy(VACUUM_ATTRACTOR, 1.0, 1.0, 0.0) = 0.5 *)
Theorem C878_stress_energy_at_vacuum_zero_beta :
  forall k_sq nu_0 : R,
    sedenion_stress_energy F_vac k_sq nu_0 0 = (1/2) * nu_0 * k_sq.
Proof.
  intros k_sq nu_0.
  unfold sedenion_stress_energy, F_vac.
  (* beta_coupling * (3/8 - 3/8) = 0, exp(0) = 1 *)
  replace (0 * (3 / 8 - 3 / 8)) with 0 by ring.
  rewrite exp_0.
  ring.
Qed.

(** Characterization: the correction vanishes iff F = F_vac or alpha_s * theta = 0. *)
Theorem C878_correction_vanishes_iff :
  forall F theta alpha_s : R,
    algebraic_york_time_correction F theta alpha_s = 0 <->
    (F = F_vac \/ alpha_s = 0 \/ theta = 0).
Proof.
  intros F theta alpha_s.
  unfold algebraic_york_time_correction, F_vac.
  split.
  - intro H.
    (* If product of three factors is 0, at least one is 0. *)
    apply Rmult_integral in H. destruct H as [H | H].
    + apply Rmult_integral in H. destruct H as [H | H].
      * right; left; exact H.
      * left. lra.
    + right; right; exact H.
  - intros [H | [H | H]].
    + subst F. ring.
    + subst alpha_s. ring.
    + subst theta. ring.
Qed.
