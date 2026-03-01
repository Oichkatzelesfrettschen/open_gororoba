(** * Prelude: Shared imports and constants for open_gororoba proofs.

    Mirrors constants from Rust crate definitions:
    - IMBALANCE_ATTRACTOR = 3/8  (adm_algebra_bridge.rs:59)
    - PI from Stdlib.Reals     (std::f64::consts::PI) *)

From Stdlib Require Export Reals Lra Psatz.
Open Scope R_scope.

(** Imbalance attractor: F_vac = 3/8.
    Sedenion imbalance densities converge to this value in the absence
    of external sources. At this value, algebraic corrections vanish. *)
Definition F_vac : R := 3 / 8.

(** 16 * PI, used in Einstein field equations. *)
Definition sixteen_pi : R := 16 * PI.

(** Helper: 3/8 is strictly between 0 and 1/2. *)
Lemma F_vac_in_range : 0 < F_vac < 1/2.
Proof.
  unfold F_vac. split; lra.
Qed.

(** F_vac equals 0.375 exactly (rational). *)
Lemma F_vac_rational : F_vac = 3 * / 8.
Proof.
  unfold F_vac. field.
Qed.
