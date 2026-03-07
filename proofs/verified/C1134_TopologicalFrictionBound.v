(** * C-1134: Topological Friction Bound (Positive Friction at dim >= 16).

    CLAIM: The CD braid introduces positive topological friction for
    dim >= 16. Specifically, there exist probes X such that the
    associator [A_rotated, X, B] != 0.

    STRATEGY: Constructive witness at dim=16. We show that the
    sedenion associator |[e_1, e_9, e_2]|^2 = 4, proving that
    braiding along ZD channel (e_1, e_9) with probe e_2 incurs
    non-zero topological friction.

    This is a LOWER BOUND: friction >= 4 for the witness triple.

    Mirrors: crates/algebra_experimental/src/majorana_braiding.rs
             (cd_braid, measure_topological_friction) *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import SedenionAssociator.

Open Scope R_scope.

(** The associator [e1, e9, e2] has norm squared = 4.
    This is the topological friction incurred by a single braid. *)
Theorem sed_assoc_e1_e9_e2_norm :
  sed_assoc_norm_sq (sed_e 1) (sed_e 9) (sed_e 2) = 4.
Proof.
  unfold sed_assoc_norm_sq, sed_assoc, sed_sub, sed_add, sed_neg.
  cbv [sed_norm_sq oct_norm_sq sed_mul sed_e
       oct_add oct_neg oct_mul oct_conj oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       quat_norm_sq sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify. lra.
Qed.

(** Corollary: the topological friction is strictly positive. *)
Corollary topological_friction_positive :
  sed_assoc_norm_sq (sed_e 1) (sed_e 9) (sed_e 2) > 0.
Proof. rewrite sed_assoc_e1_e9_e2_norm. lra. Qed.

(** Additional witness: [e1, e2, e4] at dim=16 also has nonzero friction. *)
Theorem sed_assoc_e1_e2_e4_norm :
  sed_assoc_norm_sq (sed_e 1) (sed_e 2) (sed_e 4) = 4.
Proof.
  unfold sed_assoc_norm_sq, sed_assoc, sed_sub, sed_add, sed_neg.
  cbv [sed_norm_sq oct_norm_sq sed_mul sed_e
       oct_add oct_neg oct_mul oct_conj oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       quat_norm_sq sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify. lra.
Qed.

(** Friction from ZD-crossing triple is >= 4. *)
Corollary friction_lower_bound :
  sed_assoc_norm_sq (sed_e 1) (sed_e 9) (sed_e 2) >= 4.
Proof. rewrite sed_assoc_e1_e9_e2_norm. lra. Qed.
