(** * C-023: Quaternion associator is zero (no holonomy from associator).

    C-023 speculated about holonomy computed from quaternionic paths.
    The positive content: the quaternion associator vanishes identically
    (CDAssociator.quat_assoc_zero), so any "holonomy" from quaternionic
    associators is trivially zero.

    The first nontrivial associator appears at dim 8 (C-909). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra CDAssociator.
From OpenGororoba Require Import Sedenion OctonionNorm.

(** Dim 4: associator vanishes. No holonomy from quaternionic paths. *)
Theorem C023_quat_assoc_trivial : forall a b c : CDQuat,
  quat_assoc a b c = quat_zero.
Proof. exact quat_assoc_zero. Qed.

(** Dim 8: associator is nonzero. Holonomy-like effects first appear here. *)
Theorem C023_oct_assoc_nontrivial :
  oct_assoc (oct_e 1) (oct_e 2) (oct_e 4) <> oct_zero.
Proof. exact oct_assoc_nonzero. Qed.
