(** * C-001: Cayley-Dickson algebras first lose associativity at dim 8.

    Quaternions (dim 4) are associative: quat_assoc = 0 for all triples.
    Octonions (dim 8) are NOT associative: (e1*e2)*e4 <> e1*(e2*e4).

    This is the foundational algebraic fact motivating the CD hierarchy:
    associativity is lost precisely at the octonion level, where
    alternativity still holds (see C-910). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororoba Require Import CDAssociator OctonionNorm.
From OpenGororobaVerified Require Import C909_OctonionNonAssociative.

(** Dim 4 is associative: the quaternion associator vanishes identically. *)
Theorem C001_dim4_associative : forall a b c : CDQuat,
  quat_assoc a b c = quat_zero.
Proof. exact quat_assoc_zero. Qed.

(** Dim 8 is NOT associative: the octonion associator is nonzero. *)
Theorem C001_dim8_non_associative :
  oct_assoc (oct_e 1) (oct_e 2) (oct_e 4) <> oct_zero.
Proof. exact oct_assoc_nonzero. Qed.
