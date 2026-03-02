(** * C-904: Quaternion associator vanishes.

    [a,b,c] = (a*b)*c - a*(b*c) = 0 for all a, b, c in H.
    Quaternions form an associative (but non-commutative) division algebra. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra CDAssociator.

Theorem C904_quat_associator_zero : forall a b c,
  quat_assoc a b c = quat_zero.
Proof. exact quat_assoc_zero. Qed.
