(** * C-920: Quaternion multiplication is associative (property tower: dim=4 OK).

    H preserves associativity (the next CD level, O, loses it). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C920_quat_is_associative : forall x y z : CDQuat,
  quat_mul (quat_mul x y) z = quat_mul x (quat_mul y z).
Proof. exact quat_mul_assoc. Qed.
