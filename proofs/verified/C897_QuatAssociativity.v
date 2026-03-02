(** * C-897: Quaternion multiplication is associative.

    (x * y) * z = x * (y * z) for all x, y, z in H. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C897_quat_associativity : forall x y z,
  quat_mul (quat_mul x y) z = quat_mul x (quat_mul y z).
Proof. exact quat_mul_assoc. Qed.
