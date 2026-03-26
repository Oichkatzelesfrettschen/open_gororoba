(** * C-031: Hurwitz theorem -- normed division algebras at dims 1,2,4,8 only.

    Norm multiplicativity |x*y|^2 = |x|^2 * |y|^2 holds at
    dims 1 (R), 2 (C), 4 (H), 8 (O) and fails at dim 16 (S).

    This covers all five Cayley-Dickson levels in the project. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororoba Require Import OctonionNorm HurwitzTheorem.
Open Scope R_scope.

(** Dim 1: trivial. *)
Theorem C031_dim1 : forall x y : R, (x * y) ^ 2 = x ^ 2 * y ^ 2.
Proof. exact hurwitz_dim1. Qed.

(** Dim 2: complex norm multiplicative. *)
Theorem C031_dim2 : forall z w : CDComplex,
  complex_norm_sq (complex_mul z w) = complex_norm_sq z * complex_norm_sq w.
Proof. exact hurwitz_dim2. Qed.

(** Dim 4: quaternion norm multiplicative. *)
Theorem C031_dim4 : forall p q : CDQuat,
  quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q.
Proof. exact hurwitz_dim4. Qed.

(** Dim 8: octonion norm multiplicative. *)
Theorem C031_dim8 : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq x * oct_norm_sq y.
Proof. exact hurwitz_dim8. Qed.

(** Dim 16: sedenion norm NOT multiplicative. *)
Theorem C031_dim16_fails :
  exists x y : CDSed,
    sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y.
Proof. exact hurwitz_fails_dim16. Qed.
