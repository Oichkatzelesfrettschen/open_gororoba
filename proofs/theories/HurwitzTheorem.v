(** * HurwitzTheorem: Norm composition at every Cayley-Dickson level.

    Aggregates norm multiplicativity results across dims 1, 2, 4, 8, 16:
    - dim 1 (R):    trivial -- x*y norm = |x|*|y|
    - dim 2 (C):    complex_norm_mul from CayleyDicksonAlgebra
    - dim 4 (H):    quat_norm_mul from CayleyDicksonAlgebra
    - dim 8 (O):    oct_norm_mul from OctonionNorm
    - dim 16 (S):   FAILS -- sed_norm_fails from OctonionNorm

    This establishes the Hurwitz theorem: normed division algebras
    exist only at dims 1, 2, 4, 8.  Sedenions (dim 16) break it.

    Mirrors: cd_kernel/src/cayley_dickson.rs (norm tests) *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Dim 1: reals. Norm is just squaring; multiplicativity is trivial. *)
Theorem hurwitz_dim1 : forall x y : R, (x * y) ^ 2 = x ^ 2 * y ^ 2.
Proof. intros. ring. Qed.

(** Dim 2: complex. *)
Theorem hurwitz_dim2 : forall z w : CDComplex,
  complex_norm_sq (complex_mul z w) = complex_norm_sq z * complex_norm_sq w.
Proof. exact complex_norm_mul. Qed.

(** Dim 4: quaternion. *)
Theorem hurwitz_dim4 : forall p q : CDQuat,
  quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q.
Proof. exact quat_norm_mul. Qed.

(** Dim 8: octonion. *)
Theorem hurwitz_dim8 : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq x * oct_norm_sq y.
Proof. exact oct_norm_mul. Qed.

(** Dim 16: sedenion -- FAILS.
    There exist sedenions x, y with |x*y|^2 <> |x|^2 * |y|^2. *)
Theorem hurwitz_fails_dim16 :
  exists x y : CDSed,
    sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y.
Proof. exact sed_norm_fails. Qed.
