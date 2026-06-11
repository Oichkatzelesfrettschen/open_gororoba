(** * Octonion sandwich-norm theorem.

    The octonion sandwich x * v * conj(x) scales the norm of v by |x|^2:

        |x * v * conj(x)|^2 = |x|^4 * |v|^2.

    Assembled from Hurwitz multiplicativity (oct_norm_mul, applied twice) and
    conjugate-norm invariance (brown1972_octonion_norm_conj_preserved).  This
    grounds the OTRANS octonion-sandwich virtual op on the RS480 compute-as-raster
    substrate (the octonion analogue of the quaternion rotation sandwich, whose
    norm-preservation is C876_rotation_preserves_norm).  For a unit octonion
    (|x|^2 = 1) the sandwich preserves the norm of v exactly. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
     OctonionNorm Brown1972ChapterIII.

Open Scope R_scope.

(** The squared norm of the sandwich is |x|^4 |v|^2.  Each octonion product
    contributes a factor of |x|^2 (the outer conj(x), via conj-norm invariance)
    or splits into |x|^2|v|^2 (the inner x*v), by the Hurwitz law. *)
Theorem oct_sandwich_norm : forall x v : CDOct,
  oct_norm_sq (oct_mul (oct_mul x v) (oct_conj x)) =
  oct_norm_sq x * oct_norm_sq x * oct_norm_sq v.
Proof.
  intros x v.
  rewrite !oct_norm_mul.
  rewrite brown1972_octonion_norm_conj_preserved.
  ring.
Qed.

(** Unit-octonion corollary: |x|^2 = 1 makes the sandwich norm-preserving, the
    octonion analogue of a rotation. *)
Corollary oct_sandwich_norm_unit : forall x v : CDOct,
  oct_norm_sq x = 1 ->
  oct_norm_sq (oct_mul (oct_mul x v) (oct_conj x)) = oct_norm_sq v.
Proof.
  intros x v Hunit.
  rewrite oct_sandwich_norm, Hunit.
  ring.
Qed.
