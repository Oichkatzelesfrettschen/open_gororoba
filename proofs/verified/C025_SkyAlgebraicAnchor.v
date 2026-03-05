(** * C-025: Sky alignment algebraic anchor.

    GWTC-3 sky position isotropy test (p=0.152) verified via Rust.
    Rocq scope = algebraic invariants.

    Algebraic anchor: all sedenion (octonion) basis elements have unit
    norm, so projection coordinates derived from basis elements carry
    no algebraic bias -- any anisotropy would be physical, not algebraic. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Every octonion basis element has unit norm. *)
Theorem C025_basis_unit_norm : forall i : nat,
  (i < 8)%nat -> oct_norm_sq (oct_e i) = 1.
Proof. exact oct_basis_unit_norm. Qed.

(** Norm is multiplicative (projection preserves algebraic structure). *)
Theorem C025_norm_multiplicative : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq x * oct_norm_sq y.
Proof. exact oct_norm_mul. Qed.
