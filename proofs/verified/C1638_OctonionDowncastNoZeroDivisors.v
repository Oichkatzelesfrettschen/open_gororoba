(** * C-1638: Downcasted octonion lanes have no zero divisors.

    The 8D (octonion) downcast route keeps the Hurwitz division-algebra
    boundary.  A zero product in that route forces one operand to be zero, so
    a nonzero sedenion zero-divisor witness cannot be explained by the
    hi-half-zero classifier path. *)

From OpenGororoba Require Import
  Prelude
  CayleyDicksonAlgebra
  Sedenion
  OctonionNorm
  Brown1972ChapterIII.
From OpenGororobaVerified Require Import
  C028_SedenionAutGroup
  C1630_SedenionOctonionDowncast.

Open Scope R_scope.

Theorem C1638_octonion_no_zero_divisors : forall a b : CDOct,
  oct_mul a b = oct_zero ->
  a = oct_zero \/ b = oct_zero.
Proof.
  intros a b Hprod.
  assert (Hnorm : oct_norm_sq (oct_mul a b) = 0).
  { rewrite Hprod.
    cbv [oct_norm_sq oct_zero quat_norm_sq quat_zero oct_lo oct_hi
         qa qb qc qd].
    ring. }
  rewrite oct_norm_mul in Hnorm.
  destruct (Req_EM_T (oct_norm_sq a) 0) as [Ha_zero | Ha_nonzero].
  - left.
    exact (brown1972_oct_norm_zero a Ha_zero).
  - right.
    apply brown1972_oct_norm_zero.
    nra.
Qed.

Theorem C1638_downcast_no_zero_divisors : forall a b : CDOct,
  sed_mul (sed_octonion_downcast a) (sed_octonion_downcast b) = sed_zero ->
  a = oct_zero \/ b = oct_zero.
Proof.
  intros a b Hprod.
  apply C1638_octonion_no_zero_divisors.
  assert (Hlo := f_equal sed_lo Hprod).
  rewrite C1630_downcast_mul in Hlo.
  simpl in Hlo.
  exact Hlo.
Qed.

Theorem C1638_hi_zero_operands_no_zero_divisors : forall x y : CDSed,
  sed_hi x = oct_zero ->
  sed_hi y = oct_zero ->
  sed_mul x y = sed_zero ->
  x = sed_zero \/ y = sed_zero.
Proof.
  intros x y Hx_hi Hy_hi Hprod.
  pose proof (proj1 (C1630_downcast_shape_iff x) Hx_hi) as Hx_shape.
  pose proof (proj1 (C1630_downcast_shape_iff y) Hy_hi) as Hy_shape.
  rewrite Hx_shape, Hy_shape in Hprod.
  destruct (C1638_downcast_no_zero_divisors (sed_lo x) (sed_lo y) Hprod)
    as [Hx_lo | Hy_lo].
  - left.
    rewrite Hx_shape.
    unfold sed_octonion_downcast, sed_embed_lo.
    rewrite Hx_lo.
    reflexivity.
  - right.
    rewrite Hy_shape.
    unfold sed_octonion_downcast, sed_embed_lo.
    rewrite Hy_lo.
    reflexivity.
Qed.

Record DowncastNoZeroDivisorSurface := {
  dnzd_octonion_no_zero_divisors :
    forall a b : CDOct,
      oct_mul a b = oct_zero ->
      a = oct_zero \/ b = oct_zero;
  dnzd_downcast_no_zero_divisors :
    forall a b : CDOct,
      sed_mul (sed_octonion_downcast a) (sed_octonion_downcast b) = sed_zero ->
      a = oct_zero \/ b = oct_zero;
  dnzd_hi_zero_operands_no_zero_divisors :
    forall x y : CDSed,
      sed_hi x = oct_zero ->
      sed_hi y = oct_zero ->
      sed_mul x y = sed_zero ->
      x = sed_zero \/ y = sed_zero
}.

Definition C1638_downcast_no_zero_divisor_surface :
  DowncastNoZeroDivisorSurface.
Proof.
  refine
    {| dnzd_octonion_no_zero_divisors := C1638_octonion_no_zero_divisors;
       dnzd_downcast_no_zero_divisors := C1638_downcast_no_zero_divisors;
       dnzd_hi_zero_operands_no_zero_divisors :=
         C1638_hi_zero_operands_no_zero_divisors |}.
Defined.
