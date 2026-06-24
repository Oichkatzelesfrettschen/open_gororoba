(** * C-1630: Sedenion octonion downcast boundary.

    A 16D sedenion value with zero hi-half carries exactly 8D octonion
    semantics.  Multiplying two such values cannot cross the CD-4 frontier:
    the product is the lo-half embedding of the octonion product.

    This is the Rocq boundary a driver classifier needs before routing a
    dim-16 input shape to an 8D octonion implementation.  The generic CD-4
    product remains distinct because nonzero hi-halves can expose zero
    divisors and norm failure. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororobaVerified Require Import C028_SedenionAutGroup.

(** Keep the driver-facing name local to this theorem file.  The definition is
    intentionally the same embedding as C028_SedenionAutGroup::sed_embed_lo. *)
Definition sed_octonion_downcast (x : CDOct) : CDSed := sed_embed_lo x.

Theorem C1630_downcast_shape_iff : forall x : CDSed,
  sed_hi x = oct_zero <-> x = sed_octonion_downcast (sed_lo x).
Proof.
  intros [lo hi].
  split; intro H.
  - simpl in H. subst. reflexivity.
  - unfold sed_octonion_downcast, sed_embed_lo in H.
    inversion H. reflexivity.
Qed.

Theorem C1630_downcast_mul : forall a b : CDOct,
  sed_mul (sed_octonion_downcast a) (sed_octonion_downcast b) =
  sed_octonion_downcast (oct_mul a b).
Proof.
  exact C028_lo_subalgebra_closed.
Qed.

Theorem C1630_downcast_mul_lo : forall a b : CDOct,
  sed_lo (sed_mul (sed_octonion_downcast a) (sed_octonion_downcast b)) =
  oct_mul a b.
Proof.
  intros a b.
  rewrite C1630_downcast_mul.
  reflexivity.
Qed.

Theorem C1630_downcast_mul_hi_zero : forall a b : CDOct,
  sed_hi (sed_mul (sed_octonion_downcast a) (sed_octonion_downcast b)) =
  oct_zero.
Proof.
  intros a b.
  rewrite C1630_downcast_mul.
  reflexivity.
Qed.

Theorem C1630_hi_zero_operands_downcast_mul : forall x y : CDSed,
  sed_hi x = oct_zero ->
  sed_hi y = oct_zero ->
  sed_mul x y = sed_octonion_downcast (oct_mul (sed_lo x) (sed_lo y)).
Proof.
  intros [xlo xhi] [ylo yhi] Hx Hy.
  simpl in Hx, Hy.
  subst.
  apply C1630_downcast_mul.
Qed.

Theorem C1630_hi_zero_operands_product_hi_zero : forall x y : CDSed,
  sed_hi x = oct_zero ->
  sed_hi y = oct_zero ->
  sed_hi (sed_mul x y) = oct_zero.
Proof.
  intros x y Hx Hy.
  rewrite (C1630_hi_zero_operands_downcast_mul x y Hx Hy).
  reflexivity.
Qed.

Theorem C1630_hi_zero_operands_product_lo : forall x y : CDSed,
  sed_hi x = oct_zero ->
  sed_hi y = oct_zero ->
  sed_lo (sed_mul x y) = oct_mul (sed_lo x) (sed_lo y).
Proof.
  intros x y Hx Hy.
  rewrite (C1630_hi_zero_operands_downcast_mul x y Hx Hy).
  reflexivity.
Qed.
