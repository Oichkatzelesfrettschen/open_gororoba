From Stdlib Require Import Reals.
From OpenGororoba Require Import
  Prelude
  CayleyDicksonAlgebra
  Sedenion
  OctonionNorm
  CDLinearLemmas.
Open Scope R_scope.

(** * CDFusedBilinear: flattened multiplication specs for Cayley-Dickson levels.

    This file does not try to certify machine-level scheduling wins. Instead it
    provides compact bilinear operator forms that proofs can rewrite to, so
    later Brown lanes can avoid repeatedly unfolding the recursive tower
    presentation. *)

Definition quat_mul_fused (p q : CDQuat) : CDQuat :=
  match p, q with
  | mkQuat a b c d, mkQuat e f g h =>
      mkQuat
        (a * e - b * f - c * g - d * h)
        (a * f + b * e + c * h - d * g)
        (a * g - b * h + c * e + d * f)
        (a * h + b * g - c * f + d * e)
  end.

Definition oct_mul_fused (x y : CDOct) : CDOct :=
  match x, y with
  | mkOct a b, mkOct c d =>
      mkOct
        (quat_add (quat_mul a c)
                  (quat_neg (quat_mul (quat_conj d) b)))
        (quat_add (quat_mul d a)
                  (quat_mul b (quat_conj c)))
  end.

Definition sed_mul_fused (x y : CDSed) : CDSed :=
  match x, y with
  | mkSed a b, mkSed c d =>
      mkSed
        (oct_add (oct_mul a c)
                 (oct_neg (oct_mul (oct_conj d) b)))
        (oct_add (oct_mul d a)
                 (oct_mul b (oct_conj c)))
  end.

Theorem quat_mul_fused_eq : forall p q : CDQuat,
  quat_mul_fused p q = quat_mul p q.
Proof.
  intros [a b c d] [e f g h].
  unfold quat_mul_fused, quat_mul.
  simpl.
  apply (f_equal4 mkQuat); ring.
Qed.

Theorem oct_mul_fused_eq : forall x y : CDOct,
  oct_mul_fused x y = oct_mul x y.
Proof.
  intros [a b] [c d].
  unfold oct_mul_fused, oct_mul.
  simpl.
  reflexivity.
Qed.

Theorem sed_mul_fused_eq : forall x y : CDSed,
  sed_mul_fused x y = sed_mul x y.
Proof.
  intros [a b] [c d].
  unfold sed_mul_fused, sed_mul.
  simpl.
  unfold oct_add, oct_neg.
  reflexivity.
Qed.

Record CDFusedBilinearSurface
    (A : Type)
    (add mul fused_mul : A -> A -> A)
    (scale : R -> A -> A) := {
  cd_fused_mul_eq :
    forall x y : A, fused_mul x y = mul x y;
  cd_fused_mul_add_left :
    forall x x' y : A,
      fused_mul (add x x') y =
      add (fused_mul x y) (fused_mul x' y);
  cd_fused_mul_add_right :
    forall x y y' : A,
      fused_mul x (add y y') =
      add (fused_mul x y) (fused_mul x y');
  cd_fused_mul_scale_left :
    forall (r : R) (x y : A),
      fused_mul (scale r x) y = scale r (fused_mul x y);
  cd_fused_mul_scale_right :
    forall (r : R) (x y : A),
      fused_mul x (scale r y) = scale r (fused_mul x y)
}.

Definition quat_fused_bilinear_surface :
  CDFusedBilinearSurface CDQuat quat_add quat_mul quat_mul_fused quat_scale.
Proof.
  refine
    {| cd_fused_mul_eq := quat_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_add_left.
  - intros x y y'.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_add_right.
  - intros r x y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_scale_left.
  - intros r x y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_scale_right.
Defined.

Definition oct_fused_bilinear_surface :
  CDFusedBilinearSurface CDOct oct_add oct_mul oct_mul_fused oct_scale.
Proof.
  refine
    {| cd_fused_mul_eq := oct_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_add_left.
  - intros x y y'.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_add_right.
  - intros r x y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_scale_left.
  - intros r x y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_scale_right.
Defined.

Definition sed_fused_bilinear_surface :
  CDFusedBilinearSurface CDSed sed_add sed_mul sed_mul_fused sed_scale.
Proof.
  refine
    {| cd_fused_mul_eq := sed_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_add_left.
  - intros x y y'.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_add_right.
  - intros r x y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_scale_left.
  - intros r x y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_scale_right.
Defined.
