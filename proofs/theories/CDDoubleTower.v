(** * CDDoubleTower: Concrete instantiation of the CDDouble functor.

    Provides a RealBase module (the reals as a 1D CD algebra) and
    chains CDDouble to build:
      Complex  = CDDouble RealBase
      Quat     = CDDouble Complex
      Oct      = CDDouble Quat
      Sed      = CDDouble Oct
      Pathion  = CDDouble Sed

    Each level inherits all 7 linearity theorems with zero new proofs.
    This demonstrates that the CDDoubleFunctor is a working generic
    doubling machine for the full CD tower.

    Claim: C-1474 (fuel adequacy) + CDDoubleFunctor compose to give
    a verified CD tower from dim=1 to dim=32 with no per-level proofs. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import CDDoubleFunctor.
Open Scope R_scope.

(** * RealBase: the reals as a trivial 1D CD algebra. *)

Module RealBase <: CDAlgLinear.
  Definition carrier := R.

  Definition cd_mul (a b : R) : R := a * b.
  Definition cd_add (a b : R) : R := a + b.
  Definition cd_neg (a : R) : R := - a.
  Definition cd_sub (a b : R) : R := a - b.
  Definition cd_conj (a : R) : R := a.  (* reals are self-conjugate *)
  Definition cd_scale (r : R) (a : R) : R := r * a.
  Definition cd_zero : R := 0.

  Theorem mul_scale_left : forall r (a b : R),
    cd_mul (cd_scale r a) b = cd_scale r (cd_mul a b).
  Proof. intros; unfold cd_mul, cd_scale; ring. Qed.

  Theorem mul_scale_right : forall r (a b : R),
    cd_mul a (cd_scale r b) = cd_scale r (cd_mul a b).
  Proof. intros; unfold cd_mul, cd_scale; ring. Qed.

  Theorem conj_scale : forall r (a : R),
    cd_conj (cd_scale r a) = cd_scale r (cd_conj a).
  Proof. intros; unfold cd_conj, cd_scale; reflexivity. Qed.

  Theorem neg_scale : forall r (a : R),
    cd_neg (cd_scale r a) = cd_scale r (cd_neg a).
  Proof. intros; unfold cd_neg, cd_scale; ring. Qed.

  Theorem scale_add : forall r (a b : R),
    cd_scale r (cd_add a b) = cd_add (cd_scale r a) (cd_scale r b).
  Proof. intros; unfold cd_scale, cd_add; ring. Qed.

  Theorem scale_sub : forall r (a b : R),
    cd_scale r (cd_sub a b) = cd_sub (cd_scale r a) (cd_scale r b).
  Proof. intros; unfold cd_scale, cd_sub; ring. Qed.

  Theorem sub_def : forall (a b : R),
    cd_sub a b = cd_add a (cd_neg b).
  Proof. intros; unfold cd_sub, cd_add, cd_neg; ring. Qed.
End RealBase.

(** * Tower instantiation: 5 levels from reals to pathions. *)

Module CDComplex    := CDDouble RealBase.
Module CDQuat       := CDDouble CDComplex.
Module CDOct        := CDDouble CDQuat.
Module CDSed        := CDDouble CDOct.
Module CDPathion    := CDDouble CDSed.

(** * Tower verification: linearity at sedenion level (dim=16).

    This theorem states that scaling a sedenion factor and then
    multiplying gives the same result as multiplying then scaling.
    It was proven AUTOMATICALLY by the functor chain -- no manual
    proof at the sedenion or any intermediate level. *)

Theorem sed_mul_scale_left : forall r (x y : CDSed.carrier),
  CDSed.cd_mul (CDSed.cd_scale r x) y = CDSed.cd_scale r (CDSed.cd_mul x y).
Proof.
  exact CDSed.mul_scale_left.
Qed.

(** Same at pathion level (dim=32). *)
Theorem pathion_mul_scale_left : forall r (x y : CDPathion.carrier),
  CDPathion.cd_mul (CDPathion.cd_scale r x) y =
  CDPathion.cd_scale r (CDPathion.cd_mul x y).
Proof.
  exact CDPathion.mul_scale_left.
Qed.

(** * Type-level dimension witness.

    We can verify the nesting depth at the type level.
    CDSed.carrier = (((R, R), (R, R)), ((R, R), (R, R))),
                   (((R, R), (R, R)), ((R, R), (R, R)))
    which has 2^4 = 16 R fields, confirming dim=16.

    CDPathion.carrier nests one more level: 2^5 = 32 R fields. *)

Definition sed_zero : CDSed.carrier := CDSed.cd_zero.
Definition pathion_zero : CDPathion.carrier := CDPathion.cd_zero.

(** * Full tower linearity summary:

    Level    | Dim | Module     | Linearity Theorems
    ---------|-----|------------|-------------------
    Real     | 1   | RealBase   | 7 (proven by ring)
    Complex  | 2   | CDComplex  | 7 (auto from functor)
    Quat     | 4   | CDQuat     | 7 (auto from functor)
    Oct      | 8   | CDOct      | 7 (auto from functor)
    Sed      | 16  | CDSed      | 7 (auto from functor)
    Pathion  | 32  | CDPathion  | 7 (auto from functor)
    ---------|-----|------------|-------------------
    Total    |     |            | 42 theorems, 7 manual proofs *)
