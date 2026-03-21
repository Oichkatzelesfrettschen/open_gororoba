(** * CDDoubleFunctor: Generic Cayley-Dickson doubling in Rocq.

    Defines a Module Type for CD algebras with linearity axioms,
    then a Module Functor that doubles any such algebra while
    automatically inheriting all linearity properties.

    This eliminates the per-level boilerplate in HigherCD.v and
    CDLinearLemmas.v.  Each doubling requires ZERO new proofs --
    the functor proves everything generically using 3-rewrite proofs.

    Mirrors: crates/gororoba_algebra/src/construction/cd_tower.rs *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(** * Module Type: a CD algebra with linearity axioms. *)

Module Type CDAlgLinear.
  Parameter carrier : Type.

  Parameter cd_mul : carrier -> carrier -> carrier.
  Parameter cd_add : carrier -> carrier -> carrier.
  Parameter cd_neg : carrier -> carrier.
  Parameter cd_sub : carrier -> carrier -> carrier.
  Parameter cd_conj : carrier -> carrier.
  Parameter cd_scale : R -> carrier -> carrier.
  Parameter cd_zero : carrier.

  Axiom mul_scale_left : forall r (a b : carrier),
    cd_mul (cd_scale r a) b = cd_scale r (cd_mul a b).
  Axiom mul_scale_right : forall r (a b : carrier),
    cd_mul a (cd_scale r b) = cd_scale r (cd_mul a b).
  Axiom conj_scale : forall r (a : carrier),
    cd_conj (cd_scale r a) = cd_scale r (cd_conj a).
  Axiom neg_scale : forall r (a : carrier),
    cd_neg (cd_scale r a) = cd_scale r (cd_neg a).
  Axiom scale_add : forall r (a b : carrier),
    cd_scale r (cd_add a b) = cd_add (cd_scale r a) (cd_scale r b).
  Axiom scale_sub : forall r (a b : carrier),
    cd_scale r (cd_sub a b) = cd_sub (cd_scale r a) (cd_scale r b).
  Axiom sub_def : forall (a b : carrier),
    cd_sub a b = cd_add a (cd_neg b).
End CDAlgLinear.

(** * The CDDouble functor: doubles any CDAlgLinear.

    Given an algebra A, produces a new algebra with carrier
    (A.carrier, A.carrier) and the standard CD multiplication:

      (a,b)*(c,d) = (a*c - conj(d)*b,  d*a + b*conj(c))

    All linearity axioms are proven generically in 3 rewrites. *)

Module CDDouble (A : CDAlgLinear) <: CDAlgLinear.

  Record carrier_ := mkDouble { lo : A.carrier; hi : A.carrier }.
  Definition carrier := carrier_.

  Definition cd_mul (x y : carrier) : carrier :=
    mkDouble
      (A.cd_sub (A.cd_mul (lo x) (lo y))
                (A.cd_mul (A.cd_conj (hi y)) (hi x)))
      (A.cd_add (A.cd_mul (hi y) (lo x))
                (A.cd_mul (hi x) (A.cd_conj (lo y)))).

  Definition cd_add (x y : carrier) : carrier :=
    mkDouble (A.cd_add (lo x) (lo y)) (A.cd_add (hi x) (hi y)).

  Definition cd_neg (x : carrier) : carrier :=
    mkDouble (A.cd_neg (lo x)) (A.cd_neg (hi x)).

  Definition cd_sub (x y : carrier) : carrier := cd_add x (cd_neg y).

  Definition cd_conj (x : carrier) : carrier :=
    mkDouble (A.cd_conj (lo x)) (A.cd_neg (hi x)).

  Definition cd_scale (r : R) (x : carrier) : carrier :=
    mkDouble (A.cd_scale r (lo x)) (A.cd_scale r (hi x)).

  Definition cd_zero : carrier := mkDouble A.cd_zero A.cd_zero.

  (** ** Generic linearity proofs -- 3 rewrites each, no destruct. *)

  Theorem mul_scale_left : forall r (x y : carrier),
    cd_mul (cd_scale r x) y = cd_scale r (cd_mul x y).
  Proof.
    intros r [xa xb] [ya yb].
    unfold cd_mul, cd_scale; simpl lo; simpl hi.
    f_equal.
    - rewrite A.mul_scale_left.
      rewrite A.mul_scale_right.
      rewrite <- A.scale_sub.
      reflexivity.
    - rewrite A.mul_scale_right.
      rewrite A.mul_scale_left.
      rewrite <- A.scale_add.
      reflexivity.
  Qed.

  Theorem mul_scale_right : forall r (x y : carrier),
    cd_mul x (cd_scale r y) = cd_scale r (cd_mul x y).
  Proof.
    intros r [xa xb] [ya yb].
    unfold cd_mul, cd_scale; simpl lo; simpl hi.
    f_equal.
    - rewrite A.mul_scale_right.
      rewrite A.conj_scale.
      rewrite A.mul_scale_left.
      rewrite <- A.scale_sub.
      reflexivity.
    - rewrite A.mul_scale_left.
      rewrite A.conj_scale.
      rewrite A.mul_scale_right.
      rewrite <- A.scale_add.
      reflexivity.
  Qed.

  Theorem conj_scale : forall r (x : carrier),
    cd_conj (cd_scale r x) = cd_scale r (cd_conj x).
  Proof.
    intros r [xa xb].
    unfold cd_conj, cd_scale; simpl lo; simpl hi.
    f_equal.
    - apply A.conj_scale.
    - apply A.neg_scale.
  Qed.

  Theorem neg_scale : forall r (x : carrier),
    cd_neg (cd_scale r x) = cd_scale r (cd_neg x).
  Proof.
    intros r [xa xb].
    unfold cd_neg, cd_scale; simpl lo; simpl hi.
    f_equal; apply A.neg_scale.
  Qed.

  Theorem scale_add : forall r (x y : carrier),
    cd_scale r (cd_add x y) = cd_add (cd_scale r x) (cd_scale r y).
  Proof.
    intros r [xa xb] [ya yb].
    unfold cd_scale, cd_add; simpl lo; simpl hi.
    f_equal; apply A.scale_add.
  Qed.

  Theorem scale_sub : forall r (x y : carrier),
    cd_scale r (cd_sub x y) = cd_sub (cd_scale r x) (cd_scale r y).
  Proof.
    intros r [xa xb] [ya yb].
    unfold cd_sub, cd_add, cd_neg, cd_scale; simpl lo; simpl hi.
    f_equal.
    - rewrite A.neg_scale. apply A.scale_add.
    - rewrite A.neg_scale. apply A.scale_add.
  Qed.

  Theorem sub_def : forall (x y : carrier),
    cd_sub x y = cd_add x (cd_neg y).
  Proof. reflexivity. Qed.

End CDDouble.

(** * Demonstration: the functor can be iterated.

    If we had a QuatBase module satisfying CDAlgLinear, then:

      Module Oct := CDDouble QuatBase.
      Module Sed := CDDouble Oct.
      Module Path := CDDouble Sed.
      Module Chingon := CDDouble Path.
      ...

    Each level inherits all 7 linearity theorems automatically.
    The entire tower from dim=4 to dim=65536 (13 levels) would
    require only the single quaternion base case proof. *)
