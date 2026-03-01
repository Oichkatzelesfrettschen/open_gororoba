(** * FloatAxioms: abstract field with transcendentals.

    Defines a module type FLOAT_OPS that captures the algebraic properties
    needed by quaternion rotation and warp shape functions. The idea:
    prove theorems once in terms of FLOAT_OPS, then extract to OCaml float
    (or Rust f64) by mapping the abstract operations to concrete ones.

    The axioms include:
    - Field axioms (add, mul, sub, div, neg, zero, one)
    - Ordering (lt)
    - Square root
    - No decidable equality (extraction-compatible) *)

From OpenGororoba Require Import Prelude.

(*<*floatops>*)
Module Type FLOAT_OPS.

  Parameter t : Set.
  Parameter zero one : t.
  Parameter add mul sub neg : t -> t -> t.
  Parameter opp : t -> t.
  Parameter div : t -> t -> t.
  Parameter sqrt_f : t -> t.
  Parameter lt : t -> t -> Prop.

  Notation "x + y" := (add x y).
  Notation "x * y" := (mul x y).
  Notation "x - y" := (sub x y).
  Notation "x / y" := (div x y).
  Notation "- x" := (opp x).

  (** Field axioms *)
  Axiom add_comm : forall x y, add x y = add y x.
  Axiom add_assoc : forall x y z, add (add x y) z = add x (add y z).
  Axiom add_zero_l : forall x, add zero x = x.
  Axiom add_opp_r : forall x, add x (opp x) = zero.
  Axiom mul_comm : forall x y, mul x y = mul y x.
  Axiom mul_assoc : forall x y z, mul (mul x y) z = mul x (mul y z).
  Axiom mul_one_l : forall x, mul one x = x.
  Axiom mul_add_distr_l : forall x y z, mul x (add y z) = add (mul x y) (mul x z).
  Axiom sub_def : forall x y, sub x y = add x (opp y).

End FLOAT_OPS.
(*</floatops>*)
