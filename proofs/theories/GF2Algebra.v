(** * GF(2) polynomial algebra for projective geometry separation.

    Defines evaluation and separation of GF(2) polynomials over
    finite point sets, used to formalize the minimum separating
    degree in PG(3,2).

    A GF(2) monomial is represented by a bitmask: monomial m evaluates
    to 1 on point p iff all bits of m are set in p.  A polynomial is
    a list of monomials; evaluation XORs monomial values (addition mod 2).

    Mirrors: algebra_analysis::projective_geometry::find_boolean_class_predicate *)

From Stdlib Require Import Bool Arith PeanoNat Lia List.
Import ListNotations.

(** Count set bits in a natural number (popcount). *)
Fixpoint popcount (n : nat) (bits : nat) : nat :=
  match bits with
  | 0 => 0
  | S b => (if Nat.testbit n b then 1 else 0) + popcount n b
  end.

(** Monomial degree is its popcount (number of variables). *)
Definition monomial_degree (mono n_bits : nat) : nat :=
  popcount mono n_bits.

(** Evaluate a single GF(2) monomial on a point.
    True iff all bits of the monomial mask are set in the label. *)
Definition gf2_monomial_eval (mono label : nat) : bool :=
  Nat.eqb (Nat.land mono label) mono.

(** Evaluate a GF(2) polynomial (list of monomials) on a point.
    Result is XOR of individual monomial evaluations. *)
Fixpoint gf2_poly_eval (monomials : list nat) (label : nat) : bool :=
  match monomials with
  | [] => false
  | m :: rest => xorb (gf2_monomial_eval m label) (gf2_poly_eval rest label)
  end.

(** Boolean test: does a polynomial separate labels according to classes?
    Returns true iff the polynomial evaluates to classes[i] for all i. *)
(*<*gf2sep>*)
Fixpoint separates_b (labels : list nat) (classes : list bool)
    (monos : list nat) : bool :=
  match labels, classes with
  | [], [] => true
  | l :: ls, c :: cs =>
      Bool.eqb (gf2_poly_eval monos l) c && separates_b ls cs monos
  | _, _ => false
  end.
(*</gf2sep>*)

(** Prop version of separation. *)
Definition separates (labels : list nat) (classes : list bool)
    (monos : list nat) : Prop :=
  separates_b labels classes monos = true.

(** Maximum monomial degree in a list. *)
Fixpoint max_degree (monos : list nat) (n_bits : nat) : nat :=
  match monos with
  | [] => 0
  | m :: rest => Nat.max (monomial_degree m n_bits) (max_degree rest n_bits)
  end.
