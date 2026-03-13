(** * C-1313: Thesis 42 arithmetic scaffold.

    This file formalizes the discrete arithmetic identities that support the
    evidence-first "42 / 1764" thesis bundle, without promoting any stronger
    physical interpretation.

    Formalized facts:
    - the 5th Catalan number equals 42;
    - the already-formalized sedenion assessor count matches Catalan(5);
    - the square bound is 42^2 = 1764;
    - the normalized scale used in the thesis bundle is exactly 1764/10000
      = 441/2500.

    Mirrors: crates/gororoba_cli/src/thesis_42_support.rs *)

From Stdlib Require Import Arith List.
From OpenGororoba Require Import Prelude BoxKite.

Import ListNotations.

(** Closed-form Catalan number formula:
    C_n = (2n)! / (n! (n+1)!). *)
Definition catalan_formula (n : nat) : nat :=
  Nat.div (fact (2 * n)) (fact n * fact (S n)).

(** The 5th Catalan number is 42. *)
Theorem catalan_5_value :
  catalan_formula 5 = 42%nat.
Proof. vm_compute. reflexivity. Qed.

(** The existing assessor count coincides with Catalan(5). *)
Theorem assessor_count_matches_catalan_5 :
  length assessors = catalan_formula 5.
Proof.
  rewrite assessor_count.
  rewrite catalan_5_value.
  reflexivity.
Qed.

(** The box-kite partition total also lands on Catalan(5). *)
Theorem boxkite_total_matches_catalan_5 :
  List.fold_left Nat.add (List.map (@length _) boxkites) 0%nat = catalan_formula 5.
Proof.
  rewrite boxkite_total.
  rewrite catalan_5_value.
  reflexivity.
Qed.

(** Squaring the assessor count gives the thesis square bound 1764. *)
Definition thesis_square_bound : nat :=
  length assessors * length assessors.

Theorem thesis_square_bound_value :
  thesis_square_bound = 1764%nat.
Proof.
  unfold thesis_square_bound.
  rewrite assessor_count.
  vm_compute.
  reflexivity.
Qed.

(** The normalized decimal scale behind "0.1764" is the exact rational
    1764 / 10000 = 441 / 2500. *)
Open Scope R_scope.

Definition normalized_square_bound : R :=
  INR thesis_square_bound / 10000.

Theorem normalized_square_bound_exact :
  normalized_square_bound = 441 / 2500.
Proof.
  unfold normalized_square_bound.
  rewrite thesis_square_bound_value.
  simpl.
  field.
Qed.

Theorem normalized_square_bound_in_unit_interval :
  0 < normalized_square_bound < 1.
Proof.
  rewrite normalized_square_bound_exact.
  split; lra.
Qed.
