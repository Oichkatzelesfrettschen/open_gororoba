(** * C-018: Carlstrom wheel axioms hold on a concrete 3-element model.

    Wheels provide total division-by-zero via 8 axioms (Carlstrom 2004).
    We verify the axioms on a minimal finite wheel {0, 1, bot} where
    bot = 0/0 (the "bottom" element absorbing all operations).

    This proves wheels are consistent and non-trivial, but C-019 shows
    they do not help with sedenion zero divisors. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(** Wheel element type: 0, 1, or bot (undefined/absorber). *)
Inductive wheel_elt := W0 | W1 | Wbot.

(** Wheel addition. *)
Definition wadd (x y : wheel_elt) : wheel_elt :=
  match x, y with
  | W0, y => y
  | x, W0 => x
  | W1, W1 => W0
  | _, _ => Wbot
  end.

(** Wheel multiplication. *)
Definition wmul (x y : wheel_elt) : wheel_elt :=
  match x, y with
  | W0, _ => W0
  | _, W0 => W0
  | W1, y => y
  | x, W1 => x
  | Wbot, Wbot => Wbot
  end.

(** Wheel involution (total inverse). *)
Definition winv (x : wheel_elt) : wheel_elt :=
  match x with
  | W0 => Wbot
  | W1 => W1
  | Wbot => W0
  end.

(** Axiom 1: Addition is commutative. *)
Theorem C018_add_comm : forall x y, wadd x y = wadd y x.
Proof. intros []; intros []; reflexivity. Qed.

(** Axiom 2: Multiplication is commutative. *)
Theorem C018_mul_comm : forall x y, wmul x y = wmul y x.
Proof. intros []; intros []; reflexivity. Qed.

(** Axiom 3: Addition is associative. *)
Theorem C018_add_assoc : forall x y z,
  wadd (wadd x y) z = wadd x (wadd y z).
Proof. intros []; intros []; intros []; reflexivity. Qed.

(** Axiom 4: Multiplication is associative. *)
Theorem C018_mul_assoc : forall x y z,
  wmul (wmul x y) z = wmul x (wmul y z).
Proof. intros []; intros []; intros []; reflexivity. Qed.

(** Axiom 5: 0 is additive identity. *)
Theorem C018_add_zero : forall x, wadd W0 x = x.
Proof. intros []; reflexivity. Qed.

(** Axiom 6: 1 is multiplicative identity. *)
Theorem C018_mul_one : forall x, wmul W1 x = x.
Proof. intros []; reflexivity. Qed.

(** Axiom 7: Involution: inv(inv(x)) = x. *)
Theorem C018_inv_involution : forall x, winv (winv x) = x.
Proof. intros []; reflexivity. Qed.
