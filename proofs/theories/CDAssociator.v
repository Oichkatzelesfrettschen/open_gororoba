(** * CDAssociator: Associator and flexibility at dim=4.

    The associator [a,b,c] := (a*b)*c - a*(b*c) measures deviation from
    associativity.  For quaternions it vanishes identically, which implies
    flexibility a*(b*a) = (a*b)*a and the Jordan identity.

    Mirrors: algebra_core/src/construction/cayley_dickson.rs (proptest). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

(** Quaternion associator: [a,b,c] = (a*b)*c - a*(b*c). *)
Definition quat_assoc (a b c : CDQuat) : CDQuat :=
  quat_add (quat_mul (quat_mul a b) c)
           (quat_neg (quat_mul a (quat_mul b c))).

(** Quaternion associator vanishes (H is associative). *)
Theorem quat_assoc_zero : forall a b c,
  quat_assoc a b c = quat_zero.
Proof.
  intros; destruct a, b, c;
  unfold quat_assoc, quat_add, quat_neg, quat_mul, quat_zero; simpl;
  f_equal; ring.
Qed.

(** Flexibility: (a*b)*a = a*(b*a) at dim=4. *)
Theorem quat_flexibility : forall a b,
  quat_mul (quat_mul a b) a = quat_mul a (quat_mul b a).
Proof.
  intros; destruct a, b;
  unfold quat_mul; simpl; f_equal; ring.
Qed.

(** Jordan identity: (a^2*b)*a = a^2*(b*a) at dim=4.
    This follows from associativity but holds independently in any
    alternative algebra (like octonions). *)
Theorem quat_jordan : forall a b,
  quat_mul (quat_mul (quat_mul a a) b) a =
  quat_mul (quat_mul a a) (quat_mul b a).
Proof.
  intros; destruct a, b;
  unfold quat_mul; simpl; f_equal; ring.
Qed.
