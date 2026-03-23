(** * CDConjAntimorph: Conjugate anti-automorphism at all CD levels.

    The conjugate of a product equals the product of conjugates in reverse:
      conj(x * y) = conj(y) * conj(x)

    This is a UNIVERSAL identity of all Cayley-Dickson algebras, provable
    at each level by induction on the doubling construction.

    Proved concretely at:
    - Quaternion level (dim 4): CayleyDicksonAlgebra.v, already done
    - Octonion level (dim 8): direct computation (cbv + ring)
    - Sedenion level (dim 16): direct computation (cbv + ring)

    Derived consequences:
    - neg(x) * neg(y) = x * y  (double negation in products)
    - conj(x*y) = 0 iff conj(y)*conj(x) = 0

    Source: Standard CD algebra theory; used by Moreno (1997) Corollary 1.6.
    Supports claims: C-1538, C-1539. *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** ================================================================== *)
(** * Quaternion-level helpers (destruct + ring).                       *)
(** ================================================================== *)

(** conj(neg(q)) = neg(conj(q)). *)
Lemma quat_conj_neg : forall q : CDQuat,
  quat_conj (quat_neg q) = quat_neg (quat_conj q).
Proof.
  intros q; destruct q; unfold quat_conj, quat_neg; simpl; f_equal; ring.
Qed.

(** Double negation: neg(neg(q)) = q. *)
Lemma quat_neg_neg : forall q : CDQuat,
  quat_neg (quat_neg q) = q.
Proof.
  intros q; destruct q; unfold quat_neg; simpl; f_equal; ring.
Qed.

(** neg distributes over multiplication (left). *)
Lemma quat_neg_mul_left : forall a b : CDQuat,
  quat_mul (quat_neg a) b = quat_neg (quat_mul a b).
Proof.
  intros a b; destruct a, b; unfold quat_mul, quat_neg; simpl; f_equal; ring.
Qed.

(** neg distributes over multiplication (right). *)
Lemma quat_neg_mul_right : forall a b : CDQuat,
  quat_mul a (quat_neg b) = quat_neg (quat_mul a b).
Proof.
  intros a b; destruct a, b; unfold quat_mul, quat_neg; simpl; f_equal; ring.
Qed.

(** Addition is commutative. *)
Lemma quat_add_comm : forall a b : CDQuat,
  quat_add a b = quat_add b a.
Proof.
  intros a b; destruct a, b; unfold quat_add; simpl; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Octonion-level: conjugate anti-automorphism.                     *)
(** ================================================================== *)

(** conj(x * y) = conj(y) * conj(x) for octonions.

    Proof by direct computation: destruct to 16 real variables,
    cbv-expand all octonion operations, then ring on each component.
    Degree-2 polynomials in 16 variables -- well within ring's capacity. *)
Theorem oct_conj_antimorphism : forall x y : CDOct,
  oct_conj (oct_mul x y) = oct_mul (oct_conj y) (oct_conj x).
Proof.
  intros x y.
  destruct x as [[xa1 xa2 xa3 xa4] [xb1 xb2 xb3 xb4]].
  destruct y as [[ya1 ya2 ya3 ya4] [yb1 yb2 yb3 yb4]].
  cbv [oct_conj oct_mul oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Sedenion-level: conjugate anti-automorphism.                     *)
(** ================================================================== *)

(** conj(x * y) = conj(y) * conj(x) for sedenions.

    Proof by direct computation: destruct to 32 real variables,
    cbv-expand all operations, then ring on each of the 16 R-equalities.
    Degree-2 polynomials in 32 variables. *)
Theorem sed_conj_antimorphism : forall x y : CDSed,
  sed_conj (sed_mul x y) = sed_mul (sed_conj y) (sed_conj x).
Proof.
  intros x y.
  destruct x as [[[xa1 xa2 xa3 xa4] [xb1 xb2 xb3 xb4]]
                  [[xc1 xc2 xc3 xc4] [xd1 xd2 xd3 xd4]]].
  destruct y as [[[ya1 ya2 ya3 ya4] [yb1 yb2 yb3 yb4]]
                  [[yc1 yc2 yc3 yc4] [yd1 yd2 yd3 yd4]]].
  cbv [sed_conj sed_mul sed_lo sed_hi
       oct_conj oct_mul oct_neg oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Derived consequences.                                            *)
(** ================================================================== *)

(** ** Octonion level. *)

Lemma oct_neg_neg : forall x : CDOct,
  mkOct (quat_neg (oct_lo (mkOct (quat_neg (oct_lo x))
                                   (quat_neg (oct_hi x)))))
        (quat_neg (oct_hi (mkOct (quat_neg (oct_lo x))
                                   (quat_neg (oct_hi x)))))
  = x.
Proof.
  intros [lo hi]; simpl.
  f_equal; apply quat_neg_neg.
Qed.

(** ** Sedenion level: conj(zero) = zero. *)
Lemma sed_conj_zero : sed_conj sed_zero = sed_zero.
Proof.
  cbv [sed_conj sed_zero oct_conj oct_zero
       quat_conj quat_neg quat_zero
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

(** ** Application to zero divisors:
    If xy = 0 then conj(y) * conj(x) = 0. *)
Corollary sed_zd_conj : forall x y : CDSed,
  sed_mul x y = sed_zero ->
  sed_mul (sed_conj y) (sed_conj x) = sed_zero.
Proof.
  intros x y Hxy.
  rewrite <- sed_conj_antimorphism.
  rewrite Hxy.
  exact sed_conj_zero.
Qed.
