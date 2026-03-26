(** * CDNegLemmas: Negation, addition, zero, and identity lemmas for the CD tower.

    Tower strategy (mirrors CDLinearLemmas.v):
      quat level -> oct level -> sed level
    Each higher level lifts from the previous via CD doubling structure.

    Covers:
    - Double negation: neg(neg x) = x
    - Negation distributes over addition: neg(a + b) = neg a + neg b
    - Negation distributes over multiplication: neg(a) * b = neg(a*b), etc.
    - Addition commutativity and zero element
    - Scale by zero: scale 0 x = zero
    - Right identity: x * one = x

    Supports: CDTraceZero.v, C1538_MorZDSymmetry.v, C1539_MorSkewSymm.v *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import CDConjAntimorph.
Open Scope R_scope.

(** ================================================================== *)
(** * Quaternion level (destruct + ring).                               *)
(** ================================================================== *)

(** Addition: zero element. *)
Lemma quat_add_zero_left : forall q, quat_add quat_zero q = q.
Proof. intros [a b c d]; unfold quat_add, quat_zero; simpl; f_equal; ring. Qed.

Lemma quat_add_zero_right : forall q, quat_add q quat_zero = q.
Proof. intros [a b c d]; unfold quat_add, quat_zero; simpl; f_equal; ring. Qed.

(** Scale by zero. *)
Lemma quat_scale_zero : forall q, quat_scale 0 q = quat_zero.
Proof. intros [a b c d]; unfold quat_scale, quat_zero; simpl; f_equal; ring. Qed.

(** Negation distributes over addition. *)
Lemma quat_neg_add : forall a b,
  quat_neg (quat_add a b) = quat_add (quat_neg a) (quat_neg b).
Proof. intros [a1 a2 a3 a4] [b1 b2 b3 b4]; unfold quat_neg, quat_add; simpl; f_equal; ring. Qed.

(** neg(zero) = zero. *)
Lemma quat_neg_zero : quat_neg quat_zero = quat_zero.
Proof. unfold quat_neg, quat_zero; simpl; f_equal; ring. Qed.

(** Right identity: q * 1 = q. *)
Lemma quat_mul_one_right : forall q, quat_mul q quat_one = q.
Proof. intros [a b c d]; unfold quat_mul, quat_one; simpl; f_equal; ring. Qed.

(** Left identity: 1 * q = q. *)
Lemma quat_mul_one_left : forall q, quat_mul quat_one q = q.
Proof. intros [a b c d]; unfold quat_mul, quat_one; simpl; f_equal; ring. Qed.

(** Cancellation: a + neg a = zero. *)
Lemma quat_add_neg_cancel : forall q, quat_add q (quat_neg q) = quat_zero.
Proof. intros [a b c d]; unfold quat_add, quat_neg, quat_zero; simpl; f_equal; ring. Qed.

(** Scale by -r: scale(-r, x) = neg(scale(r, x)). *)
Lemma quat_scale_neg_r : forall r q,
  quat_scale (- r) q = quat_neg (quat_scale r q).
Proof. intros r [a b c d]; unfold quat_scale, quat_neg; simpl; f_equal; ring. Qed.

(** ================================================================== *)
(** * Octonion level (tower lift from quaternion).                     *)
(** ================================================================== *)

Lemma oct_neg_neg : forall x : CDOct, oct_neg (oct_neg x) = x.
Proof.
  intros [lo hi]; unfold oct_neg; simpl; f_equal; apply quat_neg_neg.
Qed.

Lemma oct_neg_mul_left : forall a b : CDOct,
  oct_mul (oct_neg a) b = oct_neg (oct_mul a b).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [oct_mul oct_neg oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma oct_neg_mul_right : forall a b : CDOct,
  oct_mul a (oct_neg b) = oct_neg (oct_mul a b).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [oct_mul oct_neg oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma oct_add_comm : forall a b : CDOct, oct_add a b = oct_add b a.
Proof.
  intros [alo ahi] [blo bhi]; unfold oct_add; simpl; f_equal; apply quat_add_comm.
Qed.

Lemma oct_add_zero_left : forall x : CDOct, oct_add oct_zero x = x.
Proof.
  intros [lo hi]; unfold oct_add, oct_zero; simpl; f_equal; apply quat_add_zero_left.
Qed.

Lemma oct_add_zero_right : forall x : CDOct, oct_add x oct_zero = x.
Proof.
  intros [lo hi]; unfold oct_add, oct_zero; simpl; f_equal; apply quat_add_zero_right.
Qed.

Lemma oct_scale_zero : forall x : CDOct, oct_scale 0 x = oct_zero.
Proof.
  intros [lo hi]; unfold oct_scale, oct_zero; simpl; f_equal; apply quat_scale_zero.
Qed.

Lemma oct_neg_add : forall a b : CDOct,
  oct_neg (oct_add a b) = oct_add (oct_neg a) (oct_neg b).
Proof.
  intros [alo ahi] [blo bhi]; unfold oct_neg, oct_add; simpl; f_equal; apply quat_neg_add.
Qed.

Lemma oct_neg_zero : oct_neg oct_zero = oct_zero.
Proof. unfold oct_neg, oct_zero; simpl; f_equal; apply quat_neg_zero. Qed.

Lemma oct_mul_one_right : forall x : CDOct,
  oct_mul x (mkOct quat_one quat_zero) = x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_mul oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma oct_add_neg_cancel : forall x : CDOct, oct_add x (oct_neg x) = oct_zero.
Proof.
  intros [lo hi]; unfold oct_add, oct_neg, oct_zero; simpl; f_equal; apply quat_add_neg_cancel.
Qed.

Lemma oct_scale_neg_r : forall r (x : CDOct),
  oct_scale (- r) x = oct_neg (oct_scale r x).
Proof.
  intros r [lo hi]; unfold oct_scale, oct_neg; simpl; f_equal; apply quat_scale_neg_r.
Qed.

Lemma oct_mul_zero_left : forall x : CDOct, oct_mul oct_zero x = oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma oct_mul_zero_right : forall x : CDOct, oct_mul x oct_zero = oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Sedenion level (tower lift from octonion).                       *)
(** ================================================================== *)

Lemma sed_neg_neg : forall x : CDSed, sed_neg (sed_neg x) = x.
Proof.
  intros [lo hi]; unfold sed_neg; simpl; f_equal; apply oct_neg_neg.
Qed.

Lemma sed_neg_mul_left : forall a b : CDSed,
  sed_mul (sed_neg a) b = sed_neg (sed_mul a b).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_mul sed_neg sed_lo sed_hi
       oct_mul oct_neg oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Lemma sed_neg_mul_right : forall a b : CDSed,
  sed_mul a (sed_neg b) = sed_neg (sed_mul a b).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_mul sed_neg sed_lo sed_hi
       oct_mul oct_neg oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Lemma sed_add_comm : forall a b : CDSed, sed_add a b = sed_add b a.
Proof.
  intros [alo ahi] [blo bhi]; unfold sed_add; simpl; f_equal; apply oct_add_comm.
Qed.

Lemma sed_add_zero_left : forall x : CDSed, sed_add sed_zero x = x.
Proof.
  intros [lo hi]; unfold sed_add, sed_zero; simpl; f_equal; apply oct_add_zero_left.
Qed.

Lemma sed_add_zero_right : forall x : CDSed, sed_add x sed_zero = x.
Proof.
  intros [lo hi]; unfold sed_add, sed_zero; simpl; f_equal; apply oct_add_zero_right.
Qed.

Lemma sed_scale_zero : forall x : CDSed, sed_scale 0 x = sed_zero.
Proof.
  intros [lo hi]; unfold sed_scale, sed_zero; simpl; f_equal; apply oct_scale_zero.
Qed.

Lemma sed_neg_add : forall a b : CDSed,
  sed_neg (sed_add a b) = sed_add (sed_neg a) (sed_neg b).
Proof.
  intros [alo ahi] [blo bhi]; unfold sed_neg, sed_add; simpl; f_equal; apply oct_neg_add.
Qed.

Lemma sed_neg_zero : sed_neg sed_zero = sed_zero.
Proof. unfold sed_neg, sed_zero; simpl; f_equal; apply oct_neg_zero. Qed.

Definition sed_one : CDSed := mkSed (mkOct quat_one quat_zero) oct_zero.

Lemma sed_mul_one_right : forall x : CDSed, sed_mul x sed_one = x.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [sed_mul sed_one sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Lemma sed_add_neg_cancel : forall x : CDSed, sed_add x (sed_neg x) = sed_zero.
Proof.
  intros [lo hi]; unfold sed_add, sed_neg, sed_zero; simpl; f_equal; apply oct_add_neg_cancel.
Qed.

Lemma sed_scale_neg_r : forall r (x : CDSed),
  sed_scale (- r) x = sed_neg (sed_scale r x).
Proof.
  intros r [lo hi]; unfold sed_scale, sed_neg; simpl; f_equal; apply oct_scale_neg_r.
Qed.

Lemma sed_mul_zero_left : forall x : CDSed, sed_mul sed_zero x = sed_zero.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]]].
  cbv [sed_mul sed_zero sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Lemma sed_mul_zero_right : forall x : CDSed, sed_mul x sed_zero = sed_zero.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]]].
  cbv [sed_mul sed_zero sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.
