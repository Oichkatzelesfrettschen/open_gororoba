(** * OctonionNorm: Norm and arithmetic at dims 8 and 16.

    Extends Sedenion.v with addition, negation, norm, and associator
    operations for octonions and sedenions.

    Key results:
    - oct_norm_sq is multiplicative (Hurwitz at dim 8)
    - sed_norm_sq is NOT multiplicative (Hurwitz fails at dim 16)
    - Octonion associator is well-defined and nonzero for (e1,e2,e4)

    Mirrors: cd_kernel/src/cayley_dickson.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

(** * Octonion arithmetic. *)

Definition oct_add (x y : CDOct) : CDOct :=
  mkOct (quat_add (oct_lo x) (oct_lo y))
        (quat_add (oct_hi x) (oct_hi y)).

Definition oct_neg (x : CDOct) : CDOct :=
  mkOct (quat_neg (oct_lo x)) (quat_neg (oct_hi x)).

Definition oct_sub (x y : CDOct) : CDOct :=
  oct_add x (oct_neg y).

(** * Sedenion arithmetic. *)

Definition sed_add (x y : CDSed) : CDSed :=
  mkSed (oct_add (sed_lo x) (sed_lo y))
        (oct_add (sed_hi x) (sed_hi y)).

Definition sed_neg (x : CDSed) : CDSed :=
  mkSed (oct_neg (sed_lo x)) (oct_neg (sed_hi x)).

Definition sed_sub (x y : CDSed) : CDSed :=
  sed_add x (sed_neg y).

(** * Scalar multiplication. *)

Definition oct_scale (r : R) (x : CDOct) : CDOct :=
  mkOct (quat_scale r (oct_lo x)) (quat_scale r (oct_hi x)).

Definition sed_scale (r : R) (x : CDSed) : CDSed :=
  mkSed (oct_scale r (sed_lo x)) (oct_scale r (sed_hi x)).

(** * Norm squared operations. *)

Definition oct_norm_sq (x : CDOct) : R :=
  quat_norm_sq (oct_lo x) + quat_norm_sq (oct_hi x).

Definition sed_norm_sq (x : CDSed) : R :=
  oct_norm_sq (sed_lo x) + oct_norm_sq (sed_hi x).

(** * Hurwitz theorem at dim 8: octonion norm is multiplicative.
    Strategy: destruct to raw R components, then use targeted cbv
    to avoid simpl explosion on the 64-term polynomial. *)

Theorem oct_norm_mul : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq x * oct_norm_sq y.
Proof.
  intros x y.
  destruct x as [[xa xb xc xd] [xe xf xg xh]].
  destruct y as [[ya yb yc yd] [ye yf yg yh]].
  cbv [oct_norm_sq oct_mul oct_conj quat_mul quat_add
       quat_neg quat_conj quat_norm_sq oct_lo oct_hi
       qa qb qc qd].
  ring.
Qed.

(** * Hurwitz FAILS at dim 16: sedenion norm is NOT multiplicative.
    Witness: sed_zd_a * sed_zd_b = 0, but both have norm 2.
    So |a*b|^2 = 0 <> 4 = |a|^2 * |b|^2. *)

Lemma sed_zd_a_norm : sed_norm_sq sed_zd_a = 2.
Proof.
  cbv [sed_norm_sq oct_norm_sq sed_zd_a quat_norm_sq
       sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero
       qa qb qc qd].
  ring.
Qed.

Lemma sed_zd_b_norm : sed_norm_sq sed_zd_b = 2.
Proof.
  cbv [sed_norm_sq oct_norm_sq sed_zd_b quat_norm_sq
       sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero
       qa qb qc qd].
  ring.
Qed.

Lemma sed_zd_product_norm : sed_norm_sq (sed_mul sed_zd_a sed_zd_b) = 0.
Proof.
  cbv [sed_norm_sq oct_norm_sq sed_mul sed_zd_a sed_zd_b
       oct_mul oct_conj quat_mul quat_add quat_neg quat_conj
       quat_norm_sq sed_lo sed_hi oct_lo oct_hi oct_zero
       quat_zero quat_one qa qb qc qd].
  ring.
Qed.

Theorem sed_norm_fails :
  exists x y : CDSed,
    sed_norm_sq (sed_mul x y) <> sed_norm_sq x * sed_norm_sq y.
Proof.
  exists sed_zd_a, sed_zd_b.
  rewrite sed_zd_product_norm, sed_zd_a_norm, sed_zd_b_norm.
  lra.
Qed.

(** * Octonion associator. *)

Definition oct_assoc (a b c : CDOct) : CDOct :=
  oct_sub (oct_mul (oct_mul a b) c) (oct_mul a (oct_mul b c)).

(** The octonion associator is nonzero for (e1, e2, e4).
    Strategy: extract the qd component of oct_hi via f_equal,
    then cbv only the path to that single scalar to avoid
    the full 64-term polynomial explosion. *)
Theorem oct_assoc_nonzero :
  oct_assoc (oct_e 1) (oct_e 2) (oct_e 4) <> oct_zero.
Proof.
  intro H.
  assert (Hhi := f_equal oct_hi H).
  assert (Hd := f_equal qd Hhi).
  cbv [oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       oct_e oct_zero quat_mul quat_add quat_neg quat_conj
       quat_zero quat_one oct_lo oct_hi qa qb qc qd] in Hd.
  lra.
Qed.

(** * Octonion basis norms. *)

(** Every standard basis element has unit norm. *)
Theorem oct_basis_unit_norm : forall i : nat,
  (i < 8)%nat -> oct_norm_sq (oct_e i) = 1.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|[|[|[|[|n]]]]]]]];
    try (cbv [oct_e oct_norm_sq quat_norm_sq
              quat_zero quat_one oct_lo oct_hi
              qa qb qc qd]; ring);
    lia.
Qed.
