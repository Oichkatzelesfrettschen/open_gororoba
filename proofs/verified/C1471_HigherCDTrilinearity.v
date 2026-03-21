(** * C-1471: Trilinearity at dim=64, 128, 256 via strict tower lift.

    STRICT ABSTRACTION: each level uses ONLY the previous level's lemmas.
    Proof at each level is ~4 rewrites + reflexivity.

    Dependency order:
      SED LEVEL: sed_scale_add, sed_conj_scale, sed_neg_scale
      PATHION LEVEL: pathion_mul_scale_right, pathion_conj_scale,
                     pathion_scale_add
      CHINGON LEVEL: chingon_mul_scale_left/right, chingon_scale_sub/add,
                     chingon_conj_scale, chingon trilinearity
      ROUTON/VOUDON: same pattern

    Mirrors: crates/gororoba_algebra/src/construction/cd_tower.rs *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
                                 OctonionNorm CDLinearLemmas Pathion HigherCD.
From OpenGororobaVerified Require Import C1455_AssociatorTrilinear
                                         C1470_PathionTrilinearity.
Open Scope R_scope.

(** ========== SED-LEVEL MISSING LEMMAS ========== *)

Lemma sed_neg_scale : forall r x,
  sed_neg (sed_scale r x) = sed_scale r (sed_neg x).
Proof.
  intros r [[a1 a2] [a3 a4]].
  destruct a1, a2, a3, a4.
  unfold sed_neg, sed_scale, oct_neg, oct_scale, quat_neg, quat_scale;
  simpl; f_equal; f_equal; f_equal; f_equal; ring.
Qed.

Lemma sed_conj_scale : forall r x,
  sed_conj (sed_scale r x) = sed_scale r (sed_conj x).
Proof.
  intros r [[xllo xlhi] [xhlo xhhi]].
  destruct xllo, xlhi, xhlo, xhhi.
  unfold sed_conj, sed_scale, sed_neg, oct_scale, oct_neg, oct_conj,
         quat_scale, quat_neg, quat_conj;
  simpl; f_equal; f_equal; f_equal; f_equal; ring.
Qed.

(** ========== PATHION-LEVEL MISSING LEMMAS ========== *)

Lemma pathion_scale_add : forall r a b,
  pathion_add (pathion_scale r a) (pathion_scale r b) =
  pathion_scale r (pathion_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold pathion_scale, pathion_add.
  simpl pathion_lo; simpl pathion_hi.
  f_equal; apply sed_scale_add.
Qed.

Theorem pathion_mul_scale_right : forall r x y,
  pathion_mul x (pathion_scale r y) = pathion_scale r (pathion_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold pathion_mul, pathion_scale.
  simpl pathion_lo; simpl pathion_hi.
  f_equal.
  - rewrite sed_mul_scale_right.
    rewrite sed_conj_scale. rewrite sed_mul_scale_left.
    rewrite <- sed_scale_sub. reflexivity.
  - rewrite sed_mul_scale_left.
    rewrite sed_conj_scale. rewrite sed_mul_scale_right.
    rewrite <- sed_scale_add. reflexivity.
Qed.

Lemma pathion_conj_scale : forall r x,
  pathion_conj (pathion_scale r x) = pathion_scale r (pathion_conj x).
Proof.
  intros r [xlo xhi].
  unfold pathion_conj, pathion_scale.
  simpl pathion_lo; simpl pathion_hi.
  f_equal. apply sed_conj_scale. apply sed_neg_scale.
Qed.

Lemma pathion_neg_scale : forall r x,
  pathion_neg (pathion_scale r x) = pathion_scale r (pathion_neg x).
Proof.
  intros r [xlo xhi].
  unfold pathion_neg, pathion_scale.
  simpl pathion_lo; simpl pathion_hi.
  f_equal; apply sed_neg_scale.
Qed.

(** ========== CHINGON (dim=64) ========== *)

Theorem chingon_mul_scale_left : forall r x y,
  chingon_mul (chingon_scale r x) y = chingon_scale r (chingon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold chingon_mul, chingon_scale.
  simpl chingon_lo; simpl chingon_hi.
  f_equal.
  - rewrite pathion_mul_scale_left. rewrite pathion_mul_scale_right.
    rewrite <- pathion_scale_sub. reflexivity.
  - rewrite pathion_mul_scale_right. rewrite pathion_mul_scale_left.
    rewrite <- pathion_scale_add. reflexivity.
Qed.

Theorem chingon_mul_scale_right : forall r x y,
  chingon_mul x (chingon_scale r y) = chingon_scale r (chingon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold chingon_mul, chingon_scale.
  simpl chingon_lo; simpl chingon_hi.
  f_equal.
  - rewrite pathion_mul_scale_right.
    rewrite pathion_conj_scale. rewrite pathion_mul_scale_left.
    rewrite <- pathion_scale_sub. reflexivity.
  - rewrite pathion_mul_scale_left.
    rewrite pathion_conj_scale. rewrite pathion_mul_scale_right.
    rewrite <- pathion_scale_add. reflexivity.
Qed.

Lemma chingon_scale_sub : forall r a b,
  chingon_scale r (chingon_sub a b) =
  chingon_sub (chingon_scale r a) (chingon_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold chingon_scale, chingon_sub, chingon_add, chingon_neg.
  simpl chingon_lo; simpl chingon_hi.
  f_equal; apply pathion_scale_sub.
Qed.

Theorem chingon_assoc_trilinear_scale_1 : forall r a b c,
  chingon_assoc (chingon_scale r a) b c =
  chingon_scale r (chingon_assoc a b c).
Proof.
  intros. unfold chingon_assoc.
  rewrite chingon_mul_scale_left. rewrite chingon_mul_scale_left.
  rewrite chingon_mul_scale_left. rewrite <- chingon_scale_sub.
  reflexivity.
Qed.

(** ========== ROUTON (dim=128) ========== *)

Lemma chingon_scale_add : forall r a b,
  chingon_add (chingon_scale r a) (chingon_scale r b) =
  chingon_scale r (chingon_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold chingon_scale, chingon_add.
  simpl chingon_lo; simpl chingon_hi.
  f_equal; apply pathion_scale_add.
Qed.

Lemma chingon_conj_scale : forall r x,
  chingon_conj (chingon_scale r x) = chingon_scale r (chingon_conj x).
Proof.
  intros r [xlo xhi].
  unfold chingon_conj, chingon_scale.
  simpl chingon_lo; simpl chingon_hi.
  f_equal. apply pathion_conj_scale. apply pathion_neg_scale.
Qed.

Theorem routon_mul_scale_left : forall r x y,
  routon_mul (routon_scale r x) y = routon_scale r (routon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold routon_mul, routon_scale.
  simpl routon_lo; simpl routon_hi.
  f_equal.
  - rewrite chingon_mul_scale_left. rewrite chingon_mul_scale_right.
    rewrite <- chingon_scale_sub. reflexivity.
  - rewrite chingon_mul_scale_right. rewrite chingon_mul_scale_left.
    rewrite <- chingon_scale_add. reflexivity.
Qed.

Lemma routon_scale_sub : forall r a b,
  routon_scale r (routon_sub a b) =
  routon_sub (routon_scale r a) (routon_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold routon_scale, routon_sub, routon_add, routon_neg.
  simpl routon_lo; simpl routon_hi.
  f_equal; apply chingon_scale_sub.
Qed.

Theorem routon_assoc_trilinear_scale_1 : forall r a b c,
  routon_assoc (routon_scale r a) b c =
  routon_scale r (routon_assoc a b c).
Proof.
  intros. unfold routon_assoc.
  rewrite routon_mul_scale_left. rewrite routon_mul_scale_left.
  rewrite routon_mul_scale_left. rewrite <- routon_scale_sub.
  reflexivity.
Qed.

(** ========== VOUDON (dim=256) ========== *)

Theorem routon_mul_scale_right : forall r x y,
  routon_mul x (routon_scale r y) = routon_scale r (routon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold routon_mul, routon_scale.
  simpl routon_lo; simpl routon_hi.
  f_equal.
  - rewrite chingon_mul_scale_right.
    rewrite chingon_conj_scale. rewrite chingon_mul_scale_left.
    rewrite <- chingon_scale_sub. reflexivity.
  - rewrite chingon_mul_scale_left.
    rewrite chingon_conj_scale. rewrite chingon_mul_scale_right.
    rewrite <- chingon_scale_add. reflexivity.
Qed.

Lemma routon_scale_add : forall r a b,
  routon_add (routon_scale r a) (routon_scale r b) =
  routon_scale r (routon_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold routon_scale, routon_add.
  simpl routon_lo; simpl routon_hi.
  f_equal; apply chingon_scale_add.
Qed.

Lemma chingon_neg_scale : forall r x,
  chingon_neg (chingon_scale r x) = chingon_scale r (chingon_neg x).
Proof.
  intros r [xlo xhi].
  unfold chingon_neg, chingon_scale.
  simpl chingon_lo; simpl chingon_hi.
  f_equal; apply pathion_neg_scale.
Qed.

Lemma routon_conj_scale : forall r x,
  routon_conj (routon_scale r x) = routon_scale r (routon_conj x).
Proof.
  intros r [xlo xhi].
  unfold routon_conj, routon_scale.
  simpl routon_lo; simpl routon_hi.
  f_equal. apply chingon_conj_scale. apply chingon_neg_scale.
Qed.

Theorem voudon_mul_scale_left : forall r x y,
  voudon_mul (voudon_scale r x) y = voudon_scale r (voudon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold voudon_mul, voudon_scale.
  simpl voudon_lo; simpl voudon_hi.
  f_equal.
  - rewrite routon_mul_scale_left. rewrite routon_mul_scale_right.
    rewrite <- routon_scale_sub. reflexivity.
  - rewrite routon_mul_scale_right. rewrite routon_mul_scale_left.
    rewrite <- routon_scale_add. reflexivity.
Qed.

Lemma voudon_scale_sub : forall r a b,
  voudon_scale r (voudon_sub a b) =
  voudon_sub (voudon_scale r a) (voudon_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold voudon_scale, voudon_sub, voudon_add, voudon_neg.
  simpl voudon_lo; simpl voudon_hi.
  f_equal; apply routon_scale_sub.
Qed.

Theorem voudon_assoc_trilinear_scale_1 : forall r a b c,
  voudon_assoc (voudon_scale r a) b c =
  voudon_scale r (voudon_assoc a b c).
Proof.
  intros. unfold voudon_assoc.
  rewrite voudon_mul_scale_left. rewrite voudon_mul_scale_left.
  rewrite voudon_mul_scale_left. rewrite <- voudon_scale_sub.
  reflexivity.
Qed.
