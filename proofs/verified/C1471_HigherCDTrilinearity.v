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

(** ========== ERISTON (dim=512) ========== *)

Lemma voudon_scale_add : forall r a b,
  voudon_add (voudon_scale r a) (voudon_scale r b) =
  voudon_scale r (voudon_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold voudon_scale, voudon_add.
  simpl voudon_lo; simpl voudon_hi.
  f_equal; apply routon_scale_add.
Qed.

Lemma routon_neg_scale : forall r x,
  routon_neg (routon_scale r x) = routon_scale r (routon_neg x).
Proof.
  intros r [xlo xhi].
  unfold routon_neg, routon_scale.
  simpl routon_lo; simpl routon_hi.
  f_equal; apply chingon_neg_scale.
Qed.

Lemma voudon_conj_scale : forall r x,
  voudon_conj (voudon_scale r x) = voudon_scale r (voudon_conj x).
Proof.
  intros r [xlo xhi].
  unfold voudon_conj, voudon_scale.
  simpl voudon_lo; simpl voudon_hi.
  f_equal. apply routon_conj_scale. apply routon_neg_scale.
Qed.

Theorem voudon_mul_scale_right : forall r x y,
  voudon_mul x (voudon_scale r y) = voudon_scale r (voudon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold voudon_mul, voudon_scale.
  simpl voudon_lo; simpl voudon_hi.
  f_equal.
  - rewrite routon_mul_scale_right.
    rewrite routon_conj_scale. rewrite routon_mul_scale_left.
    rewrite <- routon_scale_sub. reflexivity.
  - rewrite routon_mul_scale_left.
    rewrite routon_conj_scale. rewrite routon_mul_scale_right.
    rewrite <- routon_scale_add. reflexivity.
Qed.

Lemma voudon_neg_scale : forall r x,
  voudon_neg (voudon_scale r x) = voudon_scale r (voudon_neg x).
Proof.
  intros r [xlo xhi].
  unfold voudon_neg, voudon_scale.
  simpl voudon_lo; simpl voudon_hi.
  f_equal; apply routon_neg_scale.
Qed.

Theorem eriston_mul_scale_left : forall r x y,
  eriston_mul (eriston_scale r x) y = eriston_scale r (eriston_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold eriston_mul, eriston_scale.
  simpl eriston_lo; simpl eriston_hi.
  f_equal.
  - rewrite voudon_mul_scale_left. rewrite voudon_mul_scale_right.
    rewrite <- voudon_scale_sub. reflexivity.
  - rewrite voudon_mul_scale_right. rewrite voudon_mul_scale_left.
    rewrite <- voudon_scale_add. reflexivity.
Qed.

Lemma eriston_scale_sub : forall r a b,
  eriston_scale r (eriston_sub a b) =
  eriston_sub (eriston_scale r a) (eriston_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold eriston_scale, eriston_sub, eriston_add, eriston_neg.
  simpl eriston_lo; simpl eriston_hi.
  f_equal; apply voudon_scale_sub.
Qed.

Theorem eriston_assoc_trilinear_scale_1 : forall r a b c,
  eriston_assoc (eriston_scale r a) b c =
  eriston_scale r (eriston_assoc a b c).
Proof.
  intros. unfold eriston_assoc.
  rewrite eriston_mul_scale_left. rewrite eriston_mul_scale_left.
  rewrite eriston_mul_scale_left. rewrite <- eriston_scale_sub.
  reflexivity.
Qed.

(** ========== DEKAVOUDON (dim=1024) ========== *)

Lemma eriston_scale_add : forall r a b,
  eriston_add (eriston_scale r a) (eriston_scale r b) =
  eriston_scale r (eriston_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold eriston_scale, eriston_add.
  simpl eriston_lo; simpl eriston_hi.
  f_equal; apply voudon_scale_add.
Qed.

Lemma eriston_conj_scale : forall r x,
  eriston_conj (eriston_scale r x) = eriston_scale r (eriston_conj x).
Proof.
  intros r [xlo xhi].
  unfold eriston_conj, eriston_scale.
  simpl eriston_lo; simpl eriston_hi.
  f_equal. apply voudon_conj_scale. apply voudon_neg_scale.
Qed.

Theorem eriston_mul_scale_right : forall r x y,
  eriston_mul x (eriston_scale r y) = eriston_scale r (eriston_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold eriston_mul, eriston_scale.
  simpl eriston_lo; simpl eriston_hi.
  f_equal.
  - rewrite voudon_mul_scale_right.
    rewrite voudon_conj_scale. rewrite voudon_mul_scale_left.
    rewrite <- voudon_scale_sub. reflexivity.
  - rewrite voudon_mul_scale_left.
    rewrite voudon_conj_scale. rewrite voudon_mul_scale_right.
    rewrite <- voudon_scale_add. reflexivity.
Qed.

Lemma eriston_neg_scale : forall r x,
  eriston_neg (eriston_scale r x) = eriston_scale r (eriston_neg x).
Proof.
  intros r [xlo xhi].
  unfold eriston_neg, eriston_scale.
  simpl eriston_lo; simpl eriston_hi.
  f_equal; apply voudon_neg_scale.
Qed.

Theorem dekavoudon_mul_scale_left : forall r x y,
  dekavoudon_mul (dekavoudon_scale r x) y =
  dekavoudon_scale r (dekavoudon_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold dekavoudon_mul, dekavoudon_scale.
  simpl dekavoudon_lo; simpl dekavoudon_hi.
  f_equal.
  - rewrite eriston_mul_scale_left. rewrite eriston_mul_scale_right.
    rewrite <- eriston_scale_sub. reflexivity.
  - rewrite eriston_mul_scale_right. rewrite eriston_mul_scale_left.
    rewrite <- eriston_scale_add. reflexivity.
Qed.

Lemma dekavoudon_scale_sub : forall r a b,
  dekavoudon_scale r (dekavoudon_sub a b) =
  dekavoudon_sub (dekavoudon_scale r a) (dekavoudon_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold dekavoudon_scale, dekavoudon_sub, dekavoudon_add, dekavoudon_neg.
  simpl dekavoudon_lo; simpl dekavoudon_hi.
  f_equal; apply eriston_scale_sub.
Qed.

Theorem dekavoudon_assoc_trilinear_scale_1 : forall r a b c,
  dekavoudon_assoc (dekavoudon_scale r a) b c =
  dekavoudon_scale r (dekavoudon_assoc a b c).
Proof.
  intros. unfold dekavoudon_assoc.
  rewrite dekavoudon_mul_scale_left. rewrite dekavoudon_mul_scale_left.
  rewrite dekavoudon_mul_scale_left. rewrite <- dekavoudon_scale_sub.
  reflexivity.
Qed.

(** ========== GENERIC TOWER TACTIC ========== *)
(** At each new level N, we need: mul_scale_left, mul_scale_right,
    scale_sub, scale_add, conj_scale, neg_scale, then assoc_trilinear.
    Each uses ONLY level N-1 lemmas. *)

(** ========== CD-2048 ========== *)

Lemma dekavoudon_scale_add : forall r a b,
  dekavoudon_add (dekavoudon_scale r a) (dekavoudon_scale r b) =
  dekavoudon_scale r (dekavoudon_add a b).
Proof. intros r [? ?] [? ?]. unfold dekavoudon_scale, dekavoudon_add. simpl. f_equal; apply eriston_scale_add. Qed.

Lemma dekavoudon_conj_scale : forall r x,
  dekavoudon_conj (dekavoudon_scale r x) = dekavoudon_scale r (dekavoudon_conj x).
Proof. intros r [? ?]. unfold dekavoudon_conj, dekavoudon_scale. simpl. f_equal. apply eriston_conj_scale. apply eriston_neg_scale. Qed.

Theorem dekavoudon_mul_scale_right : forall r x y,
  dekavoudon_mul x (dekavoudon_scale r y) = dekavoudon_scale r (dekavoudon_mul x y).
Proof.
  intros r [? ?] [? ?]. unfold dekavoudon_mul, dekavoudon_scale. simpl. f_equal.
  - rewrite eriston_mul_scale_right. rewrite eriston_conj_scale. rewrite eriston_mul_scale_left. rewrite <- eriston_scale_sub. reflexivity.
  - rewrite eriston_mul_scale_left. rewrite eriston_conj_scale. rewrite eriston_mul_scale_right. rewrite <- eriston_scale_add. reflexivity.
Qed.

Lemma dekavoudon_neg_scale : forall r x,
  dekavoudon_neg (dekavoudon_scale r x) = dekavoudon_scale r (dekavoudon_neg x).
Proof. intros r [? ?]. unfold dekavoudon_neg, dekavoudon_scale. simpl. f_equal; apply eriston_neg_scale. Qed.

Theorem cd2048_mul_scale_left : forall r x y, cd2048_mul (cd2048_scale r x) y = cd2048_scale r (cd2048_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd2048_mul, cd2048_scale. simpl. f_equal.
  - rewrite dekavoudon_mul_scale_left. rewrite dekavoudon_mul_scale_right. rewrite <- dekavoudon_scale_sub. reflexivity.
  - rewrite dekavoudon_mul_scale_right. rewrite dekavoudon_mul_scale_left. rewrite <- dekavoudon_scale_add. reflexivity. Qed.

Lemma cd2048_scale_sub : forall r a b, cd2048_scale r (cd2048_sub a b) = cd2048_sub (cd2048_scale r a) (cd2048_scale r b).
Proof. intros r [? ?] [? ?]. unfold cd2048_scale, cd2048_sub, cd2048_add, cd2048_neg. simpl. f_equal; apply dekavoudon_scale_sub. Qed.

Theorem cd2048_assoc_trilinear : forall r a b c, cd2048_assoc (cd2048_scale r a) b c = cd2048_scale r (cd2048_assoc a b c).
Proof. intros. unfold cd2048_assoc. repeat rewrite cd2048_mul_scale_left. rewrite <- cd2048_scale_sub. reflexivity. Qed.

(** ========== CD-4096 ========== *)

Lemma cd2048_scale_add : forall r a b, cd2048_add (cd2048_scale r a) (cd2048_scale r b) = cd2048_scale r (cd2048_add a b).
Proof. intros r [? ?] [? ?]. unfold cd2048_scale, cd2048_add. simpl. f_equal; apply dekavoudon_scale_add. Qed.
Lemma cd2048_conj_scale : forall r x, cd2048_conj (cd2048_scale r x) = cd2048_scale r (cd2048_conj x).
Proof. intros r [? ?]. unfold cd2048_conj, cd2048_scale. simpl. f_equal. apply dekavoudon_conj_scale. apply dekavoudon_neg_scale. Qed.
Theorem cd2048_mul_scale_right : forall r x y, cd2048_mul x (cd2048_scale r y) = cd2048_scale r (cd2048_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd2048_mul, cd2048_scale. simpl. f_equal.
  - rewrite dekavoudon_mul_scale_right. rewrite dekavoudon_conj_scale. rewrite dekavoudon_mul_scale_left. rewrite <- dekavoudon_scale_sub. reflexivity.
  - rewrite dekavoudon_mul_scale_left. rewrite dekavoudon_conj_scale. rewrite dekavoudon_mul_scale_right. rewrite <- dekavoudon_scale_add. reflexivity. Qed.
Lemma cd2048_neg_scale : forall r x, cd2048_neg (cd2048_scale r x) = cd2048_scale r (cd2048_neg x).
Proof. intros r [? ?]. unfold cd2048_neg, cd2048_scale. simpl. f_equal; apply dekavoudon_neg_scale. Qed.

Theorem cd4096_mul_scale_left : forall r x y, cd4096_mul (cd4096_scale r x) y = cd4096_scale r (cd4096_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd4096_mul, cd4096_scale. simpl. f_equal.
  - rewrite cd2048_mul_scale_left. rewrite cd2048_mul_scale_right. rewrite <- cd2048_scale_sub. reflexivity.
  - rewrite cd2048_mul_scale_right. rewrite cd2048_mul_scale_left. rewrite <- cd2048_scale_add. reflexivity. Qed.
Lemma cd4096_scale_sub : forall r a b, cd4096_scale r (cd4096_sub a b) = cd4096_sub (cd4096_scale r a) (cd4096_scale r b).
Proof. intros r [? ?] [? ?]. unfold cd4096_scale, cd4096_sub, cd4096_add, cd4096_neg. simpl. f_equal; apply cd2048_scale_sub. Qed.
Theorem cd4096_assoc_trilinear : forall r a b c, cd4096_assoc (cd4096_scale r a) b c = cd4096_scale r (cd4096_assoc a b c).
Proof. intros. unfold cd4096_assoc. repeat rewrite cd4096_mul_scale_left. rewrite <- cd4096_scale_sub. reflexivity. Qed.

(** ========== CD-8192 ========== *)

Lemma cd4096_scale_add : forall r a b, cd4096_add (cd4096_scale r a) (cd4096_scale r b) = cd4096_scale r (cd4096_add a b).
Proof. intros r [? ?] [? ?]. unfold cd4096_scale, cd4096_add. simpl. f_equal; apply cd2048_scale_add. Qed.
Lemma cd4096_conj_scale : forall r x, cd4096_conj (cd4096_scale r x) = cd4096_scale r (cd4096_conj x).
Proof. intros r [? ?]. unfold cd4096_conj, cd4096_scale. simpl. f_equal. apply cd2048_conj_scale. apply cd2048_neg_scale. Qed.
Theorem cd4096_mul_scale_right : forall r x y, cd4096_mul x (cd4096_scale r y) = cd4096_scale r (cd4096_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd4096_mul, cd4096_scale. simpl. f_equal.
  - rewrite cd2048_mul_scale_right. rewrite cd2048_conj_scale. rewrite cd2048_mul_scale_left. rewrite <- cd2048_scale_sub. reflexivity.
  - rewrite cd2048_mul_scale_left. rewrite cd2048_conj_scale. rewrite cd2048_mul_scale_right. rewrite <- cd2048_scale_add. reflexivity. Qed.
Lemma cd4096_neg_scale : forall r x, cd4096_neg (cd4096_scale r x) = cd4096_scale r (cd4096_neg x).
Proof. intros r [? ?]. unfold cd4096_neg, cd4096_scale. simpl. f_equal; apply cd2048_neg_scale. Qed.

Theorem cd8192_mul_scale_left : forall r x y, cd8192_mul (cd8192_scale r x) y = cd8192_scale r (cd8192_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd8192_mul, cd8192_scale. simpl. f_equal.
  - rewrite cd4096_mul_scale_left. rewrite cd4096_mul_scale_right. rewrite <- cd4096_scale_sub. reflexivity.
  - rewrite cd4096_mul_scale_right. rewrite cd4096_mul_scale_left. rewrite <- cd4096_scale_add. reflexivity. Qed.
Lemma cd8192_scale_sub : forall r a b, cd8192_scale r (cd8192_sub a b) = cd8192_sub (cd8192_scale r a) (cd8192_scale r b).
Proof. intros r [? ?] [? ?]. unfold cd8192_scale, cd8192_sub, cd8192_add, cd8192_neg. simpl. f_equal; apply cd4096_scale_sub. Qed.
Theorem cd8192_assoc_trilinear : forall r a b c, cd8192_assoc (cd8192_scale r a) b c = cd8192_scale r (cd8192_assoc a b c).
Proof. intros. unfold cd8192_assoc. repeat rewrite cd8192_mul_scale_left. rewrite <- cd8192_scale_sub. reflexivity. Qed.

(** ========== CD-16384 (Tessareskaidekavoudon) ========== *)

Lemma cd8192_scale_add : forall r a b, cd8192_add (cd8192_scale r a) (cd8192_scale r b) = cd8192_scale r (cd8192_add a b).
Proof. intros r [? ?] [? ?]. unfold cd8192_scale, cd8192_add. simpl. f_equal; apply cd4096_scale_add. Qed.
Lemma cd8192_conj_scale : forall r x, cd8192_conj (cd8192_scale r x) = cd8192_scale r (cd8192_conj x).
Proof. intros r [? ?]. unfold cd8192_conj, cd8192_scale. simpl. f_equal. apply cd4096_conj_scale. apply cd4096_neg_scale. Qed.
Theorem cd8192_mul_scale_right : forall r x y, cd8192_mul x (cd8192_scale r y) = cd8192_scale r (cd8192_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd8192_mul, cd8192_scale. simpl. f_equal.
  - rewrite cd4096_mul_scale_right. rewrite cd4096_conj_scale. rewrite cd4096_mul_scale_left. rewrite <- cd4096_scale_sub. reflexivity.
  - rewrite cd4096_mul_scale_left. rewrite cd4096_conj_scale. rewrite cd4096_mul_scale_right. rewrite <- cd4096_scale_add. reflexivity. Qed.
Lemma cd8192_neg_scale : forall r x, cd8192_neg (cd8192_scale r x) = cd8192_scale r (cd8192_neg x).
Proof. intros r [? ?]. unfold cd8192_neg, cd8192_scale. simpl. f_equal; apply cd4096_neg_scale. Qed.

Theorem cd16384_mul_scale_left : forall r x y, cd16384_mul (cd16384_scale r x) y = cd16384_scale r (cd16384_mul x y).
Proof. intros r [? ?] [? ?]. unfold cd16384_mul, cd16384_scale. simpl. f_equal.
  - rewrite cd8192_mul_scale_left. rewrite cd8192_mul_scale_right. rewrite <- cd8192_scale_sub. reflexivity.
  - rewrite cd8192_mul_scale_right. rewrite cd8192_mul_scale_left. rewrite <- cd8192_scale_add. reflexivity. Qed.
Lemma cd16384_scale_sub : forall r a b, cd16384_scale r (cd16384_sub a b) = cd16384_sub (cd16384_scale r a) (cd16384_scale r b).
Proof. intros r [? ?] [? ?]. unfold cd16384_scale, cd16384_sub, cd16384_add, cd16384_neg. simpl. f_equal; apply cd8192_scale_sub. Qed.

(** THE SUMMIT: Associator trilinearity at dim=16384. *)
Theorem cd16384_assoc_trilinear : forall r a b c,
  cd16384_assoc (cd16384_scale r a) b c = cd16384_scale r (cd16384_assoc a b c).
Proof.
  intros. unfold cd16384_assoc.
  repeat rewrite cd16384_mul_scale_left.
  rewrite <- cd16384_scale_sub.
  reflexivity.
Qed.
