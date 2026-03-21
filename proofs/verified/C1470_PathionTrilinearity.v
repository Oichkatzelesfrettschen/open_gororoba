(** * C-1470: Pathion (dim=32) Associator Trilinearity via Tower Lift.

    STRICT ABSTRACTION BOUNDARY: each lemma calls ONLY the previous
    level's lemmas. Never destruct below one CD layer.

    sed_scale_add (new, destruct to quat only inside sed)
    -> pathion_mul_scale_left (destruct to CDSed only, rewrite with sed lemmas)
    -> pathion_scale_sub (apply sed_scale_sub, NO destruct)
    -> pathion_assoc_trilinear_scale_1 (4 rewrites + reflexivity)

    Mirrors: crates/gororoba_algebra/src/construction/cd_tower.rs *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
                                 OctonionNorm CDLinearLemmas Pathion.
From OpenGororobaVerified Require Import C1455_AssociatorTrilinear.
Open Scope R_scope.

(** ========== SEDENION-LEVEL: sed_scale_add ========== *)

(** sed_scale distributes over sed_add.
    This is the LAST sedenion-level arithmetic lemma needed.
    Proof: destruct to quat level (the ONLY place ring is allowed). *)
Lemma sed_scale_add : forall r a b,
  sed_add (sed_scale r a) (sed_scale r b) = sed_scale r (sed_add a b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold sed_scale, sed_add.
  simpl sed_lo; simpl sed_hi.
  f_equal;
    destruct alo as [a1 a2], blo as [b1 b2];
    destruct ahi as [a3 a4], bhi as [b3 b4];
    destruct a1, a2, b1, b2, a3, a4, b3, b4;
    unfold oct_scale, oct_add, quat_scale, quat_add;
    simpl; f_equal; f_equal; ring.
Qed.

(** ========== PATHION-LEVEL: PURE TOWER REWRITES ========== *)

(** pathion_mul_scale_left: destruct to CDSed halves ONLY.
    Rewrite with sed_mul_scale_left/right and sed_scale_sub/add. *)
Theorem pathion_mul_scale_left : forall r x y,
  pathion_mul (pathion_scale r x) y = pathion_scale r (pathion_mul x y).
Proof.
  intros r [xa xb] [ya yb].
  unfold pathion_mul, pathion_scale.
  simpl pathion_lo; simpl pathion_hi.
  f_equal.
  - rewrite sed_mul_scale_left.
    rewrite sed_mul_scale_right.
    rewrite <- sed_scale_sub.
    reflexivity.
  - rewrite sed_mul_scale_right.
    rewrite sed_mul_scale_left.
    rewrite <- sed_scale_add.
    reflexivity.
Qed.

(** pathion_scale_sub: NO destruct at all. Apply sed lemmas directly. *)
Lemma pathion_scale_sub : forall r a b,
  pathion_scale r (pathion_sub a b) = pathion_sub (pathion_scale r a) (pathion_scale r b).
Proof.
  intros r [alo ahi] [blo bhi].
  unfold pathion_scale, pathion_sub, pathion_add, pathion_neg.
  simpl pathion_lo; simpl pathion_hi.
  f_equal; apply sed_scale_sub.
Qed.

(** Pathion associator. *)
Definition pathion_assoc (a b c : CDPathion) : CDPathion :=
  pathion_sub (pathion_mul (pathion_mul a b) c)
              (pathion_mul a (pathion_mul b c)).

(** MAIN THEOREM: dim=32 trilinearity. 4-line rewrite proof.
    IDENTICAL structure to C1455 (dim=16). *)
Theorem pathion_assoc_trilinear_scale_1 : forall r a b c,
  pathion_assoc (pathion_scale r a) b c = pathion_scale r (pathion_assoc a b c).
Proof.
  intros r a b c.
  unfold pathion_assoc.
  rewrite pathion_mul_scale_left.
  rewrite pathion_mul_scale_left.
  rewrite pathion_mul_scale_left.
  rewrite <- pathion_scale_sub.
  reflexivity.
Qed.
