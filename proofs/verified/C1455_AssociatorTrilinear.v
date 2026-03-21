(** * C-1455: Sedenion Associator Trilinearity -- Scalar in First Argument.

    [r*a, b, c] = r * [a, b, c]

    Pure rewriting proof using CDLinearLemmas tower + sed_scale_sub.
    NO dest_sed, NO ring on 48+ variables, NO simpl explosion.

    Mirrors: crates/algebra_experimental/src/lepton_mass_hierarchy.rs *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
                                 OctonionNorm CDLinearLemmas SedenionAssociator.
Open Scope R_scope.

(** sed_scale distributes over sed_sub: r*(a - b) = r*a - r*b. *)
Lemma sed_scale_sub : forall r a b,
  sed_scale r (sed_sub a b) = sed_sub (sed_scale r a) (sed_scale r b).
Proof.
  intros r a b.
  destruct a as [alo ahi], b as [blo bhi].
  unfold sed_scale, sed_sub, sed_add, sed_neg.
  simpl sed_lo; simpl sed_hi.
  f_equal;
    destruct alo as [allo alhi], blo as [bllo blhi];
    destruct ahi as [ahlo ahhi], bhi as [bhlo bhhi];
    unfold oct_scale, oct_add, oct_neg;
    simpl oct_lo; simpl oct_hi;
    f_equal;
    destruct allo, alhi, bllo, blhi, ahlo, ahhi, bhlo, bhhi;
    unfold quat_scale, quat_add, quat_neg; simpl;
    f_equal; ring.
Qed.

(** Main theorem: associator scalar-homogeneity in first argument. *)
Theorem sed_assoc_trilinear_scale_1 : forall r a b c,
  sed_assoc (sed_scale r a) b c = sed_scale r (sed_assoc a b c).
Proof.
  intros r a b c.
  unfold sed_assoc.
  rewrite sed_mul_scale_left.
  rewrite sed_mul_scale_left.
  rewrite sed_mul_scale_left.
  rewrite <- sed_scale_sub.
  reflexivity.
Qed.
