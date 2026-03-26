(** * C1546_MorEigenIFF: Moreno (1997) Theorem 2.9 -- abstract iff core.

    Moreno's Theorem 2.9 says that for alternative unit-norm trace-zero
    elements a,b in A_n, the pair (a,b) in A_(n+1) is a zero divisor iff
    -2 is an eigenvalue of L_(a+b)^2.

    The concrete witness lane lives in C1546_MorEigenZD.v.  This file
    formalizes the abstract algebraic core of the iff proof in the repo's
    staged style:

    - a Moreno-style witness equation package implies the eigenvalue -2
    - conversely, an eigenvector together with Moreno's purity /
      anti-commutation side conditions yields an explicit zero-divisor witness

    The remaining open bridge is to discharge these side conditions from the
    full Cayley-Dickson hypotheses for arbitrary alternative elements. *)

From Stdlib Require Import Reals Lia Lra.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  CDConjAntimorph CDLinearLemmas CDNegLemmas.

Open Scope R_scope.

Section Moreno29Abstract.
  Variables a b : CDOct.

  Hypothesis Ha_square_all : forall z : CDOct,
    oct_mul a (oct_mul a z) = oct_neg z.
  Hypothesis Hb_square_all : forall z : CDOct,
    oct_mul b (oct_mul b z) = oct_neg z.

  Definition moreno29_c : CDOct := oct_add a b.

  Definition moreno29_partner (y : CDOct) : CDOct :=
    oct_neg (oct_mul a (oct_mul b y)).

  Definition moreno29_witness_eqns (x y : CDOct) : Prop :=
    y <> oct_zero /\
    oct_mul b (oct_mul a y) = x /\
    oct_mul a (oct_mul b y) = oct_neg x /\
    sed_mul (mkSed a b) (mkSed x y) = sed_zero.

  Lemma moreno29_oct_add_assoc : forall u v w : CDOct,
    oct_add (oct_add u v) w = oct_add u (oct_add v w).
  Proof.
    intros [ulo uhi] [vlo vhi] [wlo whi].
    unfold oct_add.
    simpl.
    f_equal; destruct ulo, uhi, vlo, vhi, wlo, whi;
    unfold quat_add; simpl; f_equal; ring.
  Qed.

  Lemma moreno29_oct_scale_neg_two : forall y : CDOct,
    oct_scale (-2) y = oct_add (oct_neg y) (oct_neg y).
  Proof.
    intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
    unfold oct_scale, oct_add, oct_neg, quat_scale, quat_add, quat_neg.
    simpl.
    apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
  Qed.

  Lemma moreno29_oct_conj_neg : forall x : CDOct,
    oct_conj (oct_neg x) = oct_neg (oct_conj x).
  Proof.
    intros [lo hi].
    unfold oct_conj, oct_neg.
    simpl.
    f_equal; destruct lo, hi; unfold quat_conj, quat_neg; simpl; f_equal; ring.
  Qed.

  Lemma moreno29_sed_mul_pair_form : forall c d : CDOct,
    sed_mul (mkSed a b) (mkSed c d) =
      mkSed
        (oct_add (oct_mul a c) (oct_neg (oct_mul (oct_conj d) b)))
        (oct_add (oct_mul d a) (oct_mul b (oct_conj c))).
  Proof.
    intros c d.
    destruct a as [alo ahi], b as [blo bhi], c as [clo chi], d as [dlo dhi].
    unfold sed_mul, oct_add, oct_neg, oct_conj.
    simpl.
    reflexivity.
  Qed.

  Theorem moreno29_witness_implies_eigen : forall x y : CDOct,
    moreno29_witness_eqns x y ->
    oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y.
  Proof.
    intros x y [_ [Hbay [Haby _]]].
    unfold moreno29_c.
    rewrite oct_mul_add_left.
    rewrite oct_mul_add_left.
    repeat rewrite oct_mul_add_right.
    rewrite Ha_square_all.
    rewrite Haby.
    rewrite Hbay.
    rewrite Hb_square_all.
    rewrite moreno29_oct_scale_neg_two.
    rewrite <- moreno29_oct_add_assoc
      with (u := oct_add (oct_neg y) (oct_neg x)) (v := x) (w := oct_neg y).
    rewrite moreno29_oct_add_assoc
      with (u := oct_neg y) (v := oct_neg x) (w := x).
    rewrite oct_add_comm with (a := oct_neg x) (b := x).
    rewrite oct_add_neg_cancel.
    rewrite oct_add_zero_right.
    reflexivity.
  Qed.

  Theorem moreno29_eigen_cross_relation : forall y : CDOct,
    oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y ->
    oct_mul a (oct_mul b y) = oct_neg (oct_mul b (oct_mul a y)).
  Proof.
    intros y Heig.
    unfold moreno29_c in Heig.
    rewrite oct_mul_add_left in Heig.
    rewrite oct_mul_add_left in Heig.
    repeat rewrite oct_mul_add_right in Heig.
    rewrite Ha_square_all in Heig.
    rewrite Hb_square_all in Heig.
    remember (oct_mul a (oct_mul b y)) as p.
    remember (oct_mul b (oct_mul a y)) as q.
    destruct y as [[y1 y2 y3 y4] [y5 y6 y7 y8]].
    destruct p as [[p1 p2 p3 p4] [p5 p6 p7 p8]].
    destruct q as [[q1 q2 q3 q4] [q5 q6 q7 q8]].
    cbv [oct_add oct_neg oct_scale quat_add quat_neg quat_scale qa qb qc qd] in Heig.
    inversion Heig; subst.
    change
      (oct_neg
         {| oct_lo := {| qa := q1; qb := q2; qc := q3; qd := q4 |};
            oct_hi := {| qa := q5; qb := q6; qc := q7; qd := q8 |} |})
      with
      (mkOct (mkQuat (- q1) (- q2) (- q3) (- q4))
             (mkQuat (- q5) (- q6) (- q7) (- q8))).
    apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
  Qed.

  Theorem moreno29_eigen_implies_witness : forall y : CDOct,
    y <> oct_zero ->
    oct_conj y = oct_neg y ->
    oct_mul y a = oct_neg (oct_mul a y) ->
    oct_mul y b = oct_neg (oct_mul b y) ->
    oct_conj (oct_mul a (oct_mul b y)) = oct_neg (oct_mul a (oct_mul b y)) ->
    oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y ->
    moreno29_witness_eqns (moreno29_partner y) y.
  Proof.
    intros y Hy Hy_pure Hya Hyb Haby_pure Heig.
    unfold moreno29_witness_eqns, moreno29_partner.
    split.
    - exact Hy.
    - split.
      + rewrite moreno29_eigen_cross_relation by exact Heig.
        rewrite oct_neg_neg.
        reflexivity.
      + split.
        * rewrite oct_neg_neg.
          reflexivity.
        * rewrite moreno29_sed_mul_pair_form.
          apply (f_equal2 mkSed).
          -- rewrite oct_neg_mul_right.
             rewrite Ha_square_all.
             rewrite Hy_pure.
             rewrite oct_neg_mul_left.
             rewrite oct_neg_neg.
             rewrite Hyb.
             repeat rewrite oct_neg_neg.
             apply oct_add_neg_cancel.
          -- rewrite moreno29_oct_conj_neg.
             rewrite Haby_pure.
             rewrite oct_neg_neg.
             rewrite moreno29_eigen_cross_relation by exact Heig.
             rewrite oct_neg_mul_right.
             rewrite Hb_square_all.
             rewrite oct_neg_neg.
             rewrite Hya.
             repeat rewrite oct_neg_neg.
             rewrite oct_add_comm
               with (a := oct_neg (oct_mul a y)) (b := oct_mul a y).
             apply oct_add_neg_cancel.
  Qed.

  Lemma moreno29_nested_product_pure : forall y : CDOct,
    oct_conj a = oct_neg a ->
    oct_conj b = oct_neg b ->
    oct_conj y = oct_neg y ->
    oct_mul y b = oct_neg (oct_mul b y) ->
    oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y)) ->
    oct_conj (oct_mul a (oct_mul b y)) = oct_neg (oct_mul a (oct_mul b y)).
  Proof.
    intros y Ha_pure Hb_pure Hy_pure Hyb Hbya.
    rewrite oct_conj_antimorphism.
    rewrite oct_conj_antimorphism.
    rewrite Hy_pure.
    rewrite Hb_pure.
    rewrite Ha_pure.
    rewrite oct_neg_mul_right.
    rewrite oct_neg_mul_left.
    rewrite oct_neg_mul_right.
    rewrite oct_neg_neg.
    rewrite Hyb.
    rewrite oct_neg_mul_left.
    rewrite oct_neg_neg.
    exact Hbya.
  Qed.

  Corollary moreno29_abstract_iff : forall y : CDOct,
    oct_conj y = oct_neg y ->
    oct_mul y a = oct_neg (oct_mul a y) ->
    oct_mul y b = oct_neg (oct_mul b y) ->
    oct_conj (oct_mul a (oct_mul b y)) = oct_neg (oct_mul a (oct_mul b y)) ->
    (moreno29_witness_eqns (moreno29_partner y) y <->
     y <> oct_zero /\
     oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y).
  Proof.
    intros y Hy_pure Hya Hyb Haby_pure.
    split.
    - intro Hw.
      split.
      + exact (proj1 Hw).
      + exact (moreno29_witness_implies_eigen _ _ Hw).
    - intros [Hy Heig].
      exact (moreno29_eigen_implies_witness y Hy Hy_pure Hya Hyb Haby_pure Heig).
  Qed.

  Corollary moreno29_abstract_iff_anticomm : forall y : CDOct,
    oct_conj a = oct_neg a ->
    oct_conj b = oct_neg b ->
    oct_conj y = oct_neg y ->
    oct_mul y a = oct_neg (oct_mul a y) ->
    oct_mul y b = oct_neg (oct_mul b y) ->
    oct_mul (oct_mul b y) a = oct_neg (oct_mul a (oct_mul b y)) ->
    (moreno29_witness_eqns (moreno29_partner y) y <->
     y <> oct_zero /\
     oct_mul moreno29_c (oct_mul moreno29_c y) = oct_scale (-2) y).
  Proof.
    intros y Ha_pure Hb_pure Hy_pure Hya Hyb Hbya.
    apply moreno29_abstract_iff; try assumption.
    exact (moreno29_nested_product_pure y Ha_pure Hb_pure Hy_pure Hyb Hbya).
  Qed.
End Moreno29Abstract.
