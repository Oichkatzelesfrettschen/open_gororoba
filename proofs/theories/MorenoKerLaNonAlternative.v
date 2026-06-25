(** * MorenoKerLaNonAlternative: Ker(L_a) for a NON-ALTERNATIVE doubly-pure
      sedenion representative.

    Moreno 1997 (arXiv:q-alg/9710013v1) Thm 1.15-1.16 / Cor 1.17 describe the
    structure of Ker(L_a) for a doubly-pure unit-norm a in A_n, n >= 4: the
    quaternion subalgebra H_a acts on Ker(L_a) by right multiplication, so
    dim_R Ker(L_a) = 0 mod 4.  That content is VACUOUS for an alternative a
    (e.g. any basis element, or any octonion): there L_a^2 = -|a|^2 I, so L_a
    is injective and Ker(L_a) = {0}.

    The Moreno-Froloff zero divisor a = e3 + e10 is doubly pure and
    NON-alternative (its left-multiplication operator is not injective), but it
    is not normalized: OctonionNorm.sed_zd_a_norm gives norm squared 2.  This
    file proves facts for that concrete representative only:

      (1) a is purely imaginary (conj a = -a) and doubly pure: sed_lo is
          octonion e3, and sed_hi is octonion e2, i.e. sedenion e10;
      (2) Ker(L_a) is nontrivial -- sed_zd_b = e6 - e15 is a nonzero element
          (so L_a is not injective, i.e. a is non-alternative);
      (3) Ker(L_a) contains FOUR linearly independent elements, so
          dim_R Ker(L_a) >= 4 -- the concrete lower bound matching Moreno's
          mod-4 structure.

    The four annihilators are the boxkite_7 partner assessors of (3,10) with
    the sign that annihilates a: (4,13)+, (5,12)-, (6,15)-, (7,14)+.  The
    fifth partner (2,11) is the strut-opposite and annihilates with neither
    sign in this direct computation.  Their basis supports are disjoint, so
    independence is a single component extraction.  This file does not prove an
    upper bound on Ker(L_a).

    This closes, for a concrete non-alternative a, the H_a-orbit lower-bound
    step that the C-1627 abstract orbit lane leaves as a caller obligation. *)

From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm C1538_MorZDSymmetry.
Open Scope R_scope.

(** The shared reduction: unfold the sedenion product on concrete basis
    literals down to raw reals, then discharge by ring.  Same pattern as
    OctonionNorm.sed_zd_product_zero. *)
Ltac sed_kill :=
  cbv [sed_mul sed_zd_a sed_e oct_e sed_add sed_sub sed_neg
       oct_add oct_neg sed_zero oct_mul oct_conj oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  f_equal; f_equal; f_equal; ring.

(** ** (1) a = e3 + e10 is purely imaginary, and doubly pure. *)
Lemma sed_zd_a_pure : sed_conj sed_zd_a = sed_neg sed_zd_a.
Proof.
  cbv [sed_conj sed_neg sed_zd_a oct_conj oct_neg
       quat_conj quat_neg quat_zero sed_lo sed_hi oct_lo oct_hi
       qa qb qc qd].
  repeat f_equal; ring.
Qed.

(** Doubly pure: both octonion halves are purely imaginary octonions.  The
    high half is octonion e2, corresponding to sedenion e10 under CD doubling.
    This is Moreno's stronger condition on a, strictly above the
    purely-imaginary condition proved above. *)
Lemma sed_zd_a_doubly_pure :
  oct_conj (sed_lo sed_zd_a) = oct_neg (sed_lo sed_zd_a) /\
  oct_conj (sed_hi sed_zd_a) = oct_neg (sed_hi sed_zd_a).
Proof.
  split;
    cbv [sed_zd_a sed_lo sed_hi oct_conj oct_neg oct_lo oct_hi
         quat_conj quat_neg quat_zero qa qb qc qd];
    repeat f_equal; ring.
Qed.

(** ** (2) Ker(L_a) is nontrivial: sed_zd_b is a nonzero annihilator. *)

Theorem ker_La_nontrivial :
  exists y, y <> sed_zero /\ sed_mul sed_zd_a y = sed_zero.
Proof.
  exists sed_zd_b. split.
  - exact C1538_MorZDSymmetry.sed_zd_b_nonzero.
  - exact sed_zd_product_zero.
Qed.

(** Non-alternativity, operationally: L_a is not injective.  An alternative
    element a satisfies L_a^2 = -|a|^2 I, so for a <> 0 its L_a is injective;
    a nonzero kernel therefore certifies a is non-alternative. *)
Theorem sed_zd_a_L_not_injective :
  exists y, y <> sed_zero /\ sed_mul sed_zd_a y = sed_mul sed_zd_a sed_zero.
Proof.
  exists sed_zd_b. split.
  - exact C1538_MorZDSymmetry.sed_zd_b_nonzero.
  - rewrite sed_zd_product_zero.
    cbv [sed_mul sed_zd_a sed_zero oct_mul oct_conj oct_zero
         quat_mul quat_add quat_neg quat_conj quat_zero quat_one
         sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
    f_equal; f_equal; f_equal; ring.
Qed.

(** ** (3) Four linearly independent annihilators -> dim Ker(L_a) >= 4. *)

(** boxkite_7 partner assessors of (3,10), each with the annihilating sign. *)
Definition ann1 : CDSed := sed_add (sed_e 4) (sed_e 13).   (* e4 + e13 *)
Definition ann2 : CDSed := sed_sub (sed_e 5) (sed_e 12).   (* e5 - e12 *)
Definition ann3 : CDSed := sed_sub (sed_e 6) (sed_e 15).   (* e6 - e15 = sed_zd_b *)
Definition ann4 : CDSed := sed_add (sed_e 7) (sed_e 14).   (* e7 + e14 *)

Lemma ann1_annihilated : sed_mul sed_zd_a ann1 = sed_zero.
Proof. unfold ann1. sed_kill. Qed.
Lemma ann2_annihilated : sed_mul sed_zd_a ann2 = sed_zero.
Proof. unfold ann2. sed_kill. Qed.
Lemma ann3_annihilated : sed_mul sed_zd_a ann3 = sed_zero.
Proof. unfold ann3. sed_kill. Qed.
Lemma ann4_annihilated : sed_mul sed_zd_a ann4 = sed_zero.
Proof. unfold ann4. sed_kill. Qed.

(** Linear independence of four sedenions: any real combination that
    vanishes has all-zero coefficients. *)
Definition lin_indep_4 (y1 y2 y3 y4 : CDSed) : Prop :=
  forall c1 c2 c3 c4 : R,
    sed_add (sed_scale c1 y1)
      (sed_add (sed_scale c2 y2)
        (sed_add (sed_scale c3 y3) (sed_scale c4 y4))) = sed_zero ->
    c1 = 0 /\ c2 = 0 /\ c3 = 0 /\ c4 = 0.

(** The low parts e4,e5,e6,e7 of ann1..ann4 occupy the qa,qb,qc,qd slots of
    sed_lo.oct_hi; the high parts (e12..e15) live in sed_hi and do not touch
    those slots.  So each coefficient reads off one quaternion component. *)
Theorem ann_lin_indep : lin_indep_4 ann1 ann2 ann3 ann4.
Proof.
  intros c1 c2 c3 c4 H.
  assert (Ha := f_equal (fun s => qa (oct_hi (sed_lo s))) H).
  assert (Hb := f_equal (fun s => qb (oct_hi (sed_lo s))) H).
  assert (Hc := f_equal (fun s => qc (oct_hi (sed_lo s))) H).
  assert (Hd := f_equal (fun s => qd (oct_hi (sed_lo s))) H).
  cbv [ann1 ann2 ann3 ann4 sed_add sed_sub sed_neg sed_scale sed_e oct_e
       oct_add oct_neg oct_scale quat_scale quat_add quat_neg
       sed_zero oct_zero quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd] in Ha, Hb, Hc, Hd.
  repeat split; lra.
Qed.

(** Headline: Ker(L_a) for the non-alternative a = e3 + e10 contains four
    linearly independent vectors -- dim_R Ker(L_a) >= 4. *)
Theorem ker_La_dim_ge_4 :
  exists y1 y2 y3 y4 : CDSed,
    sed_mul sed_zd_a y1 = sed_zero /\
    sed_mul sed_zd_a y2 = sed_zero /\
    sed_mul sed_zd_a y3 = sed_zero /\
    sed_mul sed_zd_a y4 = sed_zero /\
    lin_indep_4 y1 y2 y3 y4.
Proof.
  exists ann1, ann2, ann3, ann4.
  split; [exact ann1_annihilated|].
  split; [exact ann2_annihilated|].
  split; [exact ann3_annihilated|].
  split; [exact ann4_annihilated|].
  exact ann_lin_indep.
Qed.
