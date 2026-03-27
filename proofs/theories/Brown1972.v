(** * Brown1972: Paper-scoped Rocq index for Brown (1972).

    Source:
      R.B. Brown, "Structure of zero divisors in higher dimensional real
      Cayley-Dickson algebras" (Brown 1972 lane in the repo's paper corpus).

    This file is the Rocq-facing paper lane for Brown (1972). It exposes the
    current Brown-specific formalization surface under one import, while also
    recording which dissertation chapters already have direct Rocq landings and
    which still live only in the Rust paper crate.

    Source-driven inventory note:
    - the on-disk Brown source packet shows a nine-chapter dissertation plus
      three appendices
    - Chapters I-II are contextual source chapters (introduction and review of
      literature) with no numbered theorem payload in the dissertation text
    - Chapter III is foundational but not the whole Brown picture
    - the theorem-dense Brown backlog is really spread across Chapters IV-VII,
      with Appendix C still relevant for the historical computation lane

    Chapter/page surfacing status:
    - Chapter III, pp. 15-16, Theorem 3.9 and Lemma 3.10:
      abstract Rocq norm/involution surface plus standard-octonion witnesses
      and direct standard-sedenion witnesses landed here; Rust lane
      `crates/brown_1972/src/norm_symmetry.rs` remains the computational
      mirror and still carries the broader generalized-norm exploration beyond
      these concrete 8D/16D witnesses.
    - Chapter IV, pp. 20-22, Theorems 4.2-4.3 and Corollary 4.4:
      source-driven standard-tower witnesses for 4.2, 4.3, and 4.4 are now
      landed here.
    - Chapter V, pp. 27-30, Theorems 5.11-5.17:
      Brown-numbered quaternion witness surface landed here; Rust lane
      `crates/brown_1972/src/exponent_properties.rs` remains the broader
      computational mirror and still carries the non-quaternion/generalized
      follow-on exploration.
    - Chapter VI, pp. 30-37, Theorems 6.1-6.11:
      Rust lane `crates/brown_1972/src/basis_element_properties.rs`; direct
      Rocq chapter surface is still open.
    - Chapter VII, pp. 45-56, Theorems 7.3-7.18:
      direct Rocq landing via `ZD_Criterion.v`, `C1538_MorZDSymmetry.v`, and
      `BrownAssessorEquivalence.v`.
    - Appendix C, pp. 78-89:
      Rust lane `crates/brown_1972/src/pl1_emulator.rs`; Rocq extraction bridge
      is still open.

    Current Brown Rocq companion map:
    - ZD_Criterion.v             : Brown Theorem 7.15 concrete criterion lane
    - C1538_MorZDSymmetry.v      : Brown 7.3-style symmetry witness at dim 16
    - BrownAssessorEquivalence.v : Brown to de Marrais assessor / box-kite bridge

    Brown-adjacent support reused by these lanes, but not themselves Brown 1972
    paper surfaces, includes `CDPowerAssociative.v` and later Moreno bridges.

    Remaining Brown-specific Rocq backlog:
    - broader Chapter V exponent surface beyond the new quaternion witness lane
    - broader generalized-norm Chapter III lane beyond the concrete octonion
      and sedenion witnesses
    - Brown-numbered Chapter VI basis-element theorem lanes
    - remaining Chapter VII numbering gaps plus Appendix C extraction bridge in Rocq

    The executable Rust companion for this paper is `crates/brown_1972/`. *)

From Stdlib Require Import List Reals ZArith Lia Lra.
Import ListNotations.
Open Scope R_scope.

From OpenGororoba Require Import
  ZDGraph
  Sedenion
  CayleyDicksonAlgebra
  OctonionNorm
  CDAssociator
  CDPowerAssociative
  CDConjAntimorph
  CDLinearLemmas
  CDNegLemmas
  CDInverse
  SedenionAssociator
  DicksonCDProcess
  SedenionAlternativityFails.
From OpenGororoba Require Export
  C1538_MorZDSymmetry
  ZD_Criterion
  BrownAssessorEquivalence.

(** Brown Chapter III is a norm/involution lane. The current Rocq landing keeps
    the theorem surface abstract and then instantiates it concretely on the
    standard octonions, reusing existing norm and conjugation infrastructure
    rather than expanding the full sedenion tower. *)

Module Type BrownChapterIIINormAlg.
  Parameter A : Type.
  Parameter add : A -> A -> A.
  Parameter sub : A -> A -> A.
  Parameter mul : A -> A -> A.
  Parameter conj : A -> A.
  Parameter norm_sq : A -> R.

  Axiom brown_norm_conj_preserved : forall x,
    norm_sq (conj x) = norm_sq x.
  Axiom brown_norm_mul : forall x y,
    norm_sq (mul x y) = (norm_sq x * norm_sq y)%R.
  Axiom brown_polarization_identity : forall x y,
    (norm_sq (add x y) + norm_sq (sub x y))%R =
    (2 * (norm_sq x + norm_sq y))%R.
End BrownChapterIIINormAlg.

Module BrownChapterIII (Alg : BrownChapterIIINormAlg).
  Import Alg.

  Theorem brown1972_theorem_3_9_i : forall x y : A,
    norm_sq (mul x (conj y)) = norm_sq (mul x y).
  Proof.
    intros x y.
    repeat rewrite brown_norm_mul.
    rewrite brown_norm_conj_preserved.
    ring.
  Qed.

  Theorem brown1972_theorem_3_9_ii : forall x y : A,
    norm_sq (mul (conj x) y) = norm_sq (mul x y).
  Proof.
    intros x y.
    repeat rewrite brown_norm_mul.
    rewrite brown_norm_conj_preserved.
    ring.
  Qed.

  Theorem brown1972_theorem_3_9_iii : forall x y : A,
    norm_sq (mul x y) = norm_sq (mul y x).
  Proof.
    intros x y.
    repeat rewrite brown_norm_mul.
    ring.
  Qed.

  Theorem brown1972_lemma_3_10 : forall x y : A,
    (norm_sq (add x y) + norm_sq (sub x y))%R =
    (2 * (norm_sq x + norm_sq y))%R.
  Proof.
    exact brown_polarization_identity.
  Qed.
End BrownChapterIII.

Lemma brown1972_octonion_norm_conj_preserved : forall x : CDOct,
  oct_norm_sq (oct_conj x) = oct_norm_sq x.
Proof.
  intros [[a b c d] [e f g h]].
  unfold oct_norm_sq, oct_conj, quat_norm_sq, quat_conj, quat_neg.
  simpl. ring.
Qed.

Theorem brown1972_octonion_lemma_3_10 : forall x y : CDOct,
  (oct_norm_sq (oct_add x y) + oct_norm_sq (oct_sub x y))%R =
  (2 * (oct_norm_sq x + oct_norm_sq y))%R.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  unfold oct_norm_sq, oct_add, oct_sub, oct_neg, quat_norm_sq, quat_add, quat_neg.
  simpl. ring.
Qed.

Module Brown1972OctonionNormAlg <: BrownChapterIIINormAlg.
  Definition A := CDOct.
  Definition add := oct_add.
  Definition sub := oct_sub.
  Definition mul := oct_mul.
  Definition conj := oct_conj.
  Definition norm_sq := oct_norm_sq.

  Theorem brown_norm_conj_preserved : forall x,
    norm_sq (conj x) = norm_sq x.
  Proof.
    exact brown1972_octonion_norm_conj_preserved.
  Qed.

  Theorem brown_norm_mul : forall x y,
    norm_sq (mul x y) = (norm_sq x * norm_sq y)%R.
  Proof.
    exact oct_norm_mul.
  Qed.

  Theorem brown_polarization_identity : forall x y,
    (norm_sq (add x y) + norm_sq (sub x y))%R =
    (2 * (norm_sq x + norm_sq y))%R.
  Proof.
    exact brown1972_octonion_lemma_3_10.
  Qed.
End Brown1972OctonionNormAlg.

Module Brown1972OctonionChapterIII := BrownChapterIII(Brown1972OctonionNormAlg).

(** Chapter III surface currently landed concretely at the standard octonion
    norm layer, matching the repo's existing norm/involution theorems. *)
Record Brown1972ChapterIIISurface := {
  brown1972_ch3_t39_i :
    forall x y : CDOct, oct_norm_sq (oct_mul x (oct_conj y)) = oct_norm_sq (oct_mul x y);
  brown1972_ch3_t39_ii :
    forall x y : CDOct, oct_norm_sq (oct_mul (oct_conj x) y) = oct_norm_sq (oct_mul x y);
  brown1972_ch3_t39_iii :
    forall x y : CDOct, oct_norm_sq (oct_mul x y) = oct_norm_sq (oct_mul y x);
  brown1972_ch3_l310 :
    forall x y : CDOct,
      (oct_norm_sq (oct_add x y) + oct_norm_sq (oct_sub x y))%R =
      (2 * (oct_norm_sq x + oct_norm_sq y))%R
}.

Definition brown1972_octonion_chapter_iii_surface :
  Brown1972ChapterIIISurface.
Proof.
  refine {| brown1972_ch3_t39_i := Brown1972OctonionChapterIII.brown1972_theorem_3_9_i;
            brown1972_ch3_t39_ii := Brown1972OctonionChapterIII.brown1972_theorem_3_9_ii;
            brown1972_ch3_t39_iii := Brown1972OctonionChapterIII.brown1972_theorem_3_9_iii;
            brown1972_ch3_l310 := Brown1972OctonionChapterIII.brown1972_lemma_3_10 |}.
Defined.

(** Brown Chapter III, standard octonion Rocq witnesses. *)
Theorem brown1972_theorem_3_9_i_octonion : forall x y : CDOct,
  oct_norm_sq (oct_mul x (oct_conj y)) = oct_norm_sq (oct_mul x y).
Proof.
  exact Brown1972OctonionChapterIII.brown1972_theorem_3_9_i.
Qed.

Theorem brown1972_theorem_3_9_ii_octonion : forall x y : CDOct,
  oct_norm_sq (oct_mul (oct_conj x) y) = oct_norm_sq (oct_mul x y).
Proof.
  exact Brown1972OctonionChapterIII.brown1972_theorem_3_9_ii.
Qed.

Theorem brown1972_theorem_3_9_iii_octonion : forall x y : CDOct,
  oct_norm_sq (oct_mul x y) = oct_norm_sq (oct_mul y x).
Proof.
  exact Brown1972OctonionChapterIII.brown1972_theorem_3_9_iii.
Qed.

(** Brown Theorem 7.3, concrete symmetry witness at the canonical sedenion pair. *)
Theorem brown1972_theorem_7_3_witness :
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_mul sed_zd_b sed_zd_a = sed_zero.
Proof.
  exact C1538_sedenion_zd_symmetry.
Qed.

(** Brown Theorem 7.15, concrete fundamental criterion witness. *)
Theorem brown1972_theorem_7_15_fundamental :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  exact zd_fundamental_major_theorem.
Qed.

(** Appendix C / assessor bridge summary currently formalized in Rocq. *)
Theorem brown1972_appendix_c_structure_summary :
  (42 * 4 = 168) /\
  (7 * 6 * 4 = 168) /\
  (6 * 4 = 24) /\
  (ZDGraph.boxkite_signatures = 15 :: 10 :: 11 :: 12 :: 13 :: 14 :: 9 :: nil).
Proof.
  exact brown_demarrais_bridge.
Qed.

Theorem brown1972_chapter_iii_octonion_summary :
  (forall x y : CDOct,
      oct_norm_sq (oct_mul x (oct_conj y)) = oct_norm_sq (oct_mul x y)) /\
  (forall x y : CDOct,
      oct_norm_sq (oct_mul (oct_conj x) y) = oct_norm_sq (oct_mul x y)) /\
  (forall x y : CDOct,
      oct_norm_sq (oct_mul x y) = oct_norm_sq (oct_mul y x)) /\
  (forall x y : CDOct,
      (oct_norm_sq (oct_add x y) + oct_norm_sq (oct_sub x y))%R =
      (2 * (oct_norm_sq x + oct_norm_sq y))%R).
Proof.
  repeat split.
  - exact brown1972_theorem_3_9_i_octonion.
  - exact brown1972_theorem_3_9_ii_octonion.
  - exact brown1972_theorem_3_9_iii_octonion.
  - exact brown1972_octonion_lemma_3_10.
Qed.

(** Brown Chapter III, sourced 16D standard-sedenion witnesses.

    The dissertation states Theorem 3.9 for flexible algebras with centered
    involution. The current abstract Rocq functor above still packages the
    stronger octonion/multiplicative-norm route. For the standard sedenions we
    instead land the Brown-numbered identities directly on coordinates, so the
    16D lane is source-driven without forcing it through the stronger abstract
    interface. *)

Theorem brown1972_lemma_3_10_sedenion : forall x y : CDSed,
  (sed_norm_sq (sed_add x y) + sed_norm_sq (sed_sub x y))%R =
  (2 * (sed_norm_sq x + sed_norm_sq y))%R.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_norm_sq sed_add sed_sub sed_neg
       oct_norm_sq oct_add oct_neg
       quat_norm_sq quat_add quat_neg
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_9_i_sedenion : forall x y : CDSed,
  sed_norm_sq (sed_mul x (sed_conj y)) = sed_norm_sq (sed_mul x y).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_norm_sq sed_mul sed_conj
       oct_norm_sq oct_mul oct_conj oct_neg
       quat_norm_sq quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_9_ii_sedenion : forall x y : CDSed,
  sed_norm_sq (sed_mul (sed_conj x) y) = sed_norm_sq (sed_mul x y).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_norm_sq sed_mul sed_conj
       oct_norm_sq oct_mul oct_conj oct_neg
       quat_norm_sq quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_9_iii_sedenion : forall x y : CDSed,
  sed_norm_sq (sed_mul x y) = sed_norm_sq (sed_mul y x).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_norm_sq sed_mul
       oct_norm_sq oct_mul oct_conj oct_neg
       quat_norm_sq quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Record Brown1972ChapterIIISedenionSurface := {
  brown1972_ch3_t39_i_sed :
    forall x y : CDSed, sed_norm_sq (sed_mul x (sed_conj y)) = sed_norm_sq (sed_mul x y);
  brown1972_ch3_t39_ii_sed :
    forall x y : CDSed, sed_norm_sq (sed_mul (sed_conj x) y) = sed_norm_sq (sed_mul x y);
  brown1972_ch3_t39_iii_sed :
    forall x y : CDSed, sed_norm_sq (sed_mul x y) = sed_norm_sq (sed_mul y x);
  brown1972_ch3_l310_sed :
    forall x y : CDSed,
      (sed_norm_sq (sed_add x y) + sed_norm_sq (sed_sub x y))%R =
      (2 * (sed_norm_sq x + sed_norm_sq y))%R
}.

Definition brown1972_sedenion_chapter_iii_surface :
  Brown1972ChapterIIISedenionSurface.
Proof.
  refine {| brown1972_ch3_t39_i_sed := brown1972_theorem_3_9_i_sedenion;
            brown1972_ch3_t39_ii_sed := brown1972_theorem_3_9_ii_sedenion;
            brown1972_ch3_t39_iii_sed := brown1972_theorem_3_9_iii_sedenion;
            brown1972_ch3_l310_sed := brown1972_lemma_3_10_sedenion |}.
Defined.

Theorem brown1972_chapter_iii_sedenion_summary :
  (forall x y : CDSed,
      sed_norm_sq (sed_mul x (sed_conj y)) = sed_norm_sq (sed_mul x y)) /\
  (forall x y : CDSed,
      sed_norm_sq (sed_mul (sed_conj x) y) = sed_norm_sq (sed_mul x y)) /\
  (forall x y : CDSed,
      sed_norm_sq (sed_mul x y) = sed_norm_sq (sed_mul y x)) /\
  (forall x y : CDSed,
      (sed_norm_sq (sed_add x y) + sed_norm_sq (sed_sub x y))%R =
      (2 * (sed_norm_sq x + sed_norm_sq y))%R).
Proof.
  repeat split.
  - exact brown1972_theorem_3_9_i_sedenion.
  - exact brown1972_theorem_3_9_ii_sedenion.
  - exact brown1972_theorem_3_9_iii_sedenion.
  - exact brown1972_lemma_3_10_sedenion.
Qed.

(** Brown Chapter IV starts the source-driven structural lane:
    4.2 proves flexibility for the tower,
    4.3 gives the one-step associator formula,
    4.4 extracts alternativity exactly when the generating algebra is
    associative. We land these here as standard-tower witnesses. *)

Theorem brown1972_theorem_4_2_quaternion : forall x y : CDQuat,
  quat_mul (quat_mul x y) x = quat_mul x (quat_mul y x).
Proof.
  exact quat_flexible.
Qed.

Theorem brown1972_theorem_4_2_octonion : forall x y : CDOct,
  oct_mul (oct_mul x y) x = oct_mul x (oct_mul y x).
Proof.
  exact oct_flexible.
Qed.

Theorem brown1972_theorem_4_3_sedenion :
  forall a1 a2 b1 b2 : CDOct,
    sed_assoc (mkSed a1 a2) (mkSed a1 a2) (mkSed b1 b2) =
    mkSed
      (oct_add (oct_assoc a1 b2 a2)
        (oct_add (oct_assoc a2 a2 b1) (oct_assoc a1 a1 b1)))
      (oct_add (oct_sub (oct_assoc a1 a1 b2) (oct_assoc a1 b1 a2))
        (oct_assoc a2 a2 b2)).
Proof.
  intros [a1lo a1hi] [a2lo a2hi] [b1lo b1hi] [b2lo b2hi].
  cbv [sed_assoc sed_sub sed_add sed_neg sed_mul
       oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       quat_assoc quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); ring.
    + apply (f_equal4 mkQuat); ring.
  - apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); ring.
    + apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_corollary_4_4_octonion_left :
  forall x y : CDOct,
    oct_assoc x x y = oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  repeat f_equal; ring.
Qed.

Theorem brown1972_corollary_4_4_octonion_right :
  forall x y : CDOct,
    oct_assoc y x x = oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  repeat f_equal; ring.
Qed.

Theorem brown1972_corollary_4_4_octonion_from_quaternion_associativity :
  (forall a b c : CDQuat, quat_assoc a b c = quat_zero) /\
  (forall x y : CDOct, oct_assoc x x y = oct_zero) /\
  (forall x y : CDOct, oct_assoc y x x = oct_zero).
Proof.
  repeat split.
  - exact quat_assoc_zero.
  - exact brown1972_corollary_4_4_octonion_left.
  - exact brown1972_corollary_4_4_octonion_right.
Qed.

Theorem brown1972_corollary_4_4_sedenion_counterexample :
  (oct_mul (oct_mul (oct_e 1) (oct_e 2)) (oct_e 4) <>
   oct_mul (oct_e 1) (oct_mul (oct_e 2) (oct_e 4))) /\
  (exists i j : nat,
      i < 16 /\ j < 16 /\ i <> j /\ i <> 0 /\ j <> 0 /\
      (sign_1_10 * sign_1_11)%Z <> (-1)%Z).
Proof.
  split.
  - exact dickson_oct_not_associative.
  - exact sedenion_not_alternative.
Qed.

Record Brown1972ChapterIVSurface := {
  brown1972_ch4_t42_quat :
    forall x y : CDQuat, quat_mul (quat_mul x y) x = quat_mul x (quat_mul y x);
  brown1972_ch4_t42_oct :
    forall x y : CDOct, oct_mul (oct_mul x y) x = oct_mul x (oct_mul y x);
  brown1972_ch4_t43_sed :
    forall a1 a2 b1 b2 : CDOct,
      sed_assoc (mkSed a1 a2) (mkSed a1 a2) (mkSed b1 b2) =
      mkSed
        (oct_add (oct_assoc a1 b2 a2)
          (oct_add (oct_assoc a2 a2 b1) (oct_assoc a1 a1 b1)))
        (oct_add (oct_sub (oct_assoc a1 a1 b2) (oct_assoc a1 b1 a2))
          (oct_assoc a2 a2 b2));
  brown1972_ch4_c44_oct_left :
    forall x y : CDOct, oct_assoc x x y = oct_zero;
  brown1972_ch4_c44_oct_right :
    forall x y : CDOct, oct_assoc y x x = oct_zero
}.

Definition brown1972_standard_tower_chapter_iv_surface :
  Brown1972ChapterIVSurface.
Proof.
  refine {| brown1972_ch4_t42_quat := brown1972_theorem_4_2_quaternion;
            brown1972_ch4_t42_oct := brown1972_theorem_4_2_octonion;
            brown1972_ch4_t43_sed := brown1972_theorem_4_3_sedenion;
            brown1972_ch4_c44_oct_left := brown1972_corollary_4_4_octonion_left;
            brown1972_ch4_c44_oct_right := brown1972_corollary_4_4_octonion_right |}.
Defined.

(** Brown Chapter V is source-driven from the dissertation's integer-power
    conventions. The full Brown statement is for all Cayley-Dickson algebras;
    the current Rocq landing here records a clean quaternion witness surface,
    built from associativity plus the explicit Brown inverse convention. *)

Definition brown1972_quat_inv (a : CDQuat) : CDQuat :=
  quat_scale (/ quat_norm_sq a) (quat_conj a).

Fixpoint brown1972_quat_nat_pow (a : CDQuat) (n : nat) : CDQuat :=
  match n with
  | O => quat_one
  | S k => quat_mul (brown1972_quat_nat_pow a k) a
  end.

Definition brown1972_quat_zpow (a : CDQuat) (n : Z) : CDQuat :=
  match n with
  | Z0 => quat_one
  | Zpos p => brown1972_quat_nat_pow a (Pos.to_nat p)
  | Zneg p => brown1972_quat_nat_pow (brown1972_quat_inv a) (Pos.to_nat p)
  end.

Lemma brown1972_quat_inv_mul_left : forall a,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_inv a) a = quat_one.
Proof.
  intros [a b c d] Hnz.
  unfold brown1972_quat_inv, quat_mul, quat_scale, quat_conj, quat_one, quat_norm_sq in *.
  simpl.
  f_equal; field.
  all: simpl in Hnz; lra || exact Hnz.
Qed.

Lemma brown1972_quat_inv_mul_right : forall a,
  quat_norm_sq a <> 0%R ->
  quat_mul a (brown1972_quat_inv a) = quat_one.
Proof.
  intros [a b c d] Hnz.
  unfold brown1972_quat_inv, quat_mul, quat_scale, quat_conj, quat_one, quat_norm_sq in *.
  simpl.
  f_equal; field.
  all: simpl in Hnz; lra || exact Hnz.
Qed.

Lemma brown1972_quat_norm_conj_preserved : forall a,
  quat_norm_sq (quat_conj a) = quat_norm_sq a.
Proof.
  intros [a b c d].
  unfold quat_norm_sq, quat_conj.
  simpl.
  ring.
Qed.

Lemma brown1972_quat_inv_conj : forall a,
  quat_conj (brown1972_quat_inv a) = brown1972_quat_inv (quat_conj a).
Proof.
  intros a.
  unfold brown1972_quat_inv.
  rewrite quat_conj_scale.
  rewrite quat_conj_involution.
  rewrite brown1972_quat_norm_conj_preserved.
  reflexivity.
Qed.

Lemma brown1972_quat_nat_pow_add : forall a m n,
  brown1972_quat_nat_pow a (m + n) =
  quat_mul (brown1972_quat_nat_pow a m) (brown1972_quat_nat_pow a n).
Proof.
  intros a m n.
  induction n as [|n IH].
  - simpl. rewrite Nat.add_0_r. symmetry. apply quat_mul_one_right.
  - rewrite Nat.add_succ_r. simpl.
    rewrite IH.
    rewrite quat_mul_assoc.
    reflexivity.
Qed.

Lemma brown1972_quat_nat_pow_commute_base : forall a n,
  quat_mul a (brown1972_quat_nat_pow a n) =
  quat_mul (brown1972_quat_nat_pow a n) a.
Proof.
  intros a n.
  induction n as [|n IH].
  - simpl. rewrite quat_mul_one_right. symmetry. apply quat_mul_one_left.
  - simpl.
    rewrite <- quat_mul_assoc.
    rewrite IH.
    rewrite quat_mul_assoc.
    reflexivity.
Qed.

Lemma brown1972_quat_nat_pow_left_step : forall a n,
  quat_mul a (brown1972_quat_nat_pow a n) =
  brown1972_quat_nat_pow a (S n).
Proof.
  intros a n.
  rewrite brown1972_quat_nat_pow_commute_base.
  simpl.
  reflexivity.
Qed.

Lemma brown1972_quat_nat_pow_conj : forall a n,
  quat_conj (brown1972_quat_nat_pow a n) =
  brown1972_quat_nat_pow (quat_conj a) n.
Proof.
  intros a n.
  induction n as [|n IH].
  - unfold quat_conj, quat_one. simpl. apply (f_equal4 mkQuat); ring.
  - simpl.
    rewrite quat_conj_antimorphism.
    rewrite IH.
    apply brown1972_quat_nat_pow_left_step.
Qed.

Lemma brown1972_quat_nat_pow_inv_step_right : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_nat_pow a (S n)) (brown1972_quat_inv a) =
  brown1972_quat_nat_pow a n.
Proof.
  intros a n Hnz.
  simpl.
  rewrite quat_mul_assoc.
  rewrite brown1972_quat_inv_mul_right by exact Hnz.
  apply quat_mul_one_right.
Qed.

Lemma brown1972_quat_nat_pow_inv_step_left : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_nat_pow (brown1972_quat_inv a) (S n)) a =
  brown1972_quat_nat_pow (brown1972_quat_inv a) n.
Proof.
  intros a n Hnz.
  simpl.
  rewrite quat_mul_assoc.
  rewrite brown1972_quat_inv_mul_left by exact Hnz.
  apply quat_mul_one_right.
Qed.

Theorem brown1972_lemma_5_1_quaternion : forall a n,
  quat_conj (brown1972_quat_zpow a n) =
  brown1972_quat_zpow (quat_conj a) n.
Proof.
  intros a [|p|p]; simpl.
  - unfold quat_conj, quat_one. simpl. apply (f_equal4 mkQuat); ring.
  - apply brown1972_quat_nat_pow_conj.
  - rewrite brown1972_quat_nat_pow_conj.
    rewrite brown1972_quat_inv_conj.
    reflexivity.
Qed.

Theorem brown1972_lemma_5_2_quaternion : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_inv a) (brown1972_quat_nat_pow a (S n)) =
  brown1972_quat_nat_pow a n.
Proof.
  intros a n Hnz.
  induction n as [|n IH].
  - simpl. rewrite quat_mul_one_left. apply brown1972_quat_inv_mul_left. exact Hnz.
  - simpl.
    rewrite <- quat_mul_assoc.
    change
      (quat_mul
         (quat_mul (brown1972_quat_inv a) (brown1972_quat_nat_pow a (S n))) a =
       quat_mul (brown1972_quat_nat_pow a n) a).
    rewrite IH.
    reflexivity.
Qed.

Lemma brown1972_quat_mul_zero_right : forall x : CDQuat,
  quat_mul x quat_zero = quat_zero.
Proof.
  intros [a b c d].
  unfold quat_mul, quat_zero.
  simpl.
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_quat_norm_zero : forall a : CDQuat,
  quat_norm_sq a = 0%R -> a = quat_zero.
Proof.
  intros [a b c d] Hnorm.
  unfold quat_norm_sq, quat_zero in Hnorm |- *.
  simpl in Hnorm.
  assert (Ha2 : (0 <= a * a)%R) by nra.
  assert (Hb2 : (0 <= b * b)%R) by nra.
  assert (Hc2 : (0 <= c * c)%R) by nra.
  assert (Hd2 : (0 <= d * d)%R) by nra.
  assert (Ha0 : (a * a = 0)%R) by lra.
  assert (Hb0 : (b * b = 0)%R) by lra.
  assert (Hc0 : (c * c = 0)%R) by lra.
  assert (Hd0 : (d * d = 0)%R) by lra.
  assert (a = 0%R) by nra.
  assert (b = 0%R) by nra.
  assert (c = 0%R) by nra.
  assert (d = 0%R) by nra.
  subst.
  reflexivity.
Qed.

Theorem brown1972_corollary_5_3_quaternion : forall a : CDQuat,
  quat_mul a a = a <-> a = quat_zero \/ a = quat_one.
Proof.
  intros [a b c d].
  split.
  - intro Hidem.
    assert (Hre : qa (quat_mul (mkQuat a b c d) (mkQuat a b c d)) = qa (mkQuat a b c d))
      by exact (f_equal qa Hidem).
    assert (Hb : qb (quat_mul (mkQuat a b c d) (mkQuat a b c d)) = qb (mkQuat a b c d))
      by exact (f_equal qb Hidem).
    assert (Hc : qc (quat_mul (mkQuat a b c d) (mkQuat a b c d)) = qc (mkQuat a b c d))
      by exact (f_equal qc Hidem).
    assert (Hd : qd (quat_mul (mkQuat a b c d) (mkQuat a b c d)) = qd (mkQuat a b c d))
      by exact (f_equal qd Hidem).
    unfold quat_mul, qa, qb, qc, qd in Hre, Hb, Hc, Hd.
    simpl in Hre, Hb, Hc, Hd.
    set (s := (b * b + c * c + d * d)%R).
    assert (Hs : (s = a * a - a)%R).
    { unfold s. nra. }
    assert (Hb' : ((2 * a - 1) * b = 0)%R) by nra.
    assert (Hc' : ((2 * a - 1) * c = 0)%R) by nra.
    assert (Hd' : ((2 * a - 1) * d = 0)%R) by nra.
    assert (Hlin : ((2 * a - 1) * s = 0)%R).
    {
      unfold s.
      replace
        ((2 * a - 1) * (b * b + c * c + d * d))%R
        with ((((2 * a - 1) * b) * b + ((2 * a - 1) * c) * c +
               ((2 * a - 1) * d) * d)%R) by ring.
      rewrite Hb', Hc', Hd'.
      ring.
    }
    destruct (Req_EM_T s 0) as [Hs0 | Hs0].
    + assert (b = 0%R) by (unfold s in Hs0; nra).
      assert (c = 0%R) by (unfold s in Hs0; nra).
      assert (d = 0%R) by (unfold s in Hs0; nra).
      subst b c d.
      assert (a = 0%R \/ a = 1%R) by nra.
      destruct H as [Ha0 | Ha1].
      * left. subst a. reflexivity.
      * right. subst a. reflexivity.
    + assert (Hspos : (s > 0)%R) by nra.
      assert (Ha : (a = / 2)%R) by nra.
      unfold s in Hs.
      nra.
  - intros [Hz | Ho].
    + rewrite Hz. cbv [quat_mul quat_zero qa qb qc qd]. apply (f_equal4 mkQuat); ring.
    + rewrite Ho. cbv [quat_mul quat_one qa qb qc qd]. apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_corollary_5_5_quaternion : forall a n,
  brown1972_quat_nat_pow a (S n) = quat_zero ->
  a = quat_zero.
Proof.
  intros a n.
  induction n as [|n IH].
  - intro Hpow.
    simpl in Hpow.
    rewrite quat_mul_one_left in Hpow.
    exact Hpow.
  - intro Hpow.
    destruct (Req_EM_T (quat_norm_sq a) 0) as [Hz | Hnz].
    + exact (brown1972_quat_norm_zero a Hz).
    + pose proof (brown1972_lemma_5_2_quaternion a (S n) Hnz) as Hstep.
      rewrite Hpow in Hstep.
      rewrite brown1972_quat_mul_zero_right in Hstep.
      exact (IH Hstep).
Qed.

Theorem brown1972_lemma_5_8_quaternion : forall a,
  brown1972_quat_trace (quat_mul a a) =
  (brown1972_quat_trace a * brown1972_quat_trace a - 2 * quat_norm_sq a)%R.
Proof.
  intros a.
  unfold brown1972_quat_trace.
  rewrite quat_re_square.
  ring.
Qed.

Lemma brown1972_quat_zpow_succ : forall a n,
  quat_norm_sq a <> 0%R ->
  brown1972_quat_zpow a (Z.succ n) =
  quat_mul (brown1972_quat_zpow a n) a.
  Proof.
  intros a [|p|p] Hnz.
  - simpl. rewrite quat_mul_one_left. reflexivity.
  - change
      (brown1972_quat_nat_pow a (Pos.to_nat (p + 1)) =
       quat_mul (brown1972_quat_nat_pow a (Pos.to_nat p)) a).
    replace (Pos.to_nat (p + 1)) with (S (Pos.to_nat p)).
    2:{
      rewrite Pos2Nat.inj_add.
      simpl.
      lia.
    }
    simpl.
    reflexivity.
  - destruct p as [p|p|].
    + change
        (brown1972_quat_zpow a (Zneg p~0) =
         quat_mul (brown1972_quat_zpow a (Zneg p~1)) a).
      cbn [brown1972_quat_zpow].
      replace (Pos.to_nat p~1) with (S (Pos.to_nat p~0)).
      2:{
        rewrite Pos2Nat.inj_xI.
        rewrite Pos2Nat.inj_xO.
        lia.
      }
      change
        (brown1972_quat_nat_pow (brown1972_quat_inv a) (Pos.to_nat p~0) =
         quat_mul
           (brown1972_quat_nat_pow (brown1972_quat_inv a)
             (S (Pos.to_nat p~0))) a).
      symmetry. apply brown1972_quat_nat_pow_inv_step_left. exact Hnz.
    + change
        (brown1972_quat_zpow a (Zneg (Pos.pred_double p)) =
         quat_mul (brown1972_quat_zpow a (Zneg p~0)) a).
      cbn [brown1972_quat_zpow].
      replace (Pos.to_nat p~0) with (S (Pos.to_nat (Pos.pred_double p))).
      2:{
        rewrite <- Pos2Nat.inj_succ.
        rewrite Pos.succ_pred_double.
        reflexivity.
      }
      change
        (brown1972_quat_nat_pow (brown1972_quat_inv a) (Pos.to_nat (Pos.pred_double p)) =
         quat_mul
           (brown1972_quat_nat_pow (brown1972_quat_inv a)
             (S (Pos.to_nat (Pos.pred_double p)))) a).
      symmetry. apply brown1972_quat_nat_pow_inv_step_left. exact Hnz.
    + change
        (brown1972_quat_zpow a 0%Z =
         quat_mul (brown1972_quat_zpow a (-1)%Z) a).
      cbn [brown1972_quat_zpow brown1972_quat_nat_pow].
      change (Pos.to_nat 1) with 1%nat.
      cbn [brown1972_quat_nat_pow].
      rewrite quat_mul_one_left.
      apply eq_sym. apply brown1972_quat_inv_mul_left. exact Hnz.
Qed.

Lemma brown1972_quat_zpow_pred : forall a n,
  quat_norm_sq a <> 0%R ->
  brown1972_quat_zpow a (Z.pred n) =
  quat_mul (brown1972_quat_zpow a n) (brown1972_quat_inv a).
Proof.
  intros a [|p|p] Hnz.
  - change
      (brown1972_quat_zpow a (Zneg 1) =
       quat_mul quat_one (brown1972_quat_inv a)).
    cbn [brown1972_quat_zpow brown1972_quat_nat_pow].
    change (Pos.to_nat 1) with 1%nat.
    cbn [brown1972_quat_nat_pow].
    rewrite quat_mul_one_left.
    reflexivity.
  - destruct p as [p|p|].
    + change
        (brown1972_quat_zpow a (Zpos p~0) =
         quat_mul (brown1972_quat_zpow a (Zpos p~1))
           (brown1972_quat_inv a)).
      cbn [brown1972_quat_zpow].
      replace (Pos.to_nat p~1) with (S (Pos.to_nat p~0)).
      2:{
        rewrite Pos2Nat.inj_xI.
        rewrite Pos2Nat.inj_xO.
        lia.
      }
      change
        (brown1972_quat_nat_pow a (Pos.to_nat p~0) =
         quat_mul (brown1972_quat_nat_pow a (S (Pos.to_nat p~0)))
           (brown1972_quat_inv a)).
      symmetry. apply brown1972_quat_nat_pow_inv_step_right. exact Hnz.
    + change
        (brown1972_quat_zpow a (Zpos (Pos.pred_double p)) =
         quat_mul (brown1972_quat_zpow a (Zpos p~0))
           (brown1972_quat_inv a)).
      cbn [brown1972_quat_zpow].
      replace (Pos.to_nat p~0) with (S (Pos.to_nat (Pos.pred_double p))).
      2:{
        rewrite <- Pos2Nat.inj_succ.
        rewrite Pos.succ_pred_double.
        reflexivity.
      }
      change
        (brown1972_quat_nat_pow a (Pos.to_nat (Pos.pred_double p)) =
         quat_mul (brown1972_quat_nat_pow a
           (S (Pos.to_nat (Pos.pred_double p)))) (brown1972_quat_inv a)).
      symmetry. apply brown1972_quat_nat_pow_inv_step_right. exact Hnz.
    + change
        (brown1972_quat_zpow a 0%Z =
         quat_mul (brown1972_quat_zpow a 1%Z) (brown1972_quat_inv a)).
      cbn [brown1972_quat_zpow brown1972_quat_nat_pow].
      change (Pos.to_nat 1) with 1%nat.
      cbn [brown1972_quat_nat_pow].
      rewrite quat_mul_one_left.
      apply eq_sym. apply brown1972_quat_inv_mul_right. exact Hnz.
  - replace (Z.pred (Zneg p)) with (Zneg (Pos.succ p)).
    2:{ destruct p; reflexivity. }
    cbn [brown1972_quat_zpow].
    rewrite Pos2Nat.inj_succ.
    simpl.
    reflexivity.
Qed.

Theorem brown1972_theorem_5_11_quaternion : forall a m n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_zpow a m) (brown1972_quat_zpow a n) =
  brown1972_quat_zpow a (m + n)%Z.
Proof.
  intros a m.
  apply (Z.peano_ind
    (fun n =>
       forall Hnz : quat_norm_sq a <> 0%R,
         quat_mul (brown1972_quat_zpow a m) (brown1972_quat_zpow a n) =
         brown1972_quat_zpow a (m + n)%Z)).
  - intros Hnz. simpl. rewrite quat_mul_one_right. rewrite Z.add_0_r. reflexivity.
  - intros n IH Hnz.
    rewrite brown1972_quat_zpow_succ by exact Hnz.
    rewrite <- quat_mul_assoc.
    rewrite IH by exact Hnz.
    rewrite Z.add_succ_r.
    symmetry. apply brown1972_quat_zpow_succ. exact Hnz.
  - intros n IH Hnz.
    rewrite brown1972_quat_zpow_pred by exact Hnz.
    rewrite <- quat_mul_assoc.
    rewrite IH by exact Hnz.
    rewrite Z.add_pred_r.
    symmetry. apply brown1972_quat_zpow_pred. exact Hnz.
Qed.

Lemma brown1972_quat_zpow_opp_right : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_zpow a n) (brown1972_quat_zpow a (- n)%Z) =
  quat_one.
Proof.
  intros a n Hnz.
  rewrite brown1972_theorem_5_11_quaternion by exact Hnz.
  replace (n + - n)%Z with 0%Z by lia.
  reflexivity.
Qed.

Lemma brown1972_quat_zpow_opp_left : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_mul (brown1972_quat_zpow a (- n)%Z) (brown1972_quat_zpow a n) =
  quat_one.
Proof.
  intros a n Hnz.
  rewrite brown1972_theorem_5_11_quaternion by exact Hnz.
  replace ((- n + n)%Z) with 0%Z by lia.
  reflexivity.
Qed.

Lemma brown1972_quat_zpow_nonzero : forall a n,
  quat_norm_sq a <> 0%R ->
  quat_norm_sq (brown1972_quat_zpow a n) <> 0%R.
Proof.
  intros a n Hnz Hzero.
  pose proof (brown1972_quat_zpow_opp_right a n Hnz) as Hinv.
  apply (f_equal quat_norm_sq) in Hinv.
  rewrite quat_norm_mul in Hinv.
  rewrite Hzero in Hinv.
  unfold quat_norm_sq, quat_one in Hinv.
  simpl in Hinv.
  lra.
Qed.

Lemma brown1972_quat_inverse_unique : forall x y z,
  quat_mul y x = quat_one ->
  quat_mul x z = quat_one ->
  y = z.
Proof.
  intros x y z Hy Hz.
  rewrite <- (quat_mul_one_right y).
  rewrite <- Hz.
  rewrite <- quat_mul_assoc.
  rewrite Hy.
  apply quat_mul_one_left.
Qed.

Lemma brown1972_quat_inv_of_zpow : forall a n,
  quat_norm_sq a <> 0%R ->
  brown1972_quat_inv (brown1972_quat_zpow a n) =
  brown1972_quat_zpow a (- n)%Z.
Proof.
  intros a n Hnz.
  apply brown1972_quat_inverse_unique with (x := brown1972_quat_zpow a n).
  - apply brown1972_quat_inv_mul_left.
    apply brown1972_quat_zpow_nonzero.
    exact Hnz.
  - apply brown1972_quat_zpow_opp_right.
    exact Hnz.
Qed.

Theorem brown1972_theorem_5_12_quaternion : forall a m n,
  quat_norm_sq a <> 0%R ->
  brown1972_quat_zpow (brown1972_quat_zpow a m) n =
  brown1972_quat_zpow a (m * n)%Z.
Proof.
  intros a m.
  apply (Z.peano_ind
    (fun n =>
       forall Hnz : quat_norm_sq a <> 0%R,
         brown1972_quat_zpow (brown1972_quat_zpow a m) n =
         brown1972_quat_zpow a (m * n)%Z)).
  - intros Hnz. simpl. rewrite Z.mul_0_r. reflexivity.
  - intros n IH Hnz.
    rewrite brown1972_quat_zpow_succ.
    2:{ apply brown1972_quat_zpow_nonzero. exact Hnz. }
    rewrite IH by exact Hnz.
    rewrite brown1972_theorem_5_11_quaternion by exact Hnz.
    replace (m * Z.succ n)%Z with (m * n + m)%Z by lia.
    reflexivity.
  - intros n IH Hnz.
    rewrite brown1972_quat_zpow_pred.
    2:{ apply brown1972_quat_zpow_nonzero. exact Hnz. }
    rewrite IH by exact Hnz.
    rewrite brown1972_quat_inv_of_zpow by exact Hnz.
    rewrite brown1972_theorem_5_11_quaternion by exact Hnz.
    replace (m * Z.pred n)%Z with (m * n + - m)%Z by lia.
    reflexivity.
Qed.

Definition brown1972_quat_trace (q : CDQuat) : R := 2 * qa q.

Theorem brown1972_lemma_5_13_quaternion : forall x,
  brown1972_quat_trace x = 0%R ->
  quat_mul x x = quat_scale (- quat_norm_sq x) quat_one.
Proof.
  intros x Htr.
  destruct x as [a b c d].
  unfold brown1972_quat_trace in Htr.
  simpl in Htr.
  apply quat_imaginary_square.
  simpl.
  nra.
Qed.

Theorem brown1972_lemma_5_14_quaternion : forall a b c,
  quat_mul (quat_assoc a b c) (quat_assoc a b c) =
  quat_scale (- quat_norm_sq (quat_assoc a b c)) quat_one.
Proof.
  intros a b c.
  rewrite quat_assoc_zero.
  unfold quat_mul, quat_scale, quat_norm_sq, quat_zero, quat_one.
  simpl.
  apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_lemma_5_15_quaternion : forall a b n,
  quat_assoc (brown1972_quat_nat_pow a n) b a = quat_zero.
Proof.
  intros a b n.
  apply quat_assoc_zero.
Qed.

Theorem brown1972_lemma_5_16_quaternion : forall a b m n,
  quat_assoc (brown1972_quat_nat_pow a m) b (brown1972_quat_nat_pow a n) =
  quat_zero.
Proof.
  intros a b m n.
  apply quat_assoc_zero.
Qed.

Theorem brown1972_theorem_5_17_quaternion : forall a b m n,
  quat_assoc (brown1972_quat_zpow a m) b (brown1972_quat_zpow a n) =
  quat_zero.
Proof.
  intros a b m n.
  apply quat_assoc_zero.
Qed.

Definition brown1972_oct_one : CDOct := mkOct quat_one quat_zero.

Definition brown1972_oct_inv (a : CDOct) : CDOct :=
  oct_scale (/ oct_norm_sq a) (oct_conj a).

Fixpoint brown1972_oct_nat_pow (a : CDOct) (n : nat) : CDOct :=
  match n with
  | O => brown1972_oct_one
  | S k => oct_mul (brown1972_oct_nat_pow a k) a
  end.

Definition brown1972_oct_zpow (a : CDOct) (n : Z) : CDOct :=
  match n with
  | Z0 => brown1972_oct_one
  | Zpos p => brown1972_oct_nat_pow a (Pos.to_nat p)
  | Zneg p => brown1972_oct_nat_pow (brown1972_oct_inv a) (Pos.to_nat p)
  end.

Lemma brown1972_oct_mul_one_left : forall x : CDOct,
  oct_mul brown1972_oct_one x = x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [brown1972_oct_one oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma brown1972_oct_conj_involution : forall x : CDOct,
  oct_conj (oct_conj x) = x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_conj oct_lo oct_hi quat_conj quat_neg qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma brown1972_oct_norm_conj_preserved : forall a,
  oct_norm_sq (oct_conj a) = oct_norm_sq a.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [oct_norm_sq oct_conj quat_norm_sq quat_conj quat_neg oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Lemma brown1972_oct_inv_conj : forall a,
  oct_conj (brown1972_oct_inv a) = brown1972_oct_inv (oct_conj a).
Proof.
  intros a.
  unfold brown1972_oct_inv.
  rewrite oct_conj_scale.
  rewrite brown1972_oct_conj_involution.
  rewrite brown1972_oct_norm_conj_preserved.
  reflexivity.
Qed.

Lemma brown1972_oct_nat_pow_commute_base : forall a n,
  oct_mul a (brown1972_oct_nat_pow a n) =
  oct_mul (brown1972_oct_nat_pow a n) a.
Proof.
  intros a n.
  induction n as [|n IH].
  - simpl. rewrite oct_mul_one_right. symmetry. apply brown1972_oct_mul_one_left.
  - simpl.
    symmetry.
    rewrite oct_flexible.
    rewrite IH.
    reflexivity.
Qed.

Lemma brown1972_oct_nat_pow_left_step : forall a n,
  oct_mul a (brown1972_oct_nat_pow a n) =
  brown1972_oct_nat_pow a (S n).
Proof.
  intros a n.
  rewrite brown1972_oct_nat_pow_commute_base.
  simpl.
  reflexivity.
Qed.

Lemma brown1972_oct_nat_pow_conj : forall a n,
  oct_conj (brown1972_oct_nat_pow a n) =
  brown1972_oct_nat_pow (oct_conj a) n.
Proof.
  intros a n.
  induction n as [|n IH].
  - unfold oct_conj, brown1972_oct_one. simpl. f_equal; apply quat_conj_involution.
  - simpl.
    rewrite oct_conj_antimorphism.
    rewrite IH.
    apply brown1972_oct_nat_pow_left_step.
Qed.

Theorem brown1972_lemma_5_1_octonion : forall a n,
  oct_conj (brown1972_oct_zpow a n) =
  brown1972_oct_zpow (oct_conj a) n.
Proof.
  intros a [|p|p]; simpl.
  - unfold oct_conj, brown1972_oct_one. simpl. f_equal; apply quat_conj_involution.
  - apply brown1972_oct_nat_pow_conj.
  - rewrite brown1972_oct_nat_pow_conj.
    rewrite brown1972_oct_inv_conj.
    reflexivity.
Qed.

Record Brown1972ChapterVQuaternionSurface := {
  brown1972_ch5_l51_quat :
    forall a n,
      quat_conj (brown1972_quat_zpow a n) =
      brown1972_quat_zpow (quat_conj a) n;
  brown1972_ch5_l52_quat :
    forall a n,
      quat_norm_sq a <> 0%R ->
      quat_mul (brown1972_quat_inv a) (brown1972_quat_nat_pow a (S n)) =
      brown1972_quat_nat_pow a n;
  brown1972_ch5_c53_quat :
    forall a : CDQuat,
      quat_mul a a = a <-> a = quat_zero \/ a = quat_one;
  brown1972_ch5_c55_quat :
    forall a n,
      brown1972_quat_nat_pow a (S n) = quat_zero ->
      a = quat_zero;
  brown1972_ch5_l58_quat :
    forall a,
      brown1972_quat_trace (quat_mul a a) =
      (brown1972_quat_trace a * brown1972_quat_trace a - 2 * quat_norm_sq a)%R;
  brown1972_ch5_t511_quat :
    forall a m n,
      quat_norm_sq a <> 0%R ->
      quat_mul (brown1972_quat_zpow a m) (brown1972_quat_zpow a n) =
      brown1972_quat_zpow a (m + n)%Z;
  brown1972_ch5_t512_quat :
    forall a m n,
      quat_norm_sq a <> 0%R ->
      brown1972_quat_zpow (brown1972_quat_zpow a m) n =
      brown1972_quat_zpow a (m * n)%Z;
  brown1972_ch5_l513_quat :
    forall x,
      brown1972_quat_trace x = 0%R ->
      quat_mul x x = quat_scale (- quat_norm_sq x) quat_one;
  brown1972_ch5_l514_quat :
    forall a b c,
      quat_mul (quat_assoc a b c) (quat_assoc a b c) =
      quat_scale (- quat_norm_sq (quat_assoc a b c)) quat_one;
  brown1972_ch5_l515_quat :
    forall a b n,
      quat_assoc (brown1972_quat_nat_pow a n) b a = quat_zero;
  brown1972_ch5_l516_quat :
    forall a b m n,
      quat_assoc (brown1972_quat_nat_pow a m) b (brown1972_quat_nat_pow a n) =
      quat_zero;
  brown1972_ch5_t517_quat :
    forall a b m n,
      quat_assoc (brown1972_quat_zpow a m) b (brown1972_quat_zpow a n) =
      quat_zero
}.

Definition brown1972_quaternion_chapter_v_surface :
  Brown1972ChapterVQuaternionSurface.
Proof.
  refine {| brown1972_ch5_l51_quat := brown1972_lemma_5_1_quaternion;
            brown1972_ch5_l52_quat := brown1972_lemma_5_2_quaternion;
            brown1972_ch5_c53_quat := brown1972_corollary_5_3_quaternion;
            brown1972_ch5_c55_quat := brown1972_corollary_5_5_quaternion;
            brown1972_ch5_l58_quat := brown1972_lemma_5_8_quaternion;
            brown1972_ch5_t511_quat := brown1972_theorem_5_11_quaternion;
            brown1972_ch5_t512_quat := brown1972_theorem_5_12_quaternion;
            brown1972_ch5_l513_quat := brown1972_lemma_5_13_quaternion;
            brown1972_ch5_l514_quat := brown1972_lemma_5_14_quaternion;
            brown1972_ch5_l515_quat := brown1972_lemma_5_15_quaternion;
            brown1972_ch5_l516_quat := brown1972_lemma_5_16_quaternion;
            brown1972_ch5_t517_quat := brown1972_theorem_5_17_quaternion |}.
Defined.

Record Brown1972ChapterVInitialOctonionLift := {
  brown1972_ch5_l51_oct :
    forall a n,
      oct_conj (brown1972_oct_zpow a n) =
      brown1972_oct_zpow (oct_conj a) n
}.

Definition brown1972_chapter_v_initial_octonion_lift :
  Brown1972ChapterVInitialOctonionLift.
Proof.
  refine {| brown1972_ch5_l51_oct := brown1972_lemma_5_1_octonion |}.
Defined.

Theorem Brown1972_lane_compiles : True.
Proof. exact I. Qed.
