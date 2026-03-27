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
      landed here; Rust lane `crates/brown_1972/src/norm_symmetry.rs` remains
      the computational mirror and still carries the broader 16D/generalized
      norm exploration.
    - Chapter IV, pp. 20-22, Theorems 4.2-4.3 and Corollary 4.4:
      source-driven standard-tower witnesses for 4.2 and 4.4 are now landed
      here; the exact one-step associator formula of 4.3 is the next local
      Brown theorem still open.
    - Chapter V, pp. 27-30, Theorems 5.11-5.17:
      Rust lane `crates/brown_1972/src/exponent_properties.rs`; dedicated Rocq
      lane is still open.
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
    - exact Brown-numbered Chapter IV theorem 4.3 formula
    - Brown-numbered Chapter V theorem surface over the landed exponent lane
    - broader 16D/generalized-norm Chapter III lane
    - Brown-numbered Chapter VI basis-element theorem lanes
    - remaining Chapter VII numbering gaps plus Appendix C extraction bridge in Rocq

    The executable Rust companion for this paper is `crates/brown_1972/`. *)

From Stdlib Require Import List Reals.
Import ListNotations.
Open Scope R_scope.

From OpenGororoba Require Import
  ZDGraph
  Sedenion
  CayleyDicksonAlgebra
  OctonionNorm
  CDAssociator
  CDPowerAssociative
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

(** Brown Chapter IV starts the source-driven structural lane:
    4.2 proves flexibility for the tower,
    4.3 gives the one-step associator formula,
    4.4 extracts alternativity exactly when the generating algebra is
    associative. We currently land these as standard-tower witnesses. *)

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
            brown1972_ch4_c44_oct_left := brown1972_corollary_4_4_octonion_left;
            brown1972_ch4_c44_oct_right := brown1972_corollary_4_4_octonion_right |}.
Defined.

Theorem Brown1972_lane_compiles : True.
Proof. exact I. Qed.
