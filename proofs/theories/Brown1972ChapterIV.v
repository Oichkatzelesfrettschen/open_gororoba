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
    - Chapter III, pp. 11-16, Theorem 3.1, Theorem 3.3, Lemma 3.7,
      Theorem 3.9, and Lemma 3.10:
      a standard-octonion Brown 3.1 trace surface, a standard-octonion
      Brown 3.3 / 3.7 involution-quadratic surface, a weaker shared
      octonion/sedenion quadratic/conjugation core, and the abstract Rocq
      norm/involution surface with standard-octonion and direct
      standard-sedenion Brown 3.9 / 3.10 witnesses are landed here; Rust lane
      `crates/brown_1972/src/norm_symmetry.rs` remains the computational
      mirror for the broader quadratic/conjugation exploration beyond these
      concrete 8D/16D witnesses.
    - Chapter IV, pp. 20-22, Theorems 4.2-4.3 and Corollary 4.4:
      source-driven standard-tower witnesses for 4.2, 4.3, and 4.4 are now
      landed here.
    - Chapter V, pp. 27-30, Theorems 5.11-5.17:
      a generic one-generated/trace-zero exponent surface is now landed here,
      instantiated concretely for quaternions and octonions; Rust lane
      `crates/brown_1972/src/exponent_properties.rs` remains the broader
      computational mirror for farther non-quaternion/generalized follow-on
      exploration.
    - Chapter VI, pp. 30-42, Theorems 6.2-6.17:
      a standard-octonion Brown 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15
      basis-associator surface is now landed, a source-faithful standard-
      octonion anticommutator lane for Brown 6.16 / 6.17 is now landed, a
      direct standard-sedenion adjoined-element / polynomial lane for
      6.1 / 6.2 / 6.3 / 6.4 / 6.5 / 6.6 / 6.7 / 6.8 is now landed, that
      Brown 6.4-6.7 lane is also packaged as a broader adjoined/polynomial
      lift interface above literal `mkSed` coordinates, and proof-faithful
      constructive 6.9 witnesses are now landed; the printed Brown 6.9
      pointwise iff wording does not survive unchanged in the repo's literal
      standard-pair coordinates, so the current Rocq surface records the
      constructive implications and the family form Brown's p.35 proof
      actually uses; the next Chapter VI work is any farther non-standard-
      model lift beside the broader Chapter III quadratic/conjugation lift.
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
    - broader Chapter V exponent surface beyond the current quaternion/octonion
      one-generated/trace-zero surface
    - broader Chapter III quadratic/conjugation lane beyond the landed Brown
      3.1 / 3.3 / 3.7 standard-octonion source surface, the weaker shared
      octonion/sedenion quadratic/conjugation core, and the concrete
      octonion/sedenion 3.9 / 3.10 witnesses
    - broader Brown-numbered Chapter VI basis-element theorem lanes beyond the
      landed standard-octonion 6.10 / 6.11 / 6.12 / 6.13 / 6.14 / 6.15
      basis-associator surface, the landed standard-octonion 6.16 / 6.17
      anticommutator surface, and the new broader 6.4-6.7 adjoined/
      polynomial lift interface
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
  CDSignBridge
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
