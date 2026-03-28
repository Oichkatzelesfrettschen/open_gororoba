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
      a generic one-generated/trace-zero exponent surface is now landed in
      `Brown1972ChapterV.v`,
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
  BrownAssessorEquivalence
  Brown1972ChapterIII
  Brown1972ChapterV
  Brown1972ChapterIV
  Brown1972ChapterVI.

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

Theorem Brown1972_lane_compiles : True.
Proof. exact I. Qed.
