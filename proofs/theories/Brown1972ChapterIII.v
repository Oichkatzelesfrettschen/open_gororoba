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
  CDFusedBilinear
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

Definition brown1972_quat_trace (q : CDQuat) : R := 2 * qa q.

Definition brown1972_quat_sub (x y : CDQuat) : CDQuat :=
  quat_add x (quat_neg y).

Lemma brown1972_quaternion_norm_conj_preserved : forall q : CDQuat,
  quat_norm_sq (quat_conj q) = quat_norm_sq q.
Proof.
  intros [a b c d].
  cbv [quat_norm_sq quat_conj].
  simpl.
  ring.
Qed.

Lemma brown1972_quat_trace_add : forall x y : CDQuat,
  brown1972_quat_trace (quat_add x y) =
  (brown1972_quat_trace x + brown1972_quat_trace y)%R.
Proof.
  intros [a b c d] [e f g h].
  cbv [brown1972_quat_trace quat_add qa].
  ring.
Qed.

Lemma brown1972_quat_trace_neg : forall x : CDQuat,
  brown1972_quat_trace (quat_neg x) = (- brown1972_quat_trace x)%R.
Proof.
  intros [a b c d].
  cbv [brown1972_quat_trace quat_neg qa].
  ring.
Qed.

Lemma brown1972_quat_trace_conj : forall x : CDQuat,
  brown1972_quat_trace (quat_conj x) = brown1972_quat_trace x.
Proof.
  intros [a b c d].
  cbv [brown1972_quat_trace quat_conj qa].
  ring.
Qed.

Lemma brown1972_quat_trace_sub : forall x y : CDQuat,
  brown1972_quat_trace (brown1972_quat_sub x y) =
  (brown1972_quat_trace x - brown1972_quat_trace y)%R.
Proof.
  intros x y.
  unfold brown1972_quat_sub.
  rewrite brown1972_quat_trace_add.
  rewrite brown1972_quat_trace_neg.
  ring.
Qed.

Lemma brown1972_quat_scale_zero : forall x : CDQuat,
  quat_scale 0 x = quat_zero.
Proof.
  intros [a b c d].
  unfold quat_scale, quat_zero.
  simpl.
  apply (f_equal4 mkQuat); ring.
Qed.

Theorem brown1972_theorem_3_1_i_quaternion : forall x y : CDQuat,
  brown1972_quat_trace (quat_mul x y) =
  brown1972_quat_trace (quat_mul y x).
Proof.
  intros [a b c d] [e f g h].
  cbv [brown1972_quat_trace quat_mul qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_1_ii_quaternion : forall x y z : CDQuat,
  brown1972_quat_trace (quat_mul (quat_mul x y) z) =
  brown1972_quat_trace (quat_mul x (quat_mul y z)).
Proof.
  intros [a b c d] [e f g h] [i j k l].
  cbv [brown1972_quat_trace quat_mul qa qb qc qd].
  ring.
Qed.

Lemma brown1972_quat_trace_zero_iff_pure : forall x : CDQuat,
  brown1972_quat_trace x = 0%R <-> quat_conj x = quat_neg x.
Proof.
  intros [a b c d].
  split.
  - intro H.
    cbv [brown1972_quat_trace quat_conj quat_neg qa] in H |- *.
    assert (Ha : a = 0%R) by lra.
    subst a.
    f_equal; ring.
  - intro H.
    apply (f_equal brown1972_quat_trace) in H.
    rewrite brown1972_quat_trace_conj in H.
    rewrite brown1972_quat_trace_neg in H.
    lra.
Qed.

Lemma brown1972_quat_quadratic_identity : forall x : CDQuat,
  quat_mul x x =
  quat_add (quat_scale (brown1972_quat_trace x) x)
           (quat_scale (- quat_norm_sq x) quat_one).
Proof.
  intros [a b c d].
  cbv [brown1972_quat_trace quat_mul quat_add quat_scale quat_norm_sq
       quat_one quat_zero quat_conj qa qb qc qd].
  f_equal; ring.
Qed.

Lemma brown1972_quaternion_chapter_iii_pure_square : forall x : CDQuat,
  brown1972_quat_trace x = 0%R ->
  quat_mul x x = quat_scale (- quat_norm_sq x) quat_one.
Proof.
  intros x Htr.
  rewrite brown1972_quat_quadratic_identity.
  assert (Hzero :
    quat_scale (brown1972_quat_trace x) x = quat_zero).
  {
    rewrite Htr.
    destruct x as [a b c d].
    unfold quat_scale, quat_zero.
    simpl.
    apply (f_equal4 mkQuat); ring.
  }
  rewrite Hzero.
  apply quat_add_zero_left.
Qed.

Module BrownChapterIIIQuatAlg <: BrownChapterIIINormAlg.
  Definition A := CDQuat.
  Definition add := quat_add.
  Definition sub := brown1972_quat_sub.
  Definition mul := quat_mul.
  Definition conj := quat_conj.
  Definition norm_sq := quat_norm_sq.

  Theorem brown_norm_conj_preserved : forall x,
    norm_sq (conj x) = norm_sq x.
  Proof.
    exact brown1972_quaternion_norm_conj_preserved.
  Qed.

  Theorem brown_norm_mul : forall x y,
    norm_sq (mul x y) = (norm_sq x * norm_sq y)%R.
  Proof.
    intros [a b c d] [e f g h].
    cbv [norm_sq mul quat_norm_sq quat_mul qa qb qc qd].
    ring.
  Qed.

  Theorem brown_polarization_identity : forall x y,
    (norm_sq (add x y) + norm_sq (sub x y))%R =
    (2 * (norm_sq x + norm_sq y))%R.
  Proof.
    intros [a b c d] [e f g h].
    cbv [norm_sq add sub brown1972_quat_sub quat_norm_sq quat_add quat_neg qa qb qc qd].
    ring.
  Qed.
End BrownChapterIIIQuatAlg.

Module BrownQuaternionChapterIII := BrownChapterIII(BrownChapterIIIQuatAlg).

Theorem brown1972_theorem_3_9_i_quaternion : forall x y : CDQuat,
  quat_norm_sq (quat_mul x (quat_conj y)) = quat_norm_sq (quat_mul x y).
Proof.
  exact BrownQuaternionChapterIII.brown1972_theorem_3_9_i.
Qed.

Theorem brown1972_theorem_3_9_ii_quaternion : forall x y : CDQuat,
  quat_norm_sq (quat_mul (quat_conj x) y) = quat_norm_sq (quat_mul x y).
Proof.
  exact BrownQuaternionChapterIII.brown1972_theorem_3_9_ii.
Qed.

Theorem brown1972_theorem_3_9_iii_quaternion : forall x y : CDQuat,
  quat_norm_sq (quat_mul x y) = quat_norm_sq (quat_mul y x).
Proof.
  exact BrownQuaternionChapterIII.brown1972_theorem_3_9_iii.
Qed.

Theorem brown1972_lemma_3_10_quaternion : forall x y : CDQuat,
  (quat_norm_sq (quat_add x y) + quat_norm_sq (brown1972_quat_sub x y))%R =
  (2 * (quat_norm_sq x + quat_norm_sq y))%R.
Proof.
  exact BrownQuaternionChapterIII.brown1972_lemma_3_10.
Qed.

Lemma brown1972_octonion_norm_conj_preserved : forall x : CDOct,
  oct_norm_sq (oct_conj x) = oct_norm_sq x.
Proof.
  intros [[a b c d] [e f g h]].
  unfold oct_norm_sq, oct_conj, quat_norm_sq, quat_conj, quat_neg.
  simpl. ring.
Qed.

Definition brown1972_oct_trace (x : CDOct) : R := 2 * qa (oct_lo x).

Definition brown1972_oct_one : CDOct := mkOct quat_one quat_zero.

Ltac brown1972_close_oct_ring :=
  cbv [brown1972_oct_one brown1972_oct_trace
       oct_assoc oct_add oct_sub oct_neg oct_mul oct_conj oct_scale oct_zero oct_e
       oct_norm_sq oct_lo oct_hi
       quat_add quat_neg quat_mul quat_conj quat_scale quat_zero quat_one
       quat_norm_sq qa qb qc qd];
  apply (f_equal2 mkOct);
  [apply (f_equal4 mkQuat); ring | apply (f_equal4 mkQuat); ring].

Lemma brown1972_oct_mul_one_left : forall x : CDOct,
  oct_mul brown1972_oct_one x = x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [brown1972_oct_one oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

Lemma brown1972_oct_mul_neg_e0_left : forall x : CDOct,
  oct_mul (oct_neg (oct_e 0)) x = oct_neg x.
Proof.
  intro x.
  rewrite oct_neg_mul_left.
  change (oct_e 0) with brown1972_oct_one.
  rewrite brown1972_oct_mul_one_left.
  reflexivity.
Qed.

Lemma brown1972_oct_mul_neg_e0_right : forall x : CDOct,
  oct_mul x (oct_neg (oct_e 0)) = oct_neg x.
Proof.
  intro x.
  rewrite oct_neg_mul_right.
  change (oct_e 0) with brown1972_oct_one.
  rewrite oct_mul_one_right.
  reflexivity.
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
Lemma brown1972_oct_scale_scale : forall r s : R, forall x : CDOct,
  oct_scale r (oct_scale s x) = oct_scale (r * s) x.
Proof.
  intros r s [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  unfold oct_scale, quat_scale.
  simpl.
  apply (f_equal2 mkOct).
  + apply (f_equal4 mkQuat); ring.
  + apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_oct_scale_one : forall x : CDOct,
  oct_scale 1 x = x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  unfold oct_scale, quat_scale.
  simpl.
  apply (f_equal2 mkOct).
  + apply (f_equal4 mkQuat); ring.
  + apply (f_equal4 mkQuat); ring.
Qed.

Lemma brown1972_oct_sub_eq_zero : forall x y : CDOct,
  oct_sub x y = oct_zero ->
  x = y.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]] H.
  unfold oct_sub, oct_add, oct_neg, oct_zero in H.
  simpl in H.
  inversion H.
  inversion H1.
  inversion H2.
  inversion H3.
  inversion H4.
  inversion H5.
  inversion H6.
  inversion H7.
  inversion H8.
  subst.
  apply (f_equal2 mkOct).
  + apply (f_equal4 mkQuat); lra.
  + apply (f_equal4 mkQuat); lra.
Qed.

Lemma brown1972_oct_quadratic_identity : forall a,
  oct_mul a a =
  oct_add (oct_scale (brown1972_oct_trace a) a)
          (oct_scale (- oct_norm_sq a) brown1972_oct_one).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [brown1972_oct_trace brown1972_oct_one
       oct_add oct_scale oct_mul oct_conj oct_norm_sq oct_lo oct_hi
       quat_add quat_scale quat_mul quat_neg quat_conj quat_one quat_zero
       quat_norm_sq qa qb qc qd].
  apply (f_equal2 mkOct).
  + apply (f_equal4 mkQuat); ring.
  + apply (f_equal4 mkQuat); ring.
Qed.
Lemma brown1972_oct_norm_zero : forall a : CDOct,
  oct_norm_sq a = 0%R -> a = oct_zero.
Proof.
  intros [[a b c d] [e f g h]] Hnorm.
  unfold oct_norm_sq, quat_norm_sq, oct_zero, quat_zero in Hnorm |- *.
  simpl in Hnorm.
  assert (Ha2 : (0 <= a * a)%R) by nra.
  assert (Hb2 : (0 <= b * b)%R) by nra.
  assert (Hc2 : (0 <= c * c)%R) by nra.
  assert (Hd2 : (0 <= d * d)%R) by nra.
  assert (He2 : (0 <= e * e)%R) by nra.
  assert (Hf2 : (0 <= f * f)%R) by nra.
  assert (Hg2 : (0 <= g * g)%R) by nra.
  assert (Hh2 : (0 <= h * h)%R) by nra.
  assert (Ha0 : (a * a = 0)%R) by lra.
  assert (Hb0 : (b * b = 0)%R) by lra.
  assert (Hc0 : (c * c = 0)%R) by lra.
  assert (Hd0 : (d * d = 0)%R) by lra.
  assert (He0 : (e * e = 0)%R) by lra.
  assert (Hf0 : (f * f = 0)%R) by lra.
  assert (Hg0 : (g * g = 0)%R) by lra.
  assert (Hh0 : (h * h = 0)%R) by lra.
  assert (a = 0%R) by nra.
  assert (b = 0%R) by nra.
  assert (c = 0%R) by nra.
  assert (d = 0%R) by nra.
  assert (e = 0%R) by nra.
  assert (f = 0%R) by nra.
  assert (g = 0%R) by nra.
  assert (h = 0%R) by nra.
  subst.
  reflexivity.
Qed.

Lemma brown1972_oct_trace_add : forall x y : CDOct,
  brown1972_oct_trace (oct_add x y) =
  (brown1972_oct_trace x + brown1972_oct_trace y)%R.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [brown1972_oct_trace oct_add oct_lo qa quat_add].
  ring.
Qed.

Lemma brown1972_oct_trace_neg : forall x : CDOct,
  brown1972_oct_trace (oct_neg x) = (- brown1972_oct_trace x)%R.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [brown1972_oct_trace oct_neg oct_lo qa quat_neg].
  ring.
Qed.

Lemma brown1972_oct_trace_conj : forall x : CDOct,
  brown1972_oct_trace (oct_conj x) = brown1972_oct_trace x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [brown1972_oct_trace oct_conj oct_lo qa quat_conj].
  ring.
Qed.

Lemma brown1972_oct_trace_sub : forall x y : CDOct,
  brown1972_oct_trace (oct_sub x y) =
  (brown1972_oct_trace x - brown1972_oct_trace y)%R.
Proof.
  intros x y.
  unfold oct_sub.
  rewrite brown1972_oct_trace_add.
  rewrite brown1972_oct_trace_neg.
  ring.
Qed.

Theorem brown1972_theorem_3_1_i_octonion : forall x y : CDOct,
  brown1972_oct_trace (oct_mul x y) = brown1972_oct_trace (oct_mul y x).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [brown1972_oct_trace oct_mul oct_lo oct_hi oct_conj
       quat_mul quat_add quat_neg quat_conj qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_1_assoc_trace_zero_octonion : forall x y z : CDOct,
  brown1972_oct_trace (oct_assoc x y z) = 0%R.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]]
         [[c1 c2 c3 c4] [c5 c6 c7 c8]].
  cbv [brown1972_oct_trace oct_assoc oct_sub oct_add oct_neg oct_mul oct_conj
       oct_lo oct_hi quat_mul quat_add quat_neg quat_conj qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_1_ii_octonion : forall x y z : CDOct,
  brown1972_oct_trace (oct_mul (oct_mul x y) z) =
  brown1972_oct_trace (oct_mul x (oct_mul y z)).
Proof.
  intros x y z.
  pose proof (brown1972_theorem_3_1_assoc_trace_zero_octonion x y z) as Htr.
  unfold oct_assoc in Htr.
  rewrite brown1972_oct_trace_sub in Htr.
  lra.
Qed.

Theorem brown1972_theorem_3_3_i_octonion : forall x y : CDOct,
  oct_mul (oct_conj x) (oct_mul x y) = oct_scale (oct_norm_sq x) y /\
  oct_mul x (oct_mul (oct_conj x) y) = oct_scale (oct_norm_sq x) y.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  split; brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_3_3_ii_octonion : forall x y : CDOct,
  oct_mul (oct_mul y x) (oct_conj x) = oct_scale (oct_norm_sq x) y /\
  oct_mul (oct_mul y (oct_conj x)) x = oct_scale (oct_norm_sq x) y.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  split; brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_3_3_iii_octonion : forall x y : CDOct,
  oct_mul (oct_conj x) (oct_mul (oct_mul x x) y) =
  oct_mul (oct_mul x x) (oct_mul (oct_conj x) y).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_3_3_iv_octonion : forall x y : CDOct,
  oct_mul (oct_conj x) (oct_mul x y) =
  oct_mul (oct_mul y x) (oct_conj x).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_theorem_3_3_v_octonion : forall x y : CDOct,
  oct_mul (oct_mul x x) (oct_mul y (oct_conj x)) =
  oct_mul (oct_mul (oct_mul x x) y) (oct_conj x).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  brown1972_close_oct_ring.
Qed.

Theorem brown1972_lemma_3_7_octonion : forall x : CDOct,
  oct_add (oct_mul x x)
          (oct_add (oct_scale (- brown1972_oct_trace x) x)
                   (oct_scale (oct_norm_sq x) brown1972_oct_one)) =
  oct_zero.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  brown1972_close_oct_ring.
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

(** Brown Theorem 3.1, standard-octonion trace witnesses. *)
Record Brown1972ChapterIIITraceSurface := {
  brown1972_ch3_t31_i :
    forall x y : CDOct,
      brown1972_oct_trace (oct_mul x y) = brown1972_oct_trace (oct_mul y x);
  brown1972_ch3_t31_ii :
    forall x y z : CDOct,
      brown1972_oct_trace (oct_mul (oct_mul x y) z) =
      brown1972_oct_trace (oct_mul x (oct_mul y z))
}.

Definition brown1972_octonion_chapter_iii_trace_surface :
  Brown1972ChapterIIITraceSurface.
Proof.
  refine {| brown1972_ch3_t31_i := brown1972_theorem_3_1_i_octonion;
            brown1972_ch3_t31_ii := brown1972_theorem_3_1_ii_octonion |}.
Defined.

Record Brown1972ChapterIIIBasicConsequencesSurface := {
  brown1972_ch3_t33_i :
    forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul x y) = oct_scale (oct_norm_sq x) y /\
      oct_mul x (oct_mul (oct_conj x) y) = oct_scale (oct_norm_sq x) y;
  brown1972_ch3_t33_ii :
    forall x y : CDOct,
      oct_mul (oct_mul y x) (oct_conj x) = oct_scale (oct_norm_sq x) y /\
      oct_mul (oct_mul y (oct_conj x)) x = oct_scale (oct_norm_sq x) y;
  brown1972_ch3_t33_iii :
    forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul (oct_mul x x) y) =
      oct_mul (oct_mul x x) (oct_mul (oct_conj x) y);
  brown1972_ch3_t33_iv :
    forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul x y) =
      oct_mul (oct_mul y x) (oct_conj x);
  brown1972_ch3_t33_v :
    forall x y : CDOct,
      oct_mul (oct_mul x x) (oct_mul y (oct_conj x)) =
      oct_mul (oct_mul (oct_mul x x) y) (oct_conj x);
  brown1972_ch3_l37 :
    forall x : CDOct,
      oct_add (oct_mul x x)
              (oct_add (oct_scale (- brown1972_oct_trace x) x)
                       (oct_scale (oct_norm_sq x) brown1972_oct_one)) =
      oct_zero
}.

Definition brown1972_octonion_chapter_iii_basic_consequences_surface :
  Brown1972ChapterIIIBasicConsequencesSurface.
Proof.
  refine {| brown1972_ch3_t33_i := brown1972_theorem_3_3_i_octonion;
            brown1972_ch3_t33_ii := brown1972_theorem_3_3_ii_octonion;
            brown1972_ch3_t33_iii := brown1972_theorem_3_3_iii_octonion;
            brown1972_ch3_t33_iv := brown1972_theorem_3_3_iv_octonion;
            brown1972_ch3_t33_v := brown1972_theorem_3_3_v_octonion;
            brown1972_ch3_l37 := brown1972_lemma_3_7_octonion |}.
Defined.

Theorem brown1972_chapter_iii_basic_consequences_summary :
  (forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul x y) = oct_scale (oct_norm_sq x) y /\
      oct_mul x (oct_mul (oct_conj x) y) = oct_scale (oct_norm_sq x) y) /\
  (forall x y : CDOct,
      oct_mul (oct_mul y x) (oct_conj x) = oct_scale (oct_norm_sq x) y /\
      oct_mul (oct_mul y (oct_conj x)) x = oct_scale (oct_norm_sq x) y) /\
  (forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul (oct_mul x x) y) =
      oct_mul (oct_mul x x) (oct_mul (oct_conj x) y)) /\
  (forall x y : CDOct,
      oct_mul (oct_conj x) (oct_mul x y) =
      oct_mul (oct_mul y x) (oct_conj x)) /\
  (forall x y : CDOct,
      oct_mul (oct_mul x x) (oct_mul y (oct_conj x)) =
      oct_mul (oct_mul (oct_mul x x) y) (oct_conj x)) /\
  (forall x : CDOct,
      oct_add (oct_mul x x)
              (oct_add (oct_scale (- brown1972_oct_trace x) x)
                       (oct_scale (oct_norm_sq x) brown1972_oct_one)) =
      oct_zero).
Proof.
  split.
  - exact brown1972_theorem_3_3_i_octonion.
  - split.
    + exact brown1972_theorem_3_3_ii_octonion.
    + split.
      * exact brown1972_theorem_3_3_iii_octonion.
      * split.
        { exact brown1972_theorem_3_3_iv_octonion. }
        split.
        { exact brown1972_theorem_3_3_v_octonion. }
        exact brown1972_lemma_3_7_octonion.
Qed.

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

Definition brown1972_sed_trace (x : CDSed) : R :=
  2 * qa (oct_lo (sed_lo x)).

Theorem brown1972_theorem_3_1_i_sedenion : forall x y : CDSed,
  brown1972_sed_trace (sed_mul x y) = brown1972_sed_trace (sed_mul y x).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [brown1972_sed_trace sed_mul
       oct_add oct_mul oct_conj oct_neg
       quat_add quat_mul quat_conj quat_neg
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

Theorem brown1972_theorem_3_1_ii_sedenion : forall x y z : CDSed,
  brown1972_sed_trace (sed_mul (sed_mul x y) z) =
  brown1972_sed_trace (sed_mul x (sed_mul y z)).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]] [[b9 b10 b11 b12] [b13 b14 b15 b16]]]
         [[[c1 c2 c3 c4] [c5 c6 c7 c8]] [[c9 c10 c11 c12] [c13 c14 c15 c16]]].
  cbv [brown1972_sed_trace sed_mul
       oct_add oct_mul oct_conj oct_neg
       quat_add quat_mul quat_conj quat_neg
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring.
Qed.

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
Lemma brown1972_sed_trace_add : forall x y : CDSed,
  brown1972_sed_trace (sed_add x y) =
  (brown1972_sed_trace x + brown1972_sed_trace y)%R.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[b1 b2 b3 b4] [b5 b6 b7 b8]]
         [[b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [brown1972_sed_trace sed_add sed_lo oct_add oct_lo qa quat_add].
  ring.
Qed.

Lemma brown1972_sed_trace_neg : forall x : CDSed,
  brown1972_sed_trace (sed_neg x) = (- brown1972_sed_trace x)%R.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [brown1972_sed_trace sed_neg sed_lo oct_lo qa oct_neg quat_neg].
  ring.
Qed.

Lemma brown1972_sed_trace_conj : forall x : CDSed,
  brown1972_sed_trace (sed_conj x) = brown1972_sed_trace x.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [brown1972_sed_trace sed_conj sed_lo oct_lo qa oct_conj quat_conj].
  ring.
Qed.

Lemma brown1972_sed_trace_sub : forall x y : CDSed,
  brown1972_sed_trace (sed_sub x y) =
  (brown1972_sed_trace x - brown1972_sed_trace y)%R.
Proof.
  intros x y.
  unfold sed_sub.
  rewrite brown1972_sed_trace_add.
  rewrite brown1972_sed_trace_neg.
  ring.
Qed.

Lemma brown1972_oct_trace_zero_iff_pure : forall x : CDOct,
  brown1972_oct_trace x = 0%R <-> oct_conj x = oct_neg x.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  split; intro H.
  - cbv [brown1972_oct_trace oct_conj oct_neg oct_lo oct_hi
         quat_conj quat_neg qa qb qc qd] in H |- *.
    apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); lra.
    + apply (f_equal4 mkQuat); lra.
  - apply (f_equal brown1972_oct_trace) in H.
    rewrite brown1972_oct_trace_conj in H.
    rewrite brown1972_oct_trace_neg in H.
    lra.
Qed.

Lemma brown1972_sed_trace_zero_iff_pure : forall x : CDSed,
  brown1972_sed_trace x = 0%R <-> sed_conj x = sed_neg x.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  split; intro H.
  - cbv [brown1972_sed_trace sed_conj sed_neg sed_lo sed_hi
         oct_conj oct_neg oct_lo oct_hi
         quat_conj quat_neg qa qb qc qd] in H |- *.
    apply (f_equal2 mkSed).
    + apply (f_equal2 mkOct).
      * apply (f_equal4 mkQuat); lra.
      * apply (f_equal4 mkQuat); lra.
    + apply (f_equal2 mkOct).
      * apply (f_equal4 mkQuat); lra.
      * apply (f_equal4 mkQuat); lra.
  - apply (f_equal brown1972_sed_trace) in H.
    rewrite brown1972_sed_trace_conj in H.
    rewrite brown1972_sed_trace_neg in H.
    lra.
Qed.

Lemma brown1972_sed_quadratic_identity : forall x : CDSed,
  sed_mul x x =
  sed_add (sed_scale (brown1972_sed_trace x) x)
          (sed_scale (- sed_norm_sq x) sed_one).
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
         [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [brown1972_sed_trace sed_mul sed_add sed_scale sed_norm_sq sed_one
       sed_lo sed_hi sed_conj sed_neg
       oct_norm_sq oct_add oct_scale oct_mul oct_conj oct_neg oct_lo oct_hi
       oct_zero quat_norm_sq quat_add quat_scale quat_mul quat_conj quat_neg
       quat_zero quat_one qa qb qc qd].
  apply (f_equal2 mkSed).
  - apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); ring.
    + apply (f_equal4 mkQuat); ring.
  - apply (f_equal2 mkOct).
    + apply (f_equal4 mkQuat); ring.
    + apply (f_equal4 mkQuat); ring.
Qed.

Section BrownChapterIIIQuadraticConjugationCore.
  Context {A : Type}.
  Variable zero one : A.
  Variable add mul : A -> A -> A.
  Variable neg conj : A -> A.
  Variable scale : R -> A -> A.
  Variable trace norm_sq : A -> R.

  Record Brown1972QuadraticConjugationCoreSurface := {
    brown1972_ch3_qcc_add_zero_left :
      forall x : A, add zero x = x;
    brown1972_ch3_qcc_scale_zero :
      forall x : A, scale 0 x = zero;
    brown1972_ch3_qcc_quadratic :
      forall x : A,
        mul x x =
        add (scale (trace x) x)
            (scale (- norm_sq x) one);
    brown1972_ch3_qcc_trace_zero_iff_pure :
      forall x : A,
        trace x = 0%R <-> conj x = neg x
  }.

  Context (Surf : Brown1972QuadraticConjugationCoreSurface).

  Theorem brown1972_ch3_qcc_pure_square : forall x : A,
    trace x = 0%R ->
    mul x x = scale (- norm_sq x) one.
  Proof.
    intros x Htr.
    rewrite (brown1972_ch3_qcc_quadratic Surf x).
    rewrite Htr.
    rewrite (brown1972_ch3_qcc_scale_zero Surf x).
    apply (brown1972_ch3_qcc_add_zero_left Surf).
  Qed.

  Theorem brown1972_ch3_qcc_conj_neg_pure_square : forall x : A,
    conj x = neg x ->
    mul x x = scale (- norm_sq x) one.
  Proof.
    intros x Hpure.
    apply brown1972_ch3_qcc_pure_square.
    apply (proj2 (brown1972_ch3_qcc_trace_zero_iff_pure Surf x)).
    exact Hpure.
  Qed.
End BrownChapterIIIQuadraticConjugationCore.

Section BrownChapterIIITraceQuadraticConjugation.
  Context {A : Type}.
  Variable zero one : A.
  Variable add mul : A -> A -> A.
  Variable neg conj : A -> A.
  Variable scale : R -> A -> A.
  Variable trace norm_sq : A -> R.

  Record Brown1972TraceQuadraticConjugationSurface := {
    brown1972_ch3_tqc_core :
      Brown1972QuadraticConjugationCoreSurface
        zero one add mul neg conj scale trace norm_sq;
    brown1972_ch3_tqc_t31_i :
      forall x y : A,
        trace (mul x y) = trace (mul y x);
    brown1972_ch3_tqc_t31_ii :
      forall x y z : A,
        trace (mul (mul x y) z) = trace (mul x (mul y z))
  }.

  Context (Surf : Brown1972TraceQuadraticConjugationSurface).

  Theorem brown1972_ch3_tqc_pure_square : forall x : A,
    trace x = 0%R ->
    mul x x = scale (- norm_sq x) one.
  Proof.
    apply (brown1972_ch3_qcc_pure_square
             zero one add mul neg conj scale trace norm_sq
             (brown1972_ch3_tqc_core Surf)).
  Qed.
End BrownChapterIIITraceQuadraticConjugation.

Theorem brown1972_octonion_chapter_iii_pure_square : forall x : CDOct,
  brown1972_oct_trace x = 0%R ->
  oct_mul x x = oct_scale (- oct_norm_sq x) brown1972_oct_one.
Proof.
  intros x Htr.
  rewrite brown1972_oct_quadratic_identity.
  rewrite Htr.
  rewrite oct_scale_zero.
  apply oct_add_zero_left.
Qed.

Record Brown1972ChapterIIIQuadraticConjugationSurface := {
  brown1972_ch3_qc_t31_i :
    forall x y : CDOct,
      brown1972_oct_trace (oct_mul x y) = brown1972_oct_trace (oct_mul y x);
  brown1972_ch3_qc_t31_ii :
    forall x y z : CDOct,
      brown1972_oct_trace (oct_mul (oct_mul x y) z) =
      brown1972_oct_trace (oct_mul x (oct_mul y z));
  brown1972_ch3_qc_quadratic :
    forall x : CDOct,
      oct_mul x x =
      oct_add (oct_scale (brown1972_oct_trace x) x)
              (oct_scale (- oct_norm_sq x) brown1972_oct_one);
  brown1972_ch3_qc_trace_zero_iff_pure :
    forall x : CDOct,
      brown1972_oct_trace x = 0%R <-> oct_conj x = oct_neg x;
  brown1972_ch3_qc_pure_square :
    forall x : CDOct,
      brown1972_oct_trace x = 0%R ->
      oct_mul x x = oct_scale (- oct_norm_sq x) brown1972_oct_one
}.

Definition brown1972_octonion_chapter_iii_quadratic_conjugation_surface :
  Brown1972ChapterIIIQuadraticConjugationSurface.
Proof.
  refine {| brown1972_ch3_qc_t31_i := brown1972_theorem_3_1_i_octonion;
            brown1972_ch3_qc_t31_ii := brown1972_theorem_3_1_ii_octonion;
            brown1972_ch3_qc_quadratic := brown1972_oct_quadratic_identity;
            brown1972_ch3_qc_trace_zero_iff_pure := brown1972_oct_trace_zero_iff_pure;
            brown1972_ch3_qc_pure_square :=
              brown1972_octonion_chapter_iii_pure_square |}.
Defined.

Definition brown1972_octonion_chapter_iii_quadratic_core_surface :
  Brown1972QuadraticConjugationCoreSurface
    oct_zero brown1972_oct_one oct_add oct_mul oct_neg oct_conj oct_scale
    brown1972_oct_trace oct_norm_sq.
Proof.
  refine {| brown1972_ch3_qcc_add_zero_left := oct_add_zero_left;
            brown1972_ch3_qcc_scale_zero := oct_scale_zero;
            brown1972_ch3_qcc_quadratic := brown1972_oct_quadratic_identity;
            brown1972_ch3_qcc_trace_zero_iff_pure := brown1972_oct_trace_zero_iff_pure |}.
Defined.

Definition brown1972_sedenion_chapter_iii_quadratic_core_surface :
  Brown1972QuadraticConjugationCoreSurface
    sed_zero sed_one sed_add sed_mul sed_neg sed_conj sed_scale
    brown1972_sed_trace sed_norm_sq.
Proof.
  refine {| brown1972_ch3_qcc_add_zero_left := sed_add_zero_left;
            brown1972_ch3_qcc_scale_zero := sed_scale_zero;
            brown1972_ch3_qcc_quadratic := brown1972_sed_quadratic_identity;
            brown1972_ch3_qcc_trace_zero_iff_pure := brown1972_sed_trace_zero_iff_pure |}.
Defined.

Definition brown1972_octonion_chapter_iii_trace_quadratic_surface :
  Brown1972TraceQuadraticConjugationSurface
    oct_zero brown1972_oct_one oct_add oct_mul oct_neg oct_conj oct_scale
    brown1972_oct_trace oct_norm_sq.
Proof.
  refine {| brown1972_ch3_tqc_core :=
              brown1972_octonion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_tqc_t31_i := brown1972_theorem_3_1_i_octonion;
            brown1972_ch3_tqc_t31_ii := brown1972_theorem_3_1_ii_octonion |}.
Defined.

Definition brown1972_sedenion_chapter_iii_trace_quadratic_surface :
  Brown1972TraceQuadraticConjugationSurface
    sed_zero sed_one sed_add sed_mul sed_neg sed_conj sed_scale
    brown1972_sed_trace sed_norm_sq.
Proof.
  refine {| brown1972_ch3_tqc_core :=
              brown1972_sedenion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_tqc_t31_i := brown1972_theorem_3_1_i_sedenion;
            brown1972_ch3_tqc_t31_ii := brown1972_theorem_3_1_ii_sedenion |}.
Defined.

Theorem brown1972_sedenion_pure_square : forall x : CDSed,
  brown1972_sed_trace x = 0%R ->
  sed_mul x x = sed_scale (- sed_norm_sq x) sed_one.
Proof.
  apply (brown1972_ch3_qcc_pure_square
           sed_zero sed_one sed_add sed_mul sed_neg sed_conj sed_scale
           brown1972_sed_trace sed_norm_sq
           brown1972_sedenion_chapter_iii_quadratic_core_surface).
Qed.

Record Brown1972ChapterIIIQuadraticCoreLiftSurface := {
  brown1972_ch3_qc_core_oct :
    Brown1972QuadraticConjugationCoreSurface
      oct_zero brown1972_oct_one oct_add oct_mul oct_neg oct_conj oct_scale
      brown1972_oct_trace oct_norm_sq;
  brown1972_ch3_qc_core_sed :
    Brown1972QuadraticConjugationCoreSurface
      sed_zero sed_one sed_add sed_mul sed_neg sed_conj sed_scale
      brown1972_sed_trace sed_norm_sq
}.

Definition brown1972_chapter_iii_quadratic_core_lift_surface :
  Brown1972ChapterIIIQuadraticCoreLiftSurface.
Proof.
  refine {| brown1972_ch3_qc_core_oct :=
              brown1972_octonion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_qc_core_sed :=
              brown1972_sedenion_chapter_iii_quadratic_core_surface |}.
Defined.

Record Brown1972ChapterIIITraceQuadraticLiftSurface := {
  brown1972_ch3_tqc_oct :
    Brown1972TraceQuadraticConjugationSurface
      oct_zero brown1972_oct_one oct_add oct_mul oct_neg oct_conj oct_scale
      brown1972_oct_trace oct_norm_sq;
  brown1972_ch3_tqc_sed :
    Brown1972TraceQuadraticConjugationSurface
      sed_zero sed_one sed_add sed_mul sed_neg sed_conj sed_scale
      brown1972_sed_trace sed_norm_sq
}.

Definition brown1972_chapter_iii_trace_quadratic_lift_surface :
  Brown1972ChapterIIITraceQuadraticLiftSurface.
Proof.
  refine {| brown1972_ch3_tqc_oct :=
              brown1972_octonion_chapter_iii_trace_quadratic_surface;
            brown1972_ch3_tqc_sed :=
              brown1972_sedenion_chapter_iii_trace_quadratic_surface |}.
Defined.

Record Brown1972ChapterIIIStabilizedSurface := {
  brown1972_ch3_stab_oct_norm :
    Brown1972ChapterIIISurface;
  brown1972_ch3_stab_sed_norm :
    Brown1972ChapterIIISedenionSurface;
  brown1972_ch3_stab_oct_trace :
    Brown1972ChapterIIITraceSurface;
  brown1972_ch3_stab_oct_basic :
    Brown1972ChapterIIIBasicConsequencesSurface;
  brown1972_ch3_stab_qc_core_lift :
    Brown1972ChapterIIIQuadraticCoreLiftSurface;
  brown1972_ch3_stab_tqc_lift :
    Brown1972ChapterIIITraceQuadraticLiftSurface
}.

Definition brown1972_chapter_iii_stabilized_surface :
  Brown1972ChapterIIIStabilizedSurface.
Proof.
  refine {| brown1972_ch3_stab_oct_norm :=
              brown1972_octonion_chapter_iii_surface;
            brown1972_ch3_stab_sed_norm :=
              brown1972_sedenion_chapter_iii_surface;
            brown1972_ch3_stab_oct_trace :=
              brown1972_octonion_chapter_iii_trace_surface;
            brown1972_ch3_stab_oct_basic :=
              brown1972_octonion_chapter_iii_basic_consequences_surface;
            brown1972_ch3_stab_qc_core_lift :=
              brown1972_chapter_iii_quadratic_core_lift_surface;
            brown1972_ch3_stab_tqc_lift :=
              brown1972_chapter_iii_trace_quadratic_lift_surface |}.
Defined.

Definition brown1972_quaternion_chapter_iii_quadratic_core_surface :
  Brown1972QuadraticConjugationCoreSurface
    quat_zero quat_one quat_add quat_mul quat_neg quat_conj quat_scale
    brown1972_quat_trace quat_norm_sq.
Proof.
  refine {| brown1972_ch3_qcc_add_zero_left := quat_add_zero_left;
            brown1972_ch3_qcc_scale_zero := brown1972_quat_scale_zero;
            brown1972_ch3_qcc_quadratic :=
              brown1972_quat_quadratic_identity;
            brown1972_ch3_qcc_trace_zero_iff_pure :=
              brown1972_quat_trace_zero_iff_pure |}.
Defined.

Definition brown1972_quaternion_chapter_iii_trace_quadratic_surface :
  Brown1972TraceQuadraticConjugationSurface
    quat_zero quat_one quat_add quat_mul quat_neg quat_conj quat_scale
    brown1972_quat_trace quat_norm_sq.
Proof.
  refine {| brown1972_ch3_tqc_core :=
              brown1972_quaternion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_tqc_t31_i :=
              brown1972_theorem_3_1_i_quaternion;
            brown1972_ch3_tqc_t31_ii :=
              brown1972_theorem_3_1_ii_quaternion |}.
Defined.

Record Brown1972ChapterIIIExtendedTowerSurface := {
  brown1972_ch3_ext_base :
    Brown1972ChapterIIIStabilizedSurface;
  brown1972_ch3_ext_quat_qc_core :
    Brown1972QuadraticConjugationCoreSurface
      quat_zero quat_one quat_add quat_mul quat_neg quat_conj quat_scale
      brown1972_quat_trace quat_norm_sq;
  brown1972_ch3_ext_quat_tqc :
    Brown1972TraceQuadraticConjugationSurface
      quat_zero quat_one quat_add quat_mul quat_neg quat_conj quat_scale
      brown1972_quat_trace quat_norm_sq
}.

Definition brown1972_chapter_iii_extended_tower_surface :
  Brown1972ChapterIIIExtendedTowerSurface.
Proof.
  refine {| brown1972_ch3_ext_base :=
              brown1972_chapter_iii_stabilized_surface;
            brown1972_ch3_ext_quat_qc_core :=
              brown1972_quaternion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_ext_quat_tqc :=
              brown1972_quaternion_chapter_iii_trace_quadratic_surface |}.
Defined.

Record Brown1972ChapterIIISourcedInterface
  (A : Type)
  (zero one : A)
  (add sub mul : A -> A -> A)
  (neg conj : A -> A)
  (scale : R -> A -> A)
  (trace norm_sq : A -> R) := {
  brown1972_ch3_src_tqc :
    Brown1972TraceQuadraticConjugationSurface
      zero one add mul neg conj scale trace norm_sq;
  brown1972_ch3_src_t39_i :
    forall x y : A, norm_sq (mul x (conj y)) = norm_sq (mul x y);
  brown1972_ch3_src_t39_ii :
    forall x y : A, norm_sq (mul (conj x) y) = norm_sq (mul x y);
  brown1972_ch3_src_t39_iii :
    forall x y : A, norm_sq (mul x y) = norm_sq (mul y x);
  brown1972_ch3_src_l310 :
    forall x y : A,
      (norm_sq (add x y) + norm_sq (sub x y))%R =
      (2 * (norm_sq x + norm_sq y))%R
}.

Definition brown1972_quaternion_chapter_iii_sourced_interface :
  Brown1972ChapterIIISourcedInterface
    CDQuat
    quat_zero quat_one quat_add brown1972_quat_sub quat_mul
    quat_neg quat_conj quat_scale brown1972_quat_trace quat_norm_sq.
Proof.
  refine {| brown1972_ch3_src_tqc :=
              brown1972_quaternion_chapter_iii_trace_quadratic_surface;
            brown1972_ch3_src_t39_i :=
              brown1972_theorem_3_9_i_quaternion;
            brown1972_ch3_src_t39_ii :=
              brown1972_theorem_3_9_ii_quaternion;
            brown1972_ch3_src_t39_iii :=
              brown1972_theorem_3_9_iii_quaternion;
            brown1972_ch3_src_l310 :=
              brown1972_lemma_3_10_quaternion |}.
Defined.

Definition brown1972_octonion_chapter_iii_sourced_interface :
  Brown1972ChapterIIISourcedInterface
    CDOct
    oct_zero brown1972_oct_one oct_add oct_sub oct_mul
    oct_neg oct_conj oct_scale brown1972_oct_trace oct_norm_sq.
Proof.
  refine {| brown1972_ch3_src_tqc :=
              brown1972_octonion_chapter_iii_trace_quadratic_surface;
            brown1972_ch3_src_t39_i :=
              brown1972_theorem_3_9_i_octonion;
            brown1972_ch3_src_t39_ii :=
              brown1972_theorem_3_9_ii_octonion;
            brown1972_ch3_src_t39_iii :=
              brown1972_theorem_3_9_iii_octonion;
            brown1972_ch3_src_l310 :=
              brown1972_octonion_lemma_3_10 |}.
Defined.

Definition brown1972_sedenion_chapter_iii_sourced_interface :
  Brown1972ChapterIIISourcedInterface
    CDSed
    sed_zero sed_one sed_add sed_sub sed_mul
    sed_neg sed_conj sed_scale brown1972_sed_trace sed_norm_sq.
Proof.
  refine {| brown1972_ch3_src_tqc :=
              brown1972_sedenion_chapter_iii_trace_quadratic_surface;
            brown1972_ch3_src_t39_i :=
              brown1972_theorem_3_9_i_sedenion;
            brown1972_ch3_src_t39_ii :=
              brown1972_theorem_3_9_ii_sedenion;
            brown1972_ch3_src_t39_iii :=
              brown1972_theorem_3_9_iii_sedenion;
            brown1972_ch3_src_l310 :=
              brown1972_lemma_3_10_sedenion |}.
Defined.

Record Brown1972ChapterIIISourcedQuadraticConjugationInterface
  (A : Type)
  (zero one : A)
  (add sub mul : A -> A -> A)
  (neg conj : A -> A)
  (scale : R -> A -> A)
  (trace norm_sq : A -> R) := {
  brown1972_ch3_sqc_source :
    Brown1972ChapterIIISourcedInterface
      A zero one add sub mul neg conj scale trace norm_sq;
  brown1972_ch3_sqc_core :
    Brown1972QuadraticConjugationCoreSurface
      zero one add mul neg conj scale trace norm_sq;
  brown1972_ch3_sqc_pure_square :
    forall x : A,
      trace x = 0%R ->
      mul x x = scale (- norm_sq x) one
}.

Definition brown1972_quaternion_chapter_iii_sourced_quadratic_interface :
  Brown1972ChapterIIISourcedQuadraticConjugationInterface
    CDQuat
    quat_zero quat_one quat_add brown1972_quat_sub quat_mul
    quat_neg quat_conj quat_scale brown1972_quat_trace quat_norm_sq.
Proof.
  refine {| brown1972_ch3_sqc_source :=
              brown1972_quaternion_chapter_iii_sourced_interface;
            brown1972_ch3_sqc_core :=
              brown1972_quaternion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_sqc_pure_square :=
              brown1972_quaternion_chapter_iii_pure_square |}.
Defined.

Definition brown1972_octonion_chapter_iii_sourced_quadratic_interface :
  Brown1972ChapterIIISourcedQuadraticConjugationInterface
    CDOct
    oct_zero brown1972_oct_one oct_add oct_sub oct_mul
    oct_neg oct_conj oct_scale brown1972_oct_trace oct_norm_sq.
Proof.
  refine {| brown1972_ch3_sqc_source :=
              brown1972_octonion_chapter_iii_sourced_interface;
            brown1972_ch3_sqc_core :=
              brown1972_octonion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_sqc_pure_square :=
              brown1972_octonion_chapter_iii_pure_square |}.
Defined.

Definition brown1972_sedenion_chapter_iii_sourced_quadratic_interface :
  Brown1972ChapterIIISourcedQuadraticConjugationInterface
    CDSed
    sed_zero sed_one sed_add sed_sub sed_mul
    sed_neg sed_conj sed_scale brown1972_sed_trace sed_norm_sq.
Proof.
  refine {| brown1972_ch3_sqc_source :=
              brown1972_sedenion_chapter_iii_sourced_interface;
            brown1972_ch3_sqc_core :=
              brown1972_sedenion_chapter_iii_quadratic_core_surface;
            brown1972_ch3_sqc_pure_square :=
              brown1972_sedenion_pure_square |}.
Defined.

Record Brown1972ChapterIIIReusableAnchorSurface := {
  brown1972_ch3_anchor_base :
    Brown1972ChapterIIIExtendedTowerSurface;
  brown1972_ch3_anchor_quat :
    Brown1972ChapterIIISourcedInterface
      CDQuat
      quat_zero quat_one quat_add brown1972_quat_sub quat_mul
      quat_neg quat_conj quat_scale brown1972_quat_trace quat_norm_sq;
  brown1972_ch3_anchor_oct :
    Brown1972ChapterIIISourcedInterface
      CDOct
      oct_zero brown1972_oct_one oct_add oct_sub oct_mul
      oct_neg oct_conj oct_scale brown1972_oct_trace oct_norm_sq;
  brown1972_ch3_anchor_sed :
    Brown1972ChapterIIISourcedInterface
      CDSed
      sed_zero sed_one sed_add sed_sub sed_mul
      sed_neg sed_conj sed_scale brown1972_sed_trace sed_norm_sq
}.

Definition brown1972_chapter_iii_reusable_anchor_surface :
  Brown1972ChapterIIIReusableAnchorSurface.
Proof.
  refine {| brown1972_ch3_anchor_base :=
              brown1972_chapter_iii_extended_tower_surface;
            brown1972_ch3_anchor_quat :=
              brown1972_quaternion_chapter_iii_sourced_interface;
            brown1972_ch3_anchor_oct :=
              brown1972_octonion_chapter_iii_sourced_interface;
            brown1972_ch3_anchor_sed :=
              brown1972_sedenion_chapter_iii_sourced_interface |}.
Defined.

Record Brown1972ChapterIIIBroaderReusableAnchorSurface := {
  brown1972_ch3_broader_base :
    Brown1972ChapterIIIReusableAnchorSurface;
  brown1972_ch3_broader_quat :
    Brown1972ChapterIIISourcedQuadraticConjugationInterface
      CDQuat
      quat_zero quat_one quat_add brown1972_quat_sub quat_mul
      quat_neg quat_conj quat_scale brown1972_quat_trace quat_norm_sq;
  brown1972_ch3_broader_oct :
    Brown1972ChapterIIISourcedQuadraticConjugationInterface
      CDOct
      oct_zero brown1972_oct_one oct_add oct_sub oct_mul
      oct_neg oct_conj oct_scale brown1972_oct_trace oct_norm_sq;
  brown1972_ch3_broader_sed :
    Brown1972ChapterIIISourcedQuadraticConjugationInterface
      CDSed
      sed_zero sed_one sed_add sed_sub sed_mul
      sed_neg sed_conj sed_scale brown1972_sed_trace sed_norm_sq
}.

Definition brown1972_chapter_iii_broader_reusable_anchor_surface :
  Brown1972ChapterIIIBroaderReusableAnchorSurface.
Proof.
  refine {| brown1972_ch3_broader_base :=
              brown1972_chapter_iii_reusable_anchor_surface;
            brown1972_ch3_broader_quat :=
              brown1972_quaternion_chapter_iii_sourced_quadratic_interface;
            brown1972_ch3_broader_oct :=
              brown1972_octonion_chapter_iii_sourced_quadratic_interface;
            brown1972_ch3_broader_sed :=
              brown1972_sedenion_chapter_iii_sourced_quadratic_interface |}.
Defined.

Theorem brown1972_quaternion_chapter_iii_pure_square_fused : forall x : CDQuat,
  brown1972_quat_trace x = 0%R ->
  quat_mul_fused x x = quat_scale (- quat_norm_sq x) quat_one.
Proof.
  intros x Htr.
  rewrite quat_mul_fused_eq.
  exact (brown1972_quaternion_chapter_iii_pure_square x Htr).
Qed.

Theorem brown1972_octonion_chapter_iii_pure_square_fused : forall x : CDOct,
  brown1972_oct_trace x = 0%R ->
  oct_mul_fused x x = oct_scale (- oct_norm_sq x) brown1972_oct_one.
Proof.
  intros x Htr.
  rewrite oct_mul_fused_eq.
  exact (brown1972_octonion_chapter_iii_pure_square x Htr).
Qed.

Theorem brown1972_sedenion_chapter_iii_pure_square_fused : forall x : CDSed,
  brown1972_sed_trace x = 0%R ->
  sed_mul_fused x x = sed_scale (- sed_norm_sq x) sed_one.
Proof.
  intros x Htr.
  rewrite sed_mul_fused_eq.
  exact (brown1972_sedenion_pure_square x Htr).
Qed.

Theorem brown1972_theorem_3_3_i_octonion_fused : forall x y : CDOct,
  oct_mul_fused (oct_conj x) (oct_mul_fused x y) = oct_scale (oct_norm_sq x) y /\
  oct_mul_fused x (oct_mul_fused (oct_conj x) y) = oct_scale (oct_norm_sq x) y.
Proof.
  intros x y.
  repeat rewrite oct_mul_fused_eq.
  exact (brown1972_theorem_3_3_i_octonion x y).
Qed.

Theorem brown1972_theorem_3_3_ii_octonion_fused : forall x y : CDOct,
  oct_mul_fused (oct_mul_fused y x) (oct_conj x) = oct_scale (oct_norm_sq x) y /\
  oct_mul_fused (oct_mul_fused y (oct_conj x)) x = oct_scale (oct_norm_sq x) y.
Proof.
  intros x y.
  repeat rewrite oct_mul_fused_eq.
  exact (brown1972_theorem_3_3_ii_octonion x y).
Qed.

Theorem brown1972_lemma_3_7_octonion_fused : forall x : CDOct,
  oct_add (oct_mul_fused x x)
          (oct_add (oct_scale (- brown1972_oct_trace x) x)
                   (oct_scale (oct_norm_sq x) brown1972_oct_one)) =
  oct_zero.
Proof.
  intros x.
  rewrite oct_mul_fused_eq.
  exact (brown1972_lemma_3_7_octonion x).
Qed.

Record Brown1972ChapterIIIFusedQuadraticAnchorSurface := {
  brown1972_ch3_fused_base :
    Brown1972ChapterIIIBroaderReusableAnchorSurface;
  brown1972_ch3_fused_quat_bilinear :
    CDFusedBilinearSurface CDQuat quat_add quat_mul quat_mul_fused quat_scale;
  brown1972_ch3_fused_oct_bilinear :
    CDFusedBilinearSurface CDOct oct_add oct_mul oct_mul_fused oct_scale;
  brown1972_ch3_fused_sed_bilinear :
    CDFusedBilinearSurface CDSed sed_add sed_mul sed_mul_fused sed_scale;
  brown1972_ch3_fused_quat_pure_square :
    forall x : CDQuat,
      brown1972_quat_trace x = 0%R ->
      quat_mul_fused x x = quat_scale (- quat_norm_sq x) quat_one;
  brown1972_ch3_fused_oct_pure_square :
    forall x : CDOct,
      brown1972_oct_trace x = 0%R ->
      oct_mul_fused x x = oct_scale (- oct_norm_sq x) brown1972_oct_one;
  brown1972_ch3_fused_sed_pure_square :
    forall x : CDSed,
      brown1972_sed_trace x = 0%R ->
      sed_mul_fused x x = sed_scale (- sed_norm_sq x) sed_one;
  brown1972_ch3_fused_oct_t33_i :
    forall x y : CDOct,
      oct_mul_fused (oct_conj x) (oct_mul_fused x y) =
      oct_scale (oct_norm_sq x) y /\
      oct_mul_fused x (oct_mul_fused (oct_conj x) y) =
      oct_scale (oct_norm_sq x) y;
  brown1972_ch3_fused_oct_t33_ii :
    forall x y : CDOct,
      oct_mul_fused (oct_mul_fused y x) (oct_conj x) =
      oct_scale (oct_norm_sq x) y /\
      oct_mul_fused (oct_mul_fused y (oct_conj x)) x =
      oct_scale (oct_norm_sq x) y;
  brown1972_ch3_fused_oct_l37 :
    forall x : CDOct,
      oct_add (oct_mul_fused x x)
              (oct_add (oct_scale (- brown1972_oct_trace x) x)
                       (oct_scale (oct_norm_sq x) brown1972_oct_one)) =
      oct_zero
}.

Definition brown1972_chapter_iii_fused_quadratic_anchor_surface :
  Brown1972ChapterIIIFusedQuadraticAnchorSurface.
Proof.
  refine {| brown1972_ch3_fused_base :=
              brown1972_chapter_iii_broader_reusable_anchor_surface;
            brown1972_ch3_fused_quat_bilinear := quat_fused_bilinear_surface;
            brown1972_ch3_fused_oct_bilinear := oct_fused_bilinear_surface;
            brown1972_ch3_fused_sed_bilinear := sed_fused_bilinear_surface;
            brown1972_ch3_fused_quat_pure_square :=
              brown1972_quaternion_chapter_iii_pure_square_fused;
            brown1972_ch3_fused_oct_pure_square :=
              brown1972_octonion_chapter_iii_pure_square_fused;
            brown1972_ch3_fused_sed_pure_square :=
              brown1972_sedenion_chapter_iii_pure_square_fused;
            brown1972_ch3_fused_oct_t33_i :=
              brown1972_theorem_3_3_i_octonion_fused;
            brown1972_ch3_fused_oct_t33_ii :=
              brown1972_theorem_3_3_ii_octonion_fused;
            brown1972_ch3_fused_oct_l37 :=
              brown1972_lemma_3_7_octonion_fused |}.
Defined.
