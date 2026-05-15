(** * Moreno1997: Paper-scoped Rocq index for Moreno (1997).

    Source:
      Guillermo Moreno, "The zero divisors of the Cayley-Dickson algebras
      over the real numbers", arXiv:q-alg/9710013v1.

    This file is the Rocq-facing paper lane for Moreno (1997). It does not try
    to duplicate the proofs; instead, it re-exports the theorem files that
    currently carry the formalized Moreno content so downstream developments can
    import the paper as one module.

    Current Moreno companion map:
    - C1539_MorSkewSymm.v  : Proposition 1.7, skew-symmetry of L_x and R_x
    - CDTraceZero.v        : Corollary 1.6 trace-zero route
    - C1538_MorZDSymmetry.v: Corollary 1.6, zero-divisor symmetry
    - C1540_MorHaQuaternion.v : Theorem 1.13 canonical octonion H_a witness
    - C1541_MorDirectSum.v : Theorem 1.15 canonical octonion direct-sum lane
    - C1542_MorVlambdaMod4.v : Theorem 1.16 canonical octonion V_lambda lane
    - C1542_MorVlambdaOrbit.v : Theorem 1.16 abstract H_a-orbit/module lane
    - C1548_MorVlambdaHBlock.v : Theorem 1.16 nonempty H_a-block stepping bridge (T-065)
    - C1627_MorVlambdaQuadOrbit.v : Theorem 1.16 explicit {x, y, xe, ye} four-tuple
      witness (T-065 geometric materialization)
    - C1544_MorKerSDecomp.v : Theorem 2.3 + Corollary 2.4 canonical special-couple lane
    - C1546_MorEigenZD.v : Theorem 2.9 canonical eigenvalue witness lane
    - C1546_MorEigenIFF.v : Theorem 2.9 abstract iff core with explicit side conditions
    - C1547_MorSpecialTripleOct.v : Theorem 2.13 canonical special-triple witness
    - C1543_MorMod4Bound.v : Corollary 1.17, mod-4 and upper-bound lane
    - ZD_Criterion.v       : fused Brown/de Marrais-compatible fundamental ZD criterion
    - CDFusedBilinear.v    : fused 8D/16D bilinear and basis-sign surfaces
    - CDSignBridge.v       : finite sign-table bridge used by the concrete lane
    - CDSignHalfStep.v     : one-step structural lemmas for sign fuel
    - CDSignSection.v      : fixed-dimension sign-table sections

    Remaining Moreno-specific Rocq backlog:
    - Theorem 1.16 arbitrary-a CDOct closure: COMPLETE (C1542_MorVlambdaArbClose.v,
      moreno16_vlambda_arb_trivial).  For any unit imaginary a in CDOct,
      oct_unit_imag_left_sq_neg (C1542_MorHaStabilityArb) gives L_a^2 = -I everywhere,
      so every V_lambda(a) with lambda != 1 is trivially zero; the HModuleFinDim
      instance is hmd_zero (vlambda_dim = 0).  L-1542 closed.
    - Theorem 2.9 full arbitrary-alternative CD discharge; the abstract iff
      core and the canonical witness lane are both formalized, but the bridge
      from Moreno's CD hypotheses to the explicit side conditions remains open

    The executable Rust companion for this paper is `crates/moreno_1997/`. *)

From Stdlib Require Import Reals.
From OpenGororoba Require Export
  Sedenion
  OctonionNorm
  FinDimHModule
  C1539_MorSkewSymm
  CDTraceZero
  C1538_MorZDSymmetry
  C1540_MorHaQuaternion
  C1541_MorDirectSum
  C1542_MorVlambdaMod4
  C1542_MorVlambdaOrbit
  C1548_MorVlambdaHBlock
  C1627_MorVlambdaQuadOrbit
  C1542_MorHaStabilityArb
  C1542_MorVlambdaArbClose
  C1544_MorKerSDecomp
  C1546_MorEigenZD
  C1546_MorEigenIFF
  C1547_MorSpecialTripleOct
  C1543_MorMod4Bound
  ZD_Criterion
  CDFusedBilinear
  CDSignBridge
  CDSignHalfStep
  CDSignSection.

Open Scope nat_scope.

(** The first concrete-to-abstract Moreno 1.16 bridge.

    The full arbitrary-a instantiation is still open, but the canonical A_3
    witness now routes through the abstract H-module dimension theorem rather
    than only through the direct 4 = 0 mod 4 arithmetic close-out. *)

Theorem Moreno1997_theorem_1_16_canonical_v1_bridge :
  is_h_module_dim moreno15_ker_t_tilde_dim /\
  Nat.modulo moreno15_ker_t_tilde_dim 4 = 0.
Proof.
  split.
  - unfold moreno15_ker_t_tilde_dim.
    exact moreno16_hmd_4.
  - apply moreno16_orbit_dim_mod4.
    unfold moreno15_ker_t_tilde_dim.
    exact moreno16_hmd_4.
Qed.

Theorem Moreno1997_theorem_1_16_canonical_nonunit_bridge :
  is_h_module_dim moreno15_vlambda_total_dim /\
  Nat.modulo moreno15_vlambda_total_dim 4 = 0.
Proof.
  split.
  - unfold moreno15_vlambda_total_dim.
    exact hmd_zero.
  - apply moreno16_orbit_dim_mod4.
    unfold moreno15_vlambda_total_dim.
    exact hmd_zero.
Qed.

Corollary Moreno1997_theorem_1_16_canonical_bridge :
  (is_h_module_dim moreno15_ker_t_tilde_dim /\
   Nat.modulo moreno15_ker_t_tilde_dim 4 = 0) /\
  (is_h_module_dim moreno15_vlambda_total_dim /\
   Nat.modulo moreno15_vlambda_total_dim 4 = 0).
Proof.
  split.
  - exact Moreno1997_theorem_1_16_canonical_v1_bridge.
  - exact Moreno1997_theorem_1_16_canonical_nonunit_bridge.
Qed.

Theorem Moreno1997_theorem_1_16_concrete_witness_bridge :
  forall W : Moreno16ConcreteVlambdaWitness,
    Nat.modulo (moreno16_concrete_vlambda_dim W) 4 = 0.
Proof.
  intro W.
  exact (moreno16_concrete_witness_dim_mod4 W).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_hypotheses_bridge :
  forall H : Moreno16ArbitraryAConcreteHypotheses,
    Nat.modulo (moreno16_concrete_vlambda_dim (moreno16_concrete_witness H)) 4 = 0.
Proof.
  intro H.
  exact (moreno16_arbitrary_a_dim_mod4 H).
Qed.

Theorem Moreno1997_theorem_1_16_canonical_arbitrary_a_bridge :
  Nat.modulo
    (moreno16_concrete_vlambda_dim
       (moreno16_concrete_witness moreno16_canonical_arbitrary_a_hypotheses))
    4 = 0.
Proof.
  exact
    (moreno16_arbitrary_a_dim_mod4 moreno16_canonical_arbitrary_a_hypotheses).
Qed.

Theorem Moreno1997_theorem_1_16_explicit_hypotheses_builder_bridge :
  forall (a : CDOct)
         (Ha_unit : oct_norm_sq a = 1%R)
         (Ha_pure : oct_conj a = oct_neg a)
         (W : Moreno16ConcreteVlambdaWitness),
    Nat.modulo
      (moreno16_concrete_vlambda_dim
         (moreno16_concrete_witness
            (moreno16_build_arbitrary_a_concrete_hypotheses a Ha_unit Ha_pure W))) 4 = 0.
Proof.
  intros a Ha_unit Ha_pure W.
  exact
    (moreno16_arbitrary_a_dim_mod4
       (moreno16_build_arbitrary_a_concrete_hypotheses a Ha_unit Ha_pure W)).
Qed.

Theorem Moreno1997_theorem_1_16_noncanonical_witness_bridge :
  forall W : Moreno16ArbitraryAVlambdaWitness,
    Nat.modulo (moreno16_arbitrary_vlambda_dim W) 4 = 0.
Proof.
  intro W.
  exact (moreno16_arbitrary_witness_dim_mod4 W).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_concrete_to_witness_bridge :
  forall H : Moreno16ArbitraryAConcreteVlambdaHypotheses,
    exists W : Moreno16ArbitraryAVlambdaWitness,
      moreno16_arbitrary_a W = moreno16_concrete_hyp_a H /\
      moreno16_arbitrary_lambda W = moreno16_concrete_hyp_lambda H /\
      moreno16_arbitrary_vlambda_dim W = moreno16_concrete_hyp_vlambda_dim H /\
      Nat.modulo (moreno16_arbitrary_vlambda_dim W) 4 = 0.
Proof.
  intro H.
  exists (moreno16_build_arbitrary_a_witness_from_concrete_hypotheses H).
  repeat split; try reflexivity.
  exact
    (moreno16_arbitrary_witness_dim_mod4
       (moreno16_build_arbitrary_a_witness_from_concrete_hypotheses H)).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_concrete_hypotheses_bridge :
  forall H : Moreno16ArbitraryAConcreteVlambdaHypotheses,
    Nat.modulo (moreno16_concrete_hyp_vlambda_dim H) 4 = 0.
Proof.
  intro H.
  exact (moreno16_concrete_hypotheses_dim_mod4 H).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_geometry_to_witness_bridge :
  forall G : Moreno16ArbitraryAGeometricVlambdaData,
    exists W : Moreno16ArbitraryAVlambdaWitness,
      moreno16_arbitrary_a W = moreno16_geom_a G /\
      moreno16_arbitrary_lambda W = moreno16_geom_lambda G /\
      moreno16_arbitrary_vlambda_dim W = moreno16_geom_vlambda_dim G /\
      Nat.modulo (moreno16_arbitrary_vlambda_dim W) 4 = 0.
Proof.
  intro G.
  exists (moreno16_geometry_to_witness G).
  repeat split; try reflexivity.
  exact (moreno16_geometry_dim_mod4 G).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_geometry_bridge :
  forall G : Moreno16ArbitraryAGeometricVlambdaData,
    Nat.modulo (moreno16_geom_vlambda_dim G) 4 = 0.
Proof.
  intro G.
  exact (moreno16_geometry_dim_mod4 G).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_concrete_geometry_to_witness_bridge :
  forall G : Moreno16ArbitraryAConcreteVlambdaGeometry,
    exists W : Moreno16ArbitraryAVlambdaWitness,
      moreno16_arbitrary_a W = moreno16_concrete_geom_a G /\
      moreno16_arbitrary_lambda W = moreno16_concrete_geom_lambda G /\
      moreno16_arbitrary_vlambda_dim W = 0%nat /\
      Nat.modulo (moreno16_arbitrary_vlambda_dim W) 4 = 0 /\
      (forall y : CDOct,
        oct_mul (moreno16_concrete_geom_a G)
          (oct_mul (moreno16_concrete_geom_a G) y) =
          oct_scale
            (-(moreno16_concrete_geom_lambda G *
               moreno16_concrete_geom_lambda G)) y ->
        y = oct_zero).
Proof.
  intro G.
  exists (moreno16_concrete_geometry_to_witness G).
  repeat split; try reflexivity.
  intros y Hvl.
  exact (moreno16_concrete_geometry_members_zero G y Hvl).
Qed.

Theorem Moreno1997_theorem_1_16_arbitrary_a_concrete_geometry_derived_bridge :
  forall (a : CDOct) (lambda : R),
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    (0 < lambda)%R ->
    lambda <> 1%R ->
    exists W : Moreno16ArbitraryAVlambdaWitness,
      moreno16_arbitrary_a W = a /\
      moreno16_arbitrary_lambda W = lambda /\
      moreno16_arbitrary_vlambda_dim W = 0%nat /\
      Nat.modulo (moreno16_arbitrary_vlambda_dim W) 4 = 0.
Proof.
  intros a lambda Ha_unit Ha_pure Hpos Hne.
  set (G :=
         moreno16_build_arbitrary_a_concrete_vlambda_geometry
           a Ha_unit Ha_pure lambda Hpos Hne).
  exists (moreno16_concrete_geometry_to_witness G).
  repeat split; reflexivity.
Qed.

(** ** Full paper-facing arbitrary-a bridge for Theorem 1.16.

    The statements below are the paper-level formulations of Theorem 1.16:
    if V_lambda carries a finite-dimensional H_a-module structure (captured
    by the is_h_module_dim inductive predicate), then its R-dimension is
    divisible by 4 and hence congruent to 0 mod 4.

    Three equivalent presentations are provided:

    1. A universal proposition over any (vl_dim, lambda) pair where
       is_h_module_dim vl_dim holds -- the most directly paper-faithful form.

    2. A Section-parameterized version, useful for downstream theories that
       want to introduce vl_dim and the H-module predicate as section variables.

    3. A Module functor version that instantiates the MorenoVlambdaDim
       functor from C1542_MorVlambdaOrbit, giving a self-contained module
       for any concrete M : MorenoVlambdaModule.

    These complete the NA-038 bridge: the abstract H_a-orbit/module lane from
    C1542_MorVlambdaOrbit is now exposed under the paper-facing Moreno1997
    namespace. *)

(** 1. Universal proposition form. *)
Theorem Moreno1997_theorem_1_16_abstract :
  forall (vl_dim : nat) (lambda : R),
    is_h_module_dim vl_dim ->
    (0 < lambda)%R ->
    Nat.modulo vl_dim 4 = 0.
Proof.
  intros vl_dim lambda Hmod _Hlambda.
  exact (moreno16_orbit_dim_mod4 vl_dim Hmod).
Qed.

Corollary Moreno1997_theorem_1_16_abstract_div4 :
  forall (vl_dim : nat) (lambda : R),
    is_h_module_dim vl_dim ->
    (0 < lambda)%R ->
    exists k : nat, vl_dim = 4 * k.
Proof.
  intros vl_dim lambda Hmod _Hlambda.
  exact (moreno16_orbit_dim_div4 vl_dim Hmod).
Qed.

(** 2. Section form: introduces vl_dim and the H-module hypothesis as
       section variables so downstream developments can use them without
       carrying the universally-quantified form. *)
Section Moreno1997Theorem116.
  Variable vl_dim : nat.
  Variable lambda  : R.
  Hypothesis Hmod      : is_h_module_dim vl_dim.
  Hypothesis Hlambda   : (0 < lambda)%R.

  Theorem moreno1997_vlambda_dim_mod4 : Nat.modulo vl_dim 4 = 0.
  Proof. exact (moreno16_orbit_dim_mod4 vl_dim Hmod). Qed.

  Theorem moreno1997_vlambda_dim_div4 : exists k : nat, vl_dim = 4 * k.
  Proof. exact (moreno16_orbit_dim_div4 vl_dim Hmod). Qed.
End Moreno1997Theorem116.

(** 3. Module functor form: for any M : MorenoVlambdaModule, the functor
       derives dim mod 4 = 0 directly from M.v_dim_is_h_module_dim.
       This is the closest formalization of Moreno's actual proof shape. *)
Module Moreno1997Theorem116Functor (M : MorenoVlambdaModule).
  Module Import Dim := MorenoVlambdaDim M.

  Theorem moreno1997_theorem_1_16 : Nat.modulo M.v_dim 4 = 0.
  Proof. exact moreno16_vlambda_dim_mod4. Qed.

  Corollary moreno1997_theorem_1_16_div4 : exists k : nat, M.v_dim = 4 * k.
  Proof. exact moreno16_vlambda_dim_div4. Qed.
End Moreno1997Theorem116Functor.

(** ** T-065 bridge: nonempty H_a-block stepping form of Theorem 1.16.

    For any nonempty block package N (block_count >= 1), the hmd_step
    constructor derives is_h_module_dim directly from the two orbit-stepping
    prerequisites.  This is the closest Rocq formalization of Moreno's
    proof narrative for Theorem 1.16. *)

Theorem Moreno1997_theorem_1_16_nonempty_block_step_bridge :
  forall N : Moreno16NonemptyBlockData,
    4 <= moreno16_nb_vlambda_dim N /\
    is_h_module_dim (moreno16_nb_vlambda_dim N - 4) /\
    is_h_module_dim (moreno16_nb_vlambda_dim N) /\
    Nat.modulo (moreno16_nb_vlambda_dim N) 4 = 0.
Proof.
  intro N.
  refine (conj _ (conj _ (conj _ _))).
  - exact (moreno16_nb_lower_bound N).
  - exact (moreno16_nb_complement_hmd N).
  - exact (moreno16_nb_is_h_module_dim N).
  - exact (moreno16_nb_vlambda_dim_mod4 N).
Qed.

(** ** L-1542 closure: arbitrary-a CDOct V_lambda triviality.
    moreno16_vlambda_arb_trivial proves that for any doubly-pure unit-norm
    a and any lambda > 0 with lambda != 1, the eigenspace V_lambda(a) in CDOct
    contains only the zero vector.  The dim-0 trivial module closes L-1542. *)

Theorem Moreno1997_theorem_1_16_arb_a_cdo_close :
  forall a y : CDOct, forall lambda : R,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    (0 < lambda)%R ->
    lambda <> 1%R ->
    oct_mul a (oct_mul a y) = oct_scale (-(lambda * lambda)) y ->
    y = oct_zero.
Proof.
  intros a y lambda Hu Hp Hpos Hne Hvl.
  exact (moreno16_vlambda_arb_trivial a y lambda Hu Hp Hpos Hne Hvl).
Qed.

Corollary Moreno1997_theorem_1_16_arb_a_cdo_dim_zero :
  forall a : CDOct, forall lambda : R,
    oct_norm_sq a = 1%R ->
    oct_conj a = oct_neg a ->
    (0 < lambda)%R ->
    lambda <> 1%R ->
    is_h_module_dim 0%nat /\ Nat.modulo 0 4 = 0.
Proof.
  intros. exact (moreno16_arb_close_bridge a lambda H H0 H1 H2).
Qed.

Theorem Moreno1997_lane_compiles : True.
Proof. exact I. Qed.
