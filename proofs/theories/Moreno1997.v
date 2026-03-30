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
    - Theorem 1.16 concrete arbitrary-a CD instantiation; the abstract
      H_a-orbit/module proof, the canonical octonion profile, and a first
      canonical concrete-to-abstract dimension bridge are now formalized, the
      explicit arbitrary-a builder is landed, and a first non-canonical
      witness package now routes through the paper lane, but the bridge from
      arbitrary concrete V_lambda subspaces to that module interface remains
      open; the next exact substeps are (a) a finite-dim H_a-module witness
      for the arbitrary-a V_lambda lane and (b) a bridge from Moreno's
      concrete V_lambda hypotheses to that witness; the first concrete
      hypothesis package now lands in C1542_MorVlambdaOrbit.v, but it still
      assumes the module-dimension witness rather than deriving it from the
      full V_lambda geometry
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

Theorem Moreno1997_lane_compiles : True.
Proof. exact I. Qed.
