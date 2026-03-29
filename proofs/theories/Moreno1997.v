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
      H_a-orbit/module proof and the canonical octonion profile are both
      formalized, but the bridge from concrete V_lambda subspaces to that
      module interface remains open
    - Theorem 2.9 full arbitrary-alternative CD discharge; the abstract iff
      core and the canonical witness lane are both formalized, but the bridge
      from Moreno's CD hypotheses to the explicit side conditions remains open

    The executable Rust companion for this paper is `crates/moreno_1997/`. *)

From OpenGororoba Require Export
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

Theorem Moreno1997_lane_compiles : True.
Proof. exact I. Qed.
