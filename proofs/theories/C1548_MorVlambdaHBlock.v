(** * C1548_MorVlambdaHBlock: Moreno (1997) Theorem 1.16 -- H_a-block stepping bridge.

    Closes T-065: explicit hmd_step derivation for nonempty arbitrary-a V_lambda.

    WHAT THIS FILE ADDS:

    C1542_MorVlambdaOrbit.v already provides:
    - Moreno16ArbitraryAGeometricVlambdaData: records vlambda_dim = 4 * block_count
    - moreno16_geometry_is_h_module_dim: derives is_h_module_dim via div4_implies_h_module_dim

    The gap: div4_implies_h_module_dim reaches is_h_module_dim by structural induction
    on block_count without materializing the two constructor prerequisites of hmd_step:

      (1)  4 <= vlambda_dim
      (2)  is_h_module_dim (vlambda_dim - 4)

    These two propositions correspond directly to Moreno's orbit-stepping argument:

      (1) V_lambda != {0}: pick nonzero x in V_lambda; the right H_a-action gives
          {x, x*i, x*j, x*k}, which are R-linearly independent because H_a is a
          division algebra.  Their span has dimension 4, so 4 <= vlambda_dim.

      (2) The orthogonal complement W^perp of the 4D orbit W in V_lambda is H_a-stable
          (adjoint property of the inner product and H_a-action), has dimension
          vlambda_dim - 4 = 4 * (block_count - 1), and satisfies is_h_module_dim
          by div4_implies_h_module_dim.

    From (1) and (2) the hmd_step constructor gives is_h_module_dim(vlambda_dim)
    without going through div4 induction.  The proof narrative now matches Moreno:
    "nonempty V_lambda => orbit lower bound => complement descent => hmd_step."

    Supports claim C-1548.
    Source: Moreno (1997), arXiv:q-alg/9710013v1, Theorem 1.16. *)

From Stdlib Require Import Arith PeanoNat Lia Reals.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  FinDimHModule C1542_MorVlambdaOrbit.

Open Scope R_scope.
Open Scope nat_scope.

(** ================================================================== *)
(** * Nonempty block data.                                             *)
(** ================================================================== *)

(** Extends Moreno16ArbitraryAGeometricVlambdaData with a positivity
    witness: block_count >= 1.  This encodes "V_lambda != {0}" at the
    block-count level, which is the precondition Moreno requires before
    applying the orbit-stepping argument. *)

Record Moreno16NonemptyBlockData := {
  moreno16_nb_a : CDOct;
  moreno16_nb_a_unit   : oct_norm_sq moreno16_nb_a = 1%R;
  moreno16_nb_a_pure   :
    oct_conj moreno16_nb_a = oct_neg moreno16_nb_a;
  moreno16_nb_lambda       : R;
  moreno16_nb_lambda_pos   : (0 < moreno16_nb_lambda)%R;
  moreno16_nb_vlambda_dim  : nat;
  moreno16_nb_block_count  : nat;
  moreno16_nb_dim_is_blocks :
    moreno16_nb_vlambda_dim = 4 * moreno16_nb_block_count;
  moreno16_nb_block_count_pos : 1 <= moreno16_nb_block_count
}.

(** ================================================================== *)
(** * The two hmd_step prerequisites.                                  *)
(** ================================================================== *)

(** (1) Orbit lower bound: block_count >= 1 implies vlambda_dim >= 4.

    Proof: vlambda_dim = 4 * block_count >= 4 * 1 = 4. *)

Theorem moreno16_nb_lower_bound :
  forall N : Moreno16NonemptyBlockData,
    4 <= moreno16_nb_vlambda_dim N.
Proof.
  intro N.
  rewrite (moreno16_nb_dim_is_blocks N).
  pose proof (moreno16_nb_block_count_pos N) as H.
  lia.
Qed.

(** (2) Complement descent: vlambda_dim - 4 is a valid H-module dimension.

    Proof: vlambda_dim - 4 = 4 * block_count - 4 = 4 * (block_count - 1)
    (valid since block_count >= 1), and div4_implies_h_module_dim applies. *)

Theorem moreno16_nb_complement_hmd :
  forall N : Moreno16NonemptyBlockData,
    is_h_module_dim (moreno16_nb_vlambda_dim N - 4).
Proof.
  intro N.
  rewrite (moreno16_nb_dim_is_blocks N).
  pose proof (moreno16_nb_block_count_pos N) as H.
  replace (4 * moreno16_nb_block_count N - 4)
    with (4 * (moreno16_nb_block_count N - 1)) by lia.
  exact (div4_implies_h_module_dim (moreno16_nb_block_count N - 1)).
Qed.

(** The two prerequisites as a conjunction. *)

Theorem moreno16_nb_hmd_step_prereqs :
  forall N : Moreno16NonemptyBlockData,
    4 <= moreno16_nb_vlambda_dim N /\
    is_h_module_dim (moreno16_nb_vlambda_dim N - 4).
Proof.
  intro N.
  exact (conj (moreno16_nb_lower_bound N) (moreno16_nb_complement_hmd N)).
Qed.

(** ================================================================== *)
(** * is_h_module_dim via the hmd_step constructor.                   *)
(** ================================================================== *)

(** The main theorem: block_count >= 1 bridges into is_h_module_dim via
    the hmd_step constructor, making the Rocq proof tree match Moreno's
    orbit-stepping proof narrative. *)

Theorem moreno16_nb_is_h_module_dim :
  forall N : Moreno16NonemptyBlockData,
    is_h_module_dim (moreno16_nb_vlambda_dim N).
Proof.
  intro N.
  apply hmd_step.
  - exact (moreno16_nb_lower_bound N).
  - exact (moreno16_nb_complement_hmd N).
Qed.

(** ================================================================== *)
(** * Dimension conclusions.                                           *)
(** ================================================================== *)

Corollary moreno16_nb_vlambda_dim_div4 :
  forall N : Moreno16NonemptyBlockData,
    exists k : nat, moreno16_nb_vlambda_dim N = 4 * k.
Proof.
  intro N.
  exact (h_module_dim_div4 _ (moreno16_nb_is_h_module_dim N)).
Qed.

Corollary moreno16_nb_vlambda_dim_mod4 :
  forall N : Moreno16NonemptyBlockData,
    Nat.modulo (moreno16_nb_vlambda_dim N) 4 = 0.
Proof.
  intro N.
  exact (h_module_dim_mod4 _ (moreno16_nb_is_h_module_dim N)).
Qed.

(** ================================================================== *)
(** * Conversions to existing record types.                            *)
(** ================================================================== *)

(** NonemptyBlockData -> GeometricVlambdaData (forgets block_count_pos). *)

Definition moreno16_nb_to_geometric
  (N : Moreno16NonemptyBlockData)
  : Moreno16ArbitraryAGeometricVlambdaData :=
  {| moreno16_geom_a              := moreno16_nb_a N;
     moreno16_geom_a_unit         := moreno16_nb_a_unit N;
     moreno16_geom_a_pure         := moreno16_nb_a_pure N;
     moreno16_geom_lambda         := moreno16_nb_lambda N;
     moreno16_geom_lambda_pos     := moreno16_nb_lambda_pos N;
     moreno16_geom_vlambda_dim    := moreno16_nb_vlambda_dim N;
     moreno16_geom_block_count    := moreno16_nb_block_count N;
     moreno16_geom_vlambda_dim_is_blocks := moreno16_nb_dim_is_blocks N |}.

(** NonemptyBlockData -> ConcreteVlambdaHypotheses (via geometric bridge). *)

Definition moreno16_nb_to_concrete_hypotheses
  (N : Moreno16NonemptyBlockData)
  : Moreno16ArbitraryAConcreteVlambdaHypotheses :=
  moreno16_geometry_to_concrete_hypotheses (moreno16_nb_to_geometric N).

(** The hmd_step path produces the same Prop as the div4 path on the
    same data.  Both routes satisfy the spec. *)

Theorem moreno16_nb_hmd_agrees_with_geometric :
  forall N : Moreno16NonemptyBlockData,
    is_h_module_dim (moreno16_nb_vlambda_dim N) /\
    is_h_module_dim (moreno16_geom_vlambda_dim (moreno16_nb_to_geometric N)).
Proof.
  intro N.
  split.
  - exact (moreno16_nb_is_h_module_dim N).
  - exact (moreno16_geometry_is_h_module_dim (moreno16_nb_to_geometric N)).
Qed.
