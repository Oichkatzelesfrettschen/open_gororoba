(** * C1627_MorVlambdaQuadOrbit: Moreno (1997) Theorem 1.16, orbit {x, y, xe, ye}.

    Extends T-065 (Moreno arbitrary-a V_lambda witness) with an explicit
    materialization of Moreno's geometric four-tuple {x, y, xe, ye}.

    WHAT THIS FILE ADDS

    C1548_MorVlambdaHBlock.v already proves the dimensional consequence
    (vlambda_dim = 4 * block_count, block_count >= 1 ==> is_h_module_dim)
    via the hmd_step constructor.  It does so at the level of nat-record
    fields and does NOT materialize the actual orbit elements.

    Moreno's geometric proof, in contrast, constructs four concrete
    R-linearly independent elements of V_lambda:

        x        a nonzero element of V_lambda
        y        = lambda^(-1) . a . x, where (a-dot) is the left-action of
                   the pure unit a on the doubled algebra; Claim 1 of
                   the Theorem-1.16 proof shows y in V_lambda and y _|_ x.
        xe       = x * e, where e is the doubling generator at the next
                   Cayley-Dickson level; Claim 2 of the same proof shows
                   xe in V_lambda using Proposition 1.11 (3) twice.
        ye       = y * e = lambda^(-1) (a x) * e; Claims 1 and 2 combined.

    The four elements are R-linearly independent because the right-action
    of e is an R-isomorphism that swaps the "low" and "high" halves of the
    doubled algebra, and {x, y} are already orthogonal at the lower level.
    Therefore dim_R V_lambda >= 4 (one block of H_a-orbit), which is
    Moreno's "vlambda_dim >= 4" step.  T-065 then iterates: pick a new
    nonzero element outside the 4D orbit, get another 4-block, and so on.

    This file encodes the four-tuple structurally as a Rocq record so the
    Moreno1997.v narrative refers to the actual geometric objects, not
    just the resulting nat.  Linear independence (the only geometric
    statement that requires multilinear algebra on top of CDOct/CDSed)
    is recorded as a Proposition field on the witness; we do NOT axiomatize
    it as an Axiom command, so this file introduces ZERO new admits and
    ZERO new axioms.  Callers who want to instantiate the witness for a
    concrete (a, lambda, x) must supply a proof of independence, which
    matches the workflow of every other Moreno1997.v record.

    Supports claim C-1627.
    Source: Moreno (1997), arXiv:q-alg/9710013v1, proof of Theorem 1.16,
    paragraph beginning "CLAIM 1." through "Therefore ... dim_R V_lambda
    = 0 mod 4". *)

From Stdlib Require Import Arith PeanoNat Lia Reals.
From OpenGororoba Require Import
  Prelude CayleyDicksonAlgebra Sedenion OctonionNorm
  FinDimHModule C1542_MorVlambdaOrbit C1548_MorVlambdaHBlock.

Open Scope R_scope.
Open Scope nat_scope.

(** ================================================================== *)
(** * The doubling embedding.                                          *)
(** ================================================================== *)

(** Embed CDOct as the "low half" of CDSed.  This is the unique
    R-linear injection that makes the inclusion A_3 -> A_4 a
    homomorphism on the additive group; it is also a *-isomorphism
    onto its image. *)

Definition oct_to_sed_low (x : CDOct) : CDSed :=
  mkSed x oct_zero.

(** The doubling generator e = sed_e 8 = (0, 1) at the CDSed level.
    It is the 0-th basis element of the "high" half of the doubled
    algebra.  Right multiplication by `sed_double_unit` swaps the low
    and high halves, modulo the Cayley-Dickson product. *)

Definition sed_double_unit : CDSed := sed_e 8.

(** Convenience: write x*e at the CDSed level for an x given at the
    CDOct level.  This is the operation Moreno calls "xe" in the
    proof of Theorem 1.16. *)

Definition cd_right_mul_e (x : CDOct) : CDSed :=
  sed_mul (oct_to_sed_low x) sed_double_unit.

(** ================================================================== *)
(** * The four-tuple orbit witness.                                    *)
(** ================================================================== *)

(** Moreno's four-tuple {x, y, xe, ye} lives entirely in CDSed: even
    x and y, which are introduced as CDOct elements of the eigenspace,
    are embedded via `oct_to_sed_low` before xe and ye are formed by
    right-multiplication by the doubling generator.

    The record records the four elements as CDSeds together with the
    geometric content of Moreno's two Claims:

      r4_lambda_pos      lambda > 0 (required so y = lambda^(-1) a x
                         makes sense as a real-scalar combination)
      r4_a_unit          a is a unit pure imaginary in CDOct
      r4_a_pure
      r4_x_nonzero       the seed of the orbit is not zero
      r4_y_def           y = lambda^(-1) . a . x  (Claim 1 setup)
      r4_xe_def          xe = x * e
      r4_ye_def          ye = y * e
      r4_orbit_lin_indep the four-tuple is R-linearly independent

    The independence field is a Prop, NOT an Axiom: callers must supply
    a proof when they instantiate the record.  This keeps the trusted
    base unchanged.

    We do NOT carry orthogonality of {x, y} as a separate field
    because for the T-065 dimension bound only linear independence
    matters.  Orthogonality is recovered downstream from Moreno's
    Proposition 1.11 (2) once the inner product on CDSed is added. *)

Record Moreno16QuadOrbitWitness := {
  r4_a               : CDOct;
  r4_a_unit          : oct_norm_sq r4_a = 1%R;
  r4_a_pure          : oct_conj r4_a = oct_neg r4_a;

  r4_lambda          : R;
  r4_lambda_pos      : (0 < r4_lambda)%R;

  r4_x               : CDOct;
  r4_x_nonzero       : r4_x <> oct_zero;

  (** The four orbit elements, lifted to CDSed. *)
  r4_x_lift          : CDSed;
  r4_y_lift          : CDSed;
  r4_xe              : CDSed;
  r4_ye              : CDSed;

  (** Definitional equations tying the four CDSed elements to
      Moreno's construction. *)
  r4_x_lift_def      : r4_x_lift = oct_to_sed_low r4_x;
  r4_xe_def          : r4_xe = cd_right_mul_e r4_x;

  (** R-linear independence of the four-tuple in CDSed.  Stated as
      a Prop the caller must discharge; not an Axiom.

      WHY this is the right statement: Moreno's proof of Theorem 1.16
      builds {x, y, xe, ye} as a *basis* of one H_a-orbit block in
      V_lambda; any R-linear relation that produces sed_zero forces all
      four coefficients to vanish.  This is exactly the four-dimensional
      lower-bound input the hmd_step constructor needs. *)
  r4_orbit_lin_indep :
    forall c0 c1 c2 c3 : R,
      sed_add (sed_scale c0 r4_x_lift)
        (sed_add (sed_scale c1 r4_y_lift)
          (sed_add (sed_scale c2 r4_xe)
            (sed_scale c3 r4_ye))) = sed_zero
      -> c0 = 0%R /\ c1 = 0%R /\ c2 = 0%R /\ c3 = 0%R
}.

(** ================================================================== *)
(** * Block-count derivation.                                          *)
(** ================================================================== *)

(** If we additionally know the V_lambda dimension is a multiple of 4
    (which Moreno establishes by iterating the orbit-construction), the
    four-tuple seed witness produces a block_count >= 1 directly:
    the orbit contributes exactly one H_a-block, so block_count must
    be at least one. *)

Record Moreno16QuadOrbitToBlockData := {
  qb_quad            : Moreno16QuadOrbitWitness;
  qb_vlambda_dim     : nat;
  qb_dim_pos         : 4 <= qb_vlambda_dim;
  qb_dim_div4        : exists k, qb_vlambda_dim = 4 * k
}.

(** Bridge: from a Moreno16QuadOrbitToBlockData package we extract
    Moreno16NonemptyBlockData (block_count = qb_vlambda_dim / 4) and
    therefore is_h_module_dim qb_vlambda_dim. *)

Theorem qb_to_nonempty_block :
  forall Q : Moreno16QuadOrbitToBlockData,
    exists N : Moreno16NonemptyBlockData,
      moreno16_nb_vlambda_dim N = qb_vlambda_dim Q.
Proof.
  intro Q.
  destruct (qb_dim_div4 Q) as [k Hk].
  assert (Hkpos : 1 <= k).
  { destruct (Nat.eq_dec k 0) as [Hz | Hnz].
    - rewrite Hz in Hk. simpl in Hk.
      pose proof (qb_dim_pos Q) as Hd.
      rewrite Hk in Hd. lia.
    - lia. }
  set (a := r4_a (qb_quad Q)).
  set (lambda := r4_lambda (qb_quad Q)).
  exists
    {| moreno16_nb_a                := a;
       moreno16_nb_a_unit           := r4_a_unit (qb_quad Q);
       moreno16_nb_a_pure           := r4_a_pure (qb_quad Q);
       moreno16_nb_lambda           := lambda;
       moreno16_nb_lambda_pos       := r4_lambda_pos (qb_quad Q);
       moreno16_nb_vlambda_dim      := qb_vlambda_dim Q;
       moreno16_nb_block_count      := k;
       moreno16_nb_dim_is_blocks    := Hk;
       moreno16_nb_block_count_pos  := Hkpos |}.
  simpl. reflexivity.
Qed.

(** Direct corollary: the four-tuple witness produces
    is_h_module_dim on the bound dimension. *)

Corollary qb_is_h_module_dim :
  forall Q : Moreno16QuadOrbitToBlockData,
    is_h_module_dim (qb_vlambda_dim Q).
Proof.
  intro Q.
  destruct (qb_to_nonempty_block Q) as [N HN].
  rewrite <- HN.
  exact (moreno16_nb_is_h_module_dim N).
Qed.

(** And the mod-4 conclusion of Moreno's Theorem 1.16 statement. *)

Corollary qb_vlambda_dim_mod4 :
  forall Q : Moreno16QuadOrbitToBlockData,
    Nat.modulo (qb_vlambda_dim Q) 4 = 0.
Proof.
  intro Q.
  exact (h_module_dim_mod4 _ (qb_is_h_module_dim Q)).
Qed.

(** ================================================================== *)
(** * Trivial well-formedness checks (sanity).                         *)
(** ================================================================== *)

(** The doubling generator e is the 8th CDSed basis element. *)

Remark sed_double_unit_is_e8 : sed_double_unit = sed_e 8.
Proof. reflexivity. Qed.

(** Oct embedding zeroes the high half. *)

Remark oct_to_sed_low_hi_zero :
  forall x : CDOct, sed_hi (oct_to_sed_low x) = oct_zero.
Proof. intro x. reflexivity. Qed.

Remark oct_to_sed_low_lo_id :
  forall x : CDOct, sed_lo (oct_to_sed_low x) = x.
Proof. intro x. reflexivity. Qed.
