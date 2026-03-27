(** * Schafer1954: Paper-scoped Rocq index for Schafer (1954).

    Source:
      R.D. Schafer, "On the algebras formed by the Cayley-Dickson process,"
      American Journal of Mathematics 76(2), 1954, pp. 435-446.

    This file is the Rocq-facing paper lane for the 1954 Schafer follow-up.
    It packages the part of the paper that is already represented by existing
    theorem files, and records the later derivation-algebra section as an
    explicit remaining gap instead of leaving the paper only as "source on disk".

    Current chapter / section surfacing:
    - Section 1, pp. 436-440:
      flexibility, degree-two structure, and power-associative behavior.
      Current Rocq landing: `CDPowerAssociative.v`.
    - Section 1, pp. 439-440, Lemma 4:
      basis-element alternative behavior. Current Rocq support exists in the
      local basis-law tower, but there is not yet a dedicated Schafer-numbered
      theorem lane for those statements.
    - Global property-loss consequences across low dimensions:
      current Rocq landing: `CDPropertyTower.v`.
    - Theorem 2, p. 441:
      abstract derivation extension map to the doubled algebra.
    - Theorem 4, pp. 445-446:
      derivation algebras `D(M_t) = D(C)` with 14-dimensional type-G surface.
      The extension map is now abstractly formalized, but the equality and
      type-G identification still remain open as dedicated paper theorem lanes.

    Current Rocq companion map:
    - CDPowerAssociative.v : flexibility / power-associativity anchors
    - CDPropertyTower.v    : low-dimensional property-loss summary

    Remaining Schafer 1954 backlog:
    - dedicated basis-law lane matching Lemma 4 and the restricted basis claims
    - Theorem 3 restriction/uniqueness lane in paper order
    - explicit derivation-algebra equality formalization for Theorem 4
    - a tighter bridge from the G2 support files to Schafer's derivation language
*)

From Stdlib Require Import Logic.FunctionalExtensionality.

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDNegLemmas CDLinearLemmas CDConjAntimorph CDSignBridge.
From OpenGororoba Require Export
  CDPowerAssociative
  CDPropertyTower
  G2StabilizerDimension
  G2OctonionAutomorphisms.

(** Schafer 1954, Section 1: flexibility anchor at dim 4. *)
Theorem schafer1954_quaternion_flexibility :
  forall x y : CDQuat,
  quat_mul (quat_mul x y) x = quat_mul x (quat_mul y x).
Proof.
  exact quat_flexible.
Qed.

(** Schafer 1954, Section 1: flexibility anchor at dim 8. *)
Theorem schafer1954_octonion_flexibility :
  forall x y : CDOct,
  oct_mul (oct_mul x y) x = oct_mul x (oct_mul y x).
Proof.
  exact oct_flexible.
Qed.

(** Schafer 1954, Section 1: power-associative anchor at dim 8. *)
Theorem schafer1954_octonion_power_associativity :
  forall x : CDOct,
  oct_mul (oct_mul x x) x = oct_mul x (oct_mul x x).
Proof.
  exact oct_third_power.
Qed.

(** Schafer 1954, degree-two / quadratic identity anchor at dim 16. *)
Theorem schafer1954_sedenion_norm_quadratic_anchor :
  forall x : CDSed,
  sed_mul x (sed_conj x) =
  mkSed (mkOct (mkQuat (sed_norm_sq x) 0 0 0) quat_zero) oct_zero.
Proof.
  exact sed_conj_norm.
Qed.

(** Property-loss summary used downstream in the pre-Moreno tower. *)
Theorem schafer1954_property_tower_surface :
  commutativity_lost_at_4 /\
  associativity_lost_at_8 /\
  division_lost_at_16.
Proof.
  exact cd_property_tower.
Qed.

Section Schafer1954AbstractDerivations.
  Variable U : Type.
  Variable uadd umul : U -> U -> U.
  Variable uscale : R -> U -> U.
  Variable uconj : U -> U.
  Variable t : R.

  Hypothesis uadd_assoc : forall x y z : U,
    uadd x (uadd y z) = uadd (uadd x y) z.
  Hypothesis uadd_comm : forall x y : U,
    uadd x y = uadd y x.
  Hypothesis uscale_add_distr : forall r : R, forall x y : U,
    uscale r (uadd x y) = uadd (uscale r x) (uscale r y).

  Record Schafer1954IsDerivation (D : U -> U) : Prop := {
    s54_deriv_add :
      forall x y : U, D (uadd x y) = uadd (D x) (D y);
    s54_deriv_scale :
      forall r : R, forall x : U, D (uscale r x) = uscale r (D x);
    s54_deriv_mul :
      forall x y : U, D (umul x y) = uadd (umul (D x) y) (umul x (D y));
    s54_deriv_conj :
      forall x : U, D (uconj x) = uconj (D x);
  }.

  Definition s54_pair : Type := (U * U)%type.

  Definition s54_pair_add (p q : s54_pair) : s54_pair :=
    (uadd (fst p) (fst q), uadd (snd p) (snd q)).

  Definition s54_pair_scale (r : R) (p : s54_pair) : s54_pair :=
    (uscale r (fst p), uscale r (snd p)).

  Definition s54_pair_mul (p q : s54_pair) : s54_pair :=
    (uadd (umul (fst p) (fst q))
          (uscale t (umul (uconj (snd q)) (snd p))),
     uadd (umul (snd q) (fst p))
          (umul (snd p) (uconj (fst q)))).

  Definition s54_pair_extend (A : U -> U) (p : s54_pair) : s54_pair :=
    (A (fst p), A (snd p)).

  Record Schafer1954IsPairDerivation (D : s54_pair -> s54_pair) : Prop := {
    s54_pair_deriv_add :
      forall p q : s54_pair,
        D (s54_pair_add p q) = s54_pair_add (D p) (D q);
    s54_pair_deriv_scale :
      forall r : R, forall p : s54_pair,
        D (s54_pair_scale r p) = s54_pair_scale r (D p);
    s54_pair_deriv_mul :
      forall p q : s54_pair,
        D (s54_pair_mul p q) =
        s54_pair_add (s54_pair_mul (D p) q)
                     (s54_pair_mul p (D q));
  }.

  Lemma s54_uadd_shuffle4 :
    forall a b c d : U,
      uadd (uadd a b) (uadd c d) = uadd (uadd a c) (uadd b d).
  Proof.
    intros a b c d.
    rewrite <- uadd_assoc.
    rewrite (uadd_assoc b c d).
    rewrite (uadd_comm b c).
    rewrite <- uadd_assoc.
    rewrite <- uadd_assoc.
    rewrite uadd_assoc.
    reflexivity.
  Qed.

  Lemma s54_uadd_cross_swap :
    forall a b c d : U,
      uadd (uadd a d) (uadd b c) = uadd (uadd a c) (uadd b d).
  Proof.
    intros a b c d.
    rewrite (uadd_comm b c).
    rewrite s54_uadd_shuffle4.
    rewrite (uadd_comm d b).
    reflexivity.
  Qed.

  Lemma s54_uadd_rotate :
    forall a b c d : U,
      uadd (uadd a b) (uadd c d) = uadd (uadd b c) (uadd a d).
  Proof.
    intros a b c d.
    rewrite (uadd_comm a b).
    rewrite s54_uadd_shuffle4.
    reflexivity.
  Qed.

  Theorem schafer1954_theorem2_extension_map :
    forall A : U -> U,
      Schafer1954IsDerivation A ->
      Schafer1954IsPairDerivation (s54_pair_extend A).
  Proof.
    intros A HA.
    destruct HA as [Hadd Hscale Hmul Hconj].
    constructor.
    - intros [x y] [z w].
      unfold s54_pair_extend, s54_pair_add.
      simpl.
      f_equal.
      + exact (Hadd x z).
      + exact (Hadd y w).
    - intros r [x y].
      unfold s54_pair_extend, s54_pair_scale.
      simpl.
      f_equal.
      + exact (Hscale r x).
      + exact (Hscale r y).
    - intros [x y] [z w].
      unfold s54_pair_extend, s54_pair_mul, s54_pair_add.
      simpl.
      f_equal.
      + rewrite Hadd.
        rewrite Hmul.
        rewrite Hscale.
        rewrite Hmul.
        rewrite Hconj.
        rewrite uscale_add_distr.
        rewrite s54_uadd_shuffle4.
        rewrite s54_uadd_cross_swap.
        reflexivity.
      + rewrite Hadd.
        rewrite Hmul.
        rewrite Hmul.
        rewrite Hconj.
        rewrite s54_uadd_rotate.
        reflexivity.
  Qed.

  Record Schafer1954Theorem2ExtensionSurface (A : U -> U) : Prop := {
    s54_t2_extension_formula :
      forall x y : U, s54_pair_extend A (x, y) = (A x, A y);
    s54_t2_extension_is_derivation :
      Schafer1954IsDerivation A ->
      Schafer1954IsPairDerivation (s54_pair_extend A);
  }.

  Definition schafer1954_theorem2_extension_surface
      (A : U -> U) : Schafer1954Theorem2ExtensionSurface A.
  Proof.
    refine
      {| s54_t2_extension_formula := _;
         s54_t2_extension_is_derivation := _ |}.
    - intros x y. reflexivity.
    - exact (schafer1954_theorem2_extension_map A).
  Defined.
End Schafer1954AbstractDerivations.

(** Theorem 3, pp. 441-444:
    paper-order block restriction lane.  The OCR for equations (30)-(32)
    is noisy, so we formalize the clean midpoint bridge explicitly:
    - skew-block data `(A,B,C)`
    - equation (29) as the derivation law for `A`
    - equation (33) as the derived derivation law for `C`
    - equation (34) as `C = A + L_c`
    - equation (35) as the normalization `c = 0`
    The mixed left/right identities (30)-(32) are kept as named abstract
    hypotheses so the Rocq proof surface matches the paper without inventing
    unreadable formulas. *)
Section Schafer1954Theorem3Restriction.
  Variable U : Type.
  Variable uadd umul : U -> U -> U.
  Variable uscale : R -> U -> U.
  Variable uzero : U.

  Hypothesis uadd_assoc : forall x y z : U,
    uadd x (uadd y z) = uadd (uadd x y) z.
  Hypothesis uadd_comm : forall x y : U,
    uadd x y = uadd y x.
  Hypothesis uadd_zero_r : forall x : U,
    uadd x uzero = x.
  Hypothesis umul_zero_l : forall x : U,
    umul uzero x = uzero.

  Variable s54_eq30_formula : (U -> U) -> U -> Prop.
  Variable s54_eq31_formula : (U -> U) -> U -> Prop.
  Variable s54_eq32_formula : (U -> U) -> U -> Prop.

  Definition s54_t3_is_derivation (D : U -> U) : Prop :=
    forall x y : U,
      D (umul x y) = uadd (umul (D x) y) (umul x (D y)).

  Record Schafer1954Theorem3BlockSurface := {
    s54_t3_A : U -> U;
    s54_t3_B : U -> U;
    s54_t3_C : U -> U;
    s54_t3_c : U;
    s54_t3_eq29 :
      s54_t3_is_derivation s54_t3_A;
    s54_t3_eq30 :
      forall y : U, s54_eq30_formula s54_t3_B y;
    s54_t3_eq31 :
      forall y : U, s54_eq31_formula s54_t3_B y;
    s54_t3_eq32 :
      forall x : U, s54_eq32_formula s54_t3_C x;
    s54_t3_eq33 :
      s54_t3_is_derivation s54_t3_C;
    s54_t3_eq34 :
      forall x : U,
        s54_t3_C x = uadd (s54_t3_A x) (umul s54_t3_c x);
    s54_t3_eq35 :
      s54_t3_c = uzero;
  }.

  Theorem s54_t3_eq29_states_A_is_derivation :
    forall S : Schafer1954Theorem3BlockSurface,
      s54_t3_is_derivation (s54_t3_A S).
  Proof.
    intros S.
    exact (s54_t3_eq29 S).
  Qed.

  Theorem s54_t3_eq33_states_C_is_derivation :
    forall S : Schafer1954Theorem3BlockSurface,
      s54_t3_is_derivation (s54_t3_C S).
  Proof.
    intros S.
    exact (s54_t3_eq33 S).
  Qed.

  Theorem s54_t3_eq34_eq35_force_C_eq_A :
    forall S : Schafer1954Theorem3BlockSurface,
    forall x : U,
      s54_t3_C S x = s54_t3_A S x.
  Proof.
    intros S x.
    rewrite s54_t3_eq34.
    rewrite s54_t3_eq35.
    rewrite umul_zero_l.
    rewrite uadd_zero_r.
    reflexivity.
  Qed.

  Record Schafer1954Theorem3RestrictionSurface
      (S : Schafer1954Theorem3BlockSurface) : Prop := {
    s54_t3_surface_A_is_derivation :
      s54_t3_is_derivation (s54_t3_A S);
    s54_t3_surface_C_is_derivation :
      s54_t3_is_derivation (s54_t3_C S);
    s54_t3_surface_C_eq_A :
      forall x : U, s54_t3_C S x = s54_t3_A S x;
  }.

  Definition schafer1954_theorem3_restriction_surface
      (S : Schafer1954Theorem3BlockSurface) :
      Schafer1954Theorem3RestrictionSurface S.
  Proof.
    refine
      {| s54_t3_surface_A_is_derivation := _;
         s54_t3_surface_C_is_derivation := _;
         s54_t3_surface_C_eq_A := _ |}.
    - exact (s54_t3_eq29_states_A_is_derivation S).
    - exact (s54_t3_eq33_states_C_is_derivation S).
    - exact (s54_t3_eq34_eq35_force_C_eq_A S).
  Defined.
End Schafer1954Theorem3Restriction.

(** Theorem 3, pp. 442-445:
    coordinate uniqueness lane after the block restriction bridge.  The paper
    derives a normalization chain through equations (36)-(52), culminating in
    a diagonal form for `B` and the finite triple constraints that force the
    diagonal parameters to vanish.  We formalize that chain as a paper-order
    coordinate surface, keeping OCR-noisy equations as named hypotheses while
    retaining the clean midpoint consequence used by Theorem 4: the normalized
    coordinate parameters vanish on the tracked basis indices. *)
Section Schafer1954Theorem3Coordinates.
  Variable U : Type.
  Variable uadd umul : U -> U -> U.
  Variable uscale : R -> U -> U.
  Variable uzero uone : U.

  Hypothesis uadd_assoc : forall x y z : U,
    uadd x (uadd y z) = uadd (uadd x y) z.
  Hypothesis uadd_comm : forall x y : U,
    uadd x y = uadd y x.
  Hypothesis uadd_zero_r : forall x : U,
    uadd x uzero = x.
  Hypothesis umul_zero_l : forall x : U,
    umul uzero x = uzero.

  Variable s54_eq30_formula : (U -> U) -> U -> Prop.
  Variable s54_eq31_formula : (U -> U) -> U -> Prop.
  Variable s54_eq32_formula : (U -> U) -> U -> Prop.
  Variable s54_eq37_formula : (U -> U) -> Prop.
  Variable s54_eq38_formula : (U -> U) -> Prop.
  Variable s54_eq39_formula : (U -> U) -> U -> Prop.
  Variable s54_eq40_formula : (U -> U) -> Prop.
  Variable s54_eq42_formula : (U -> U) -> Prop.
  Variable s54_eq46_formula : (U -> U) -> Prop.
  Variable s54_eq47_formula : (U -> U) -> Prop.
  Variable s54_eq48_formula : (U -> U) -> Prop.
  Variable s54_eq49_formula : (U -> U) -> Prop.

  Variable Index : Type.
  Variable tracked_index : Index -> Prop.
  Variable s54_triple_rel : Index -> Index -> Index -> Prop.
  Variable s54_eq50_formula : (Index -> R) -> (U -> U) -> Prop.
  Variable s54_eq52_formula : (Index -> Index -> Index -> Prop) -> Prop.

  Definition s54_t3_block_t : Type :=
    Schafer1954Theorem3BlockSurface
      U uadd umul uzero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula.

  Definition s54_t3_block_A_map (S : s54_t3_block_t) : U -> U :=
    s54_t3_A
      U uadd umul uzero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula S.

  Definition s54_t3_block_B_map (S : s54_t3_block_t) : U -> U :=
    s54_t3_B
      U uadd umul uzero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula S.

  Definition s54_t3_block_C_map (S : s54_t3_block_t) : U -> U :=
    s54_t3_C
      U uadd umul uzero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula S.

  Record Schafer1954Theorem3CoordinateSurface := {
    s54_t3_coord_block : s54_t3_block_t;
    s54_t3_eq36 :
      s54_t3_block_B_map s54_t3_coord_block uone = uzero;
    s54_t3_eq37 :
      s54_eq37_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq38 :
      s54_eq38_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq39 :
      forall y : U, s54_eq39_formula (s54_t3_block_B_map s54_t3_coord_block) y;
    s54_t3_eq40 :
      s54_eq40_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq42 :
      s54_eq42_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq46 :
      s54_eq46_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq47 :
      s54_eq47_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq48 :
      s54_eq48_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq49 :
      s54_eq49_formula (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_diag_scalar : Index -> R;
    s54_t3_eq50 :
      s54_eq50_formula s54_t3_diag_scalar
        (s54_t3_block_B_map s54_t3_coord_block);
    s54_t3_eq51 :
      forall i j k : Index,
        s54_triple_rel i j k ->
        (s54_t3_diag_scalar i +
         s54_t3_diag_scalar j +
         s54_t3_diag_scalar k)%R = 0%R;
    s54_t3_eq52 :
      s54_eq52_formula s54_triple_rel;
    s54_t3_eq51_eq52_force_zero :
      forall i : Index,
        tracked_index i -> s54_t3_diag_scalar i = 0%R;
  }.

  Theorem s54_t3_coordinate_surface_forces_C_eq_A :
    forall S : Schafer1954Theorem3CoordinateSurface,
    forall x : U,
      s54_t3_block_C_map (s54_t3_coord_block S) x =
      s54_t3_block_A_map (s54_t3_coord_block S) x.
  Proof.
    intros S x.
    exact
      (s54_t3_eq34_eq35_force_C_eq_A
         U uadd umul uzero
         uadd_zero_r umul_zero_l
         s54_eq30_formula s54_eq31_formula s54_eq32_formula
         (s54_t3_coord_block S) x).
  Qed.

  Theorem s54_t3_coordinate_surface_diag_zero :
    forall S : Schafer1954Theorem3CoordinateSurface,
    forall i : Index,
      tracked_index i -> s54_t3_diag_scalar S i = 0%R.
  Proof.
    intros S i Hi.
    exact (s54_t3_eq51_eq52_force_zero S i Hi).
  Qed.

  Record Schafer1954Theorem3CoordinateUniquenessSurface
      (S : Schafer1954Theorem3CoordinateSurface) : Prop := {
    s54_t3_coord_surface_C_eq_A :
      forall x : U,
        s54_t3_block_C_map (s54_t3_coord_block S) x =
        s54_t3_block_A_map (s54_t3_coord_block S) x;
    s54_t3_coord_surface_diag_zero :
      forall i : Index,
        tracked_index i -> s54_t3_diag_scalar S i = 0%R;
  }.

  Definition schafer1954_theorem3_coordinate_uniqueness_surface
      (S : Schafer1954Theorem3CoordinateSurface) :
      Schafer1954Theorem3CoordinateUniquenessSurface S.
  Proof.
    refine
      {| s54_t3_coord_surface_C_eq_A := _;
         s54_t3_coord_surface_diag_zero := _ |}.
    - exact (s54_t3_coordinate_surface_forces_C_eq_A S).
    - exact (s54_t3_coordinate_surface_diag_zero S).
  Defined.
End Schafer1954Theorem3Coordinates.

(** Theorem 4 support surface:
    the repo already carries the octonion/G2 dimensional data that Schafer
    cites when identifying the derivation algebra with the 14-dimensional
    type-G Lie algebra.  This is support material for the paper theorem lane,
    not yet the full derivation-algebra equality proof. *)
Record Schafer1954TypeGSupportSurface := {
  s54_dim_g2_is_14 : dim_g2 = 14%nat;
  s54_stabilizer_dim_is_8 : dim_stabilizer = 8%nat;
  s54_g2_action_support :
    21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168;
}.

Definition schafer1954_type_g_support_surface :
    Schafer1954TypeGSupportSurface :=
  {| s54_dim_g2_is_14 := eq_refl;
     s54_stabilizer_dim_is_8 := stabilizer_dimension_is_8;
     s54_g2_action_support := g2_acts_on_7_fano_points |}.

Theorem schafer1954_theorem4_type_g_support :
  dim_g2 = 14%nat /\
  dim_stabilizer = 8%nat /\
  (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
Proof.
  split.
  - exact (s54_dim_g2_is_14 schafer1954_type_g_support_surface).
  - split.
    + exact (s54_stabilizer_dim_is_8 schafer1954_type_g_support_surface).
    + exact (s54_g2_action_support schafer1954_type_g_support_surface).
Qed.

(** Theorem 4, pp. 445-446:
    abstract identification layer.  Theorem 2 gives the forward extension
    direction, and the Theorem 3 restriction/coordinate uniqueness lane gives
    the converse direction.  We package that equivalence as the paper's
    `D(M_t) = D(C)` identification surface, then attach the already-landed
    type-G support used for the 14-dimensional conclusion. *)
Section Schafer1954Theorem4Identification.
  Variable U : Type.
  Variable IsDerivation : (U -> U) -> Prop.
  Variable IsPairDerivation : ((U * U)%type -> (U * U)%type) -> Prop.
  Variable extend : (U -> U) -> ((U * U)%type -> (U * U)%type).

  Hypothesis s54_t4_forward :
    forall A : U -> U,
      IsDerivation A -> IsPairDerivation (extend A).
  Hypothesis s54_t4_backward :
    forall D : (U * U)%type -> (U * U)%type,
      IsPairDerivation D ->
      exists A : U -> U, IsDerivation A /\ D = extend A.

  Record Schafer1954Theorem4IdentificationSurface := {
    s54_t4_extension_direction :
      forall A : U -> U,
        IsDerivation A -> IsPairDerivation (extend A);
    s54_t4_restriction_direction :
      forall D : (U * U)%type -> (U * U)%type,
        IsPairDerivation D ->
        exists A : U -> U, IsDerivation A /\ D = extend A;
    s54_t4_identification :
      forall D : (U * U)%type -> (U * U)%type,
        IsPairDerivation D <->
        exists A : U -> U, IsDerivation A /\ D = extend A;
  }.

  Definition schafer1954_theorem4_identification_surface :
      Schafer1954Theorem4IdentificationSurface.
  Proof.
    refine
      {| s54_t4_extension_direction := s54_t4_forward;
         s54_t4_restriction_direction := s54_t4_backward;
         s54_t4_identification := _ |}.
    intro D.
    split.
    - exact (s54_t4_backward D).
    - intros [A [HA ->]].
      exact (s54_t4_forward A HA).
  Defined.

  Record Schafer1954Theorem4TypeGIdentificationSurface := {
    s54_t4_identification_surface_data :
      Schafer1954Theorem4IdentificationSurface;
    s54_t4_type_g_support_data :
      Schafer1954TypeGSupportSurface;
  }.

  Definition schafer1954_theorem4_type_g_identification_surface :
      Schafer1954Theorem4TypeGIdentificationSurface :=
    {| s54_t4_identification_surface_data :=
         schafer1954_theorem4_identification_surface;
       s54_t4_type_g_support_data :=
         schafer1954_type_g_support_surface |}.
End Schafer1954Theorem4Identification.

(** Concrete paper-specific discharge for equations (51)-(52):
    Schafer lists the seven Cayley triples on the standard octonion basis and
    proves that, over characteristic <> 2,3, the only real solution of the
    corresponding linear system is the trivial one.  This is the concrete
    finite-basis discharge that closes the abstract coordinate uniqueness lane
    in the real octonion case tracked by this repository. *)
Inductive Schafer1954Basis7 :=
| s54_b1 | s54_b2 | s54_b3 | s54_b4 | s54_b5 | s54_b6 | s54_b7.

Inductive schafer1954_eq52_cayley_triple :
    Schafer1954Basis7 -> Schafer1954Basis7 -> Schafer1954Basis7 -> Prop :=
| s54_eq52_123 :
    schafer1954_eq52_cayley_triple s54_b1 s54_b2 s54_b3
| s54_eq52_145 :
    schafer1954_eq52_cayley_triple s54_b1 s54_b4 s54_b5
| s54_eq52_167 :
    schafer1954_eq52_cayley_triple s54_b1 s54_b7 s54_b6
| s54_eq52_246 :
    schafer1954_eq52_cayley_triple s54_b2 s54_b4 s54_b6
| s54_eq52_257 :
    schafer1954_eq52_cayley_triple s54_b2 s54_b5 s54_b7
| s54_eq52_347 :
    schafer1954_eq52_cayley_triple s54_b3 s54_b4 s54_b7
| s54_eq52_356 :
    schafer1954_eq52_cayley_triple s54_b3 s54_b6 s54_b5.

Definition schafer1954_basis7_tracked (_ : Schafer1954Basis7) : Prop := True.

Definition s54_basis7_oct (i : Schafer1954Basis7) : CDOct :=
  match i with
  | s54_b1 => oct_e 1
  | s54_b2 => oct_e 2
  | s54_b3 => oct_e 3
  | s54_b4 => oct_e 4
  | s54_b5 => oct_e 5
  | s54_b6 => oct_e 6
  | s54_b7 => oct_e 7
  end.

Definition s54_basis7_coeff (i : Schafer1954Basis7) (x : CDOct) : R :=
  match i with
  | s54_b1 => qb (oct_lo x)
  | s54_b2 => qc (oct_lo x)
  | s54_b3 => qd (oct_lo x)
  | s54_b4 => qa (oct_hi x)
  | s54_b5 => qb (oct_hi x)
  | s54_b6 => qc (oct_hi x)
  | s54_b7 => qd (oct_hi x)
  end.

Theorem schafer1954_eq52_real_solution_zero :
  forall p : Schafer1954Basis7 -> R,
    (forall i j k : Schafer1954Basis7,
        schafer1954_eq52_cayley_triple i j k ->
        (p i + p j + p k)%R = 0%R) ->
    forall i : Schafer1954Basis7,
      p i = 0%R.
Proof.
  intros p Hp i.
  assert (H123 : (p s54_b1 + p s54_b2 + p s54_b3)%R = 0%R).
  { apply (Hp s54_b1 s54_b2 s54_b3). exact s54_eq52_123. }
  assert (H145 : (p s54_b1 + p s54_b4 + p s54_b5)%R = 0%R).
  { apply (Hp s54_b1 s54_b4 s54_b5). exact s54_eq52_145. }
  assert (H167 : (p s54_b1 + p s54_b7 + p s54_b6)%R = 0%R).
  { apply (Hp s54_b1 s54_b7 s54_b6). exact s54_eq52_167. }
  assert (H246 : (p s54_b2 + p s54_b4 + p s54_b6)%R = 0%R).
  { apply (Hp s54_b2 s54_b4 s54_b6). exact s54_eq52_246. }
  assert (H257 : (p s54_b2 + p s54_b5 + p s54_b7)%R = 0%R).
  { apply (Hp s54_b2 s54_b5 s54_b7). exact s54_eq52_257. }
  assert (H347 : (p s54_b3 + p s54_b4 + p s54_b7)%R = 0%R).
  { apply (Hp s54_b3 s54_b4 s54_b7). exact s54_eq52_347. }
  assert (H356 : (p s54_b3 + p s54_b6 + p s54_b5)%R = 0%R).
  { apply (Hp s54_b3 s54_b6 s54_b5). exact s54_eq52_356. }
  destruct i; lra.
Qed.

Lemma schafer1954_basis7_conj_neg :
  forall i : Schafer1954Basis7,
    oct_conj (s54_basis7_oct i) = oct_neg (s54_basis7_oct i).
Proof.
  intros i; destruct i;
  cbv [s54_basis7_oct oct_conj oct_neg oct_e oct_lo oct_hi
       quat_conj quat_neg qa qb qc qd quat_zero quat_one];
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Ltac s54_close_eq52_basis_case :=
  rewrite oct_basis_mul_xor by lia;
  vm_compute;
  apply (f_equal2 mkOct);
  [apply (f_equal4 mkQuat); [ring | ring | ring | ring]
  |apply (f_equal4 mkQuat); [ring | ring | ring | ring] ].

Lemma schafer1954_eq52_basis_mul :
  forall i j k : Schafer1954Basis7,
    schafer1954_eq52_cayley_triple i j k ->
    oct_mul (s54_basis7_oct i) (s54_basis7_oct j) = s54_basis7_oct k.
Proof.
  intros i j k Htr.
  destruct Htr; cbn [s54_basis7_oct]; s54_close_eq52_basis_case.
Qed.

Lemma schafer1954_eq52_basis_mul_rev :
  forall i j k : Schafer1954Basis7,
    schafer1954_eq52_cayley_triple i j k ->
    oct_mul (s54_basis7_oct j) (s54_basis7_oct i) = oct_neg (s54_basis7_oct k).
Proof.
  intros i j k Htr.
  destruct Htr; cbn [s54_basis7_oct]; s54_close_eq52_basis_case.
Qed.

Lemma schafer1954_eq52_basis_mul_cyclic_l :
  forall i j k : Schafer1954Basis7,
    schafer1954_eq52_cayley_triple i j k ->
    oct_mul (s54_basis7_oct j) (s54_basis7_oct k) = s54_basis7_oct i.
Proof.
  intros i j k Htr.
  destruct Htr; cbn [s54_basis7_oct]; s54_close_eq52_basis_case.
Qed.

Lemma schafer1954_eq52_basis_mul_cyclic_r :
  forall i j k : Schafer1954Basis7,
    schafer1954_eq52_cayley_triple i j k ->
    oct_mul (s54_basis7_oct k) (s54_basis7_oct i) = s54_basis7_oct j.
Proof.
  intros i j k Htr.
  destruct Htr; cbn [s54_basis7_oct]; s54_close_eq52_basis_case.
Qed.

Section Schafer1954Theorem4Concrete.
  Variable U : Type.
  Variable IsDerivation : (U -> U) -> Prop.
  Variable IsPairDerivation : ((U * U)%type -> (U * U)%type) -> Prop.
  Variable extend : (U -> U) -> ((U * U)%type -> (U * U)%type).
  Hypothesis s54_t4_forward :
    forall A : U -> U,
      IsDerivation A -> IsPairDerivation (extend A).
  Hypothesis s54_t4_backward :
    forall D : (U * U)%type -> (U * U)%type,
      IsPairDerivation D ->
      exists A : U -> U, IsDerivation A /\ D = extend A.

  Record Schafer1954Theorem4ConcreteTypeGSurface := {
    s54_t4_concrete_identification_data :
      Schafer1954Theorem4IdentificationSurface
        U IsDerivation IsPairDerivation extend;
    s54_t4_concrete_eq52_discharge :
      forall p : Schafer1954Basis7 -> R,
        (forall i j k : Schafer1954Basis7,
            schafer1954_eq52_cayley_triple i j k ->
            (p i + p j + p k)%R = 0%R) ->
        forall i : Schafer1954Basis7, p i = 0%R;
    s54_t4_concrete_type_g_support_data :
      Schafer1954TypeGSupportSurface;
  }.

  Definition schafer1954_theorem4_concrete_type_g_surface :
      Schafer1954Theorem4ConcreteTypeGSurface :=
    {| s54_t4_concrete_identification_data :=
         schafer1954_theorem4_identification_surface
           U IsDerivation IsPairDerivation extend
           s54_t4_forward s54_t4_backward;
       s54_t4_concrete_eq52_discharge :=
         schafer1954_eq52_real_solution_zero;
       s54_t4_concrete_type_g_support_data :=
         schafer1954_type_g_support_surface |}.
End Schafer1954Theorem4Concrete.

(** Concrete instantiation on the repo's actual octonion/sedenion objects. *)
Definition schafer1954_octonion_derivation (A : CDOct -> CDOct) : Prop :=
  Schafer1954IsDerivation CDOct oct_add oct_mul oct_scale oct_conj A.

Definition schafer1954_octonion_pair_derivation
    (D : s54_pair CDOct -> s54_pair CDOct) : Prop :=
  Schafer1954IsPairDerivation
    CDOct oct_add oct_mul oct_scale oct_conj (-1) D.

Record Schafer1954IsSedenionDerivation (D : CDSed -> CDSed) : Prop := {
  s54_sed_deriv_add :
    forall x y : CDSed, D (sed_add x y) = sed_add (D x) (D y);
  s54_sed_deriv_scale :
    forall r : R, forall x : CDSed, D (sed_scale r x) = sed_scale r (D x);
  s54_sed_deriv_mul :
    forall x y : CDSed, D (sed_mul x y) = sed_add (sed_mul (D x) y) (sed_mul x (D y));
}.

Definition s54_sed_to_pair (x : CDSed) : s54_pair CDOct :=
  (sed_lo x, sed_hi x).

Definition s54_pair_to_sed (p : s54_pair CDOct) : CDSed :=
  mkSed (fst p) (snd p).

Definition s54_pair_derivation_to_sed
    (D : s54_pair CDOct -> s54_pair CDOct) :
    CDSed -> CDSed :=
  fun x => s54_pair_to_sed (D (s54_sed_to_pair x)).

Definition s54_sed_derivation_to_pair
    (D : CDSed -> CDSed) :
    s54_pair CDOct -> s54_pair CDOct :=
  fun p => s54_sed_to_pair (D (s54_pair_to_sed p)).

Definition schafer1954_sedenion_extend
    (A : CDOct -> CDOct) : CDSed -> CDSed :=
  s54_pair_derivation_to_sed (s54_pair_extend CDOct A).

Lemma s54_oct_add_assoc : forall x y z : CDOct,
  oct_add x (oct_add y z) = oct_add (oct_add x y) z.
Proof.
  intros [xlo xhi] [ylo yhi] [zlo zhi].
  unfold oct_add; simpl.
  apply (f_equal2 mkOct); unfold quat_add; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_scale_add_distr : forall r : R, forall x y : CDOct,
  oct_scale r (oct_add x y) = oct_add (oct_scale r x) (oct_scale r y).
Proof.
  intros r [xlo xhi] [ylo yhi].
  unfold oct_scale, oct_add; simpl.
  apply (f_equal2 mkOct); unfold quat_scale, quat_add; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_scale_one : forall x : CDOct,
  oct_scale 1 x = x.
Proof.
  intros [[a b c d] [e f g h]].
  unfold oct_scale; simpl.
  apply (f_equal2 mkOct); unfold quat_scale; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_scale_neg_one : forall x : CDOct,
  oct_scale (-1) x = oct_neg x.
Proof.
  intros [[a b c d] [e f g h]].
  unfold oct_scale, oct_neg; simpl.
  apply (f_equal2 mkOct); unfold quat_scale, quat_neg; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_scale_zero_right : forall r : R,
  oct_scale r oct_zero = oct_zero.
Proof.
  intro r.
  unfold oct_scale, oct_zero; simpl.
  apply (f_equal2 mkOct); unfold quat_scale, quat_zero; simpl;
  apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_conj_zero : oct_conj oct_zero = oct_zero.
Proof.
  cbv [oct_conj oct_zero oct_lo oct_hi
       quat_conj quat_neg quat_zero qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_conj_one :
  oct_conj (mkOct quat_one quat_zero) = mkOct quat_one quat_zero.
Proof.
  cbv [oct_conj oct_lo oct_hi
       quat_conj quat_neg quat_zero quat_one qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_mul_one_left : forall x : CDOct,
  oct_mul (mkOct quat_one quat_zero) x = x.
Proof.
  intros [[a b c d] [e f g h]].
  cbv [oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma s54_oct_add_cancel_left : forall z x y : CDOct,
  oct_add z x = oct_add z y -> x = y.
Proof.
  intros [[z1 z2 z3 z4] [z5 z6 z7 z8]]
         [[x1 x2 x3 x4] [x5 x6 x7 x8]]
         [[y1 y2 y3 y4] [y5 y6 y7 y8]] H.
  cbv [oct_add oct_lo oct_hi quat_add qa qb qc qd] in H.
  inversion H; clear H; subst.
  repeat match goal with
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion Hq; clear Hq; subst
  end.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

Lemma s54_pair_to_sed_add :
  forall p q : s54_pair CDOct,
    s54_pair_to_sed (s54_pair_add CDOct oct_add p q) =
    sed_add (s54_pair_to_sed p) (s54_pair_to_sed q).
Proof.
  intros [a b] [c d]. reflexivity.
Qed.

Lemma s54_sed_to_pair_add :
  forall x y : CDSed,
    s54_sed_to_pair (sed_add x y) =
    s54_pair_add CDOct oct_add (s54_sed_to_pair x) (s54_sed_to_pair y).
Proof.
  intros [a b] [c d]. reflexivity.
Qed.

Lemma s54_pair_to_sed_scale :
  forall r : R, forall p : s54_pair CDOct,
    s54_pair_to_sed (s54_pair_scale CDOct oct_scale r p) =
    sed_scale r (s54_pair_to_sed p).
Proof.
  intros r [a b]. reflexivity.
Qed.

Lemma s54_sed_to_pair_scale :
  forall r : R, forall x : CDSed,
    s54_sed_to_pair (sed_scale r x) =
    s54_pair_scale CDOct oct_scale r (s54_sed_to_pair x).
Proof.
  intros r [a b]. reflexivity.
Qed.

Lemma s54_pair_to_sed_mul :
  forall p q : s54_pair CDOct,
    s54_pair_to_sed
      (s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1) p q) =
    sed_mul (s54_pair_to_sed p) (s54_pair_to_sed q).
Proof.
  intros [a b] [c d].
  unfold s54_pair_to_sed, sed_mul, s54_pair_mul; simpl.
  apply (f_equal2 mkSed).
  - rewrite s54_oct_scale_neg_one.
    unfold oct_add, oct_neg; simpl.
    reflexivity.
  - unfold oct_add; simpl.
    reflexivity.
Qed.

Lemma s54_sed_to_pair_mul :
  forall x y : CDSed,
    s54_sed_to_pair (sed_mul x y) =
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (s54_sed_to_pair x) (s54_sed_to_pair y).
Proof.
  intros [a b] [c d].
  unfold s54_sed_to_pair, sed_mul, s54_pair_mul; simpl.
  rewrite s54_oct_scale_neg_one.
  unfold oct_add, oct_neg; simpl.
  reflexivity.
Qed.

Theorem schafer1954_sedenion_derivation_to_octonion_pair :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    schafer1954_octonion_pair_derivation (s54_sed_derivation_to_pair D).
Proof.
  intros D HD.
  destruct HD as [Hadd Hscale Hmul].
  constructor.
  - intros [a b] [c d].
    unfold s54_sed_derivation_to_pair; simpl.
    rewrite s54_pair_to_sed_add.
    rewrite Hadd.
    rewrite s54_sed_to_pair_add.
    reflexivity.
  - intros r [a b].
    unfold s54_sed_derivation_to_pair; simpl.
    rewrite s54_pair_to_sed_scale.
    rewrite Hscale.
    rewrite s54_sed_to_pair_scale.
    reflexivity.
  - intros [a b] [c d].
    unfold s54_sed_derivation_to_pair; simpl.
    rewrite s54_pair_to_sed_mul.
    rewrite Hmul.
    rewrite s54_sed_to_pair_add.
    rewrite !s54_sed_to_pair_mul.
    reflexivity.
Qed.

Theorem schafer1954_octonion_pair_derivation_to_sedenion :
  forall D : s54_pair CDOct -> s54_pair CDOct,
    schafer1954_octonion_pair_derivation D ->
    Schafer1954IsSedenionDerivation (s54_pair_derivation_to_sed D).
Proof.
  intros D HD.
  destruct HD as [Hadd Hscale Hmul].
  constructor.
  - intros x y.
    unfold s54_pair_derivation_to_sed.
    rewrite s54_sed_to_pair_add.
    rewrite Hadd.
    rewrite s54_pair_to_sed_add.
    reflexivity.
  - intros r x.
    unfold s54_pair_derivation_to_sed.
    rewrite s54_sed_to_pair_scale.
    rewrite Hscale.
    rewrite s54_pair_to_sed_scale.
    reflexivity.
  - intros x y.
    unfold s54_pair_derivation_to_sed.
    rewrite s54_sed_to_pair_mul.
    rewrite Hmul.
    repeat rewrite s54_pair_to_sed_add.
    repeat rewrite s54_pair_to_sed_mul.
    reflexivity.
Qed.

Theorem schafer1954_theorem2_octonion_to_sedenion_extension_map :
  forall A : CDOct -> CDOct,
    schafer1954_octonion_derivation A ->
    Schafer1954IsSedenionDerivation (schafer1954_sedenion_extend A).
Proof.
  intros A HA.
  unfold schafer1954_sedenion_extend.
  apply schafer1954_octonion_pair_derivation_to_sedenion.
  apply
    (schafer1954_theorem2_extension_map
       CDOct oct_add oct_mul oct_scale oct_conj (-1)).
  - exact s54_oct_add_assoc.
  - exact oct_add_comm.
  - exact s54_oct_scale_add_distr.
  - exact HA.
Qed.

Definition s54_oct_one : CDOct := mkOct quat_one quat_zero.

Definition s54_oct_embed (a : CDOct) : CDSed := mkSed a oct_zero.

Definition s54_hi_embed (a : CDOct) : CDSed := mkSed oct_zero a.

Definition s54_y : CDSed := s54_hi_embed s54_oct_one.

Lemma s54_oct_embed_add : forall a b : CDOct,
  s54_oct_embed (oct_add a b) =
  sed_add (s54_oct_embed a) (s54_oct_embed b).
Proof.
  intros a b.
  unfold s54_oct_embed, sed_add; simpl.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_oct_embed_scale : forall r : R, forall a : CDOct,
  s54_oct_embed (oct_scale r a) =
  sed_scale r (s54_oct_embed a).
Proof.
  intros r a.
  unfold s54_oct_embed, sed_scale; simpl.
  rewrite s54_oct_scale_zero_right.
  reflexivity.
Qed.

Lemma s54_hi_embed_add : forall a b : CDOct,
  s54_hi_embed (oct_add a b) =
  sed_add (s54_hi_embed a) (s54_hi_embed b).
Proof.
  intros a b.
  unfold s54_hi_embed, sed_add; simpl.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_hi_embed_scale : forall r : R, forall a : CDOct,
  s54_hi_embed (oct_scale r a) =
  sed_scale r (s54_hi_embed a).
Proof.
  intros r a.
  unfold s54_hi_embed, sed_scale; simpl.
  rewrite s54_oct_scale_zero_right.
  reflexivity.
Qed.

Lemma s54_sed_split : forall a b : CDOct,
  mkSed a b = sed_add (s54_oct_embed a) (s54_hi_embed b).
Proof.
  intros a b.
  unfold s54_oct_embed, s54_hi_embed, sed_add; simpl.
  rewrite oct_add_zero_right.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_sed_mul_one_left : forall x : CDSed,
  sed_mul sed_one x = x.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [sed_mul sed_one sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Lemma s54_sed_self_add_eq_zero : forall x : CDSed,
  x = sed_add x x -> x = sed_zero.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]] H.
  cbv [sed_add sed_zero oct_add oct_zero quat_add quat_zero
       qa qb qc qd oct_lo oct_hi sed_lo sed_hi] in H.
  inversion H; clear H; subst.
  repeat match goal with
  | Hq : mkOct _ _ = mkOct _ _ |- _ => inversion Hq; clear Hq; subst
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion Hq; clear Hq; subst
  end.
  assert (Ha1 : a1 = 0%R) by lra.
  assert (Ha2 : a2 = 0%R) by lra.
  assert (Ha3 : a3 = 0%R) by lra.
  assert (Ha4 : a4 = 0%R) by lra.
  assert (Ha5 : a5 = 0%R) by lra.
  assert (Ha6 : a6 = 0%R) by lra.
  assert (Ha7 : a7 = 0%R) by lra.
  assert (Ha8 : a8 = 0%R) by lra.
  assert (Ha9 : a9 = 0%R) by lra.
  assert (Ha10 : a10 = 0%R) by lra.
  assert (Ha11 : a11 = 0%R) by lra.
  assert (Ha12 : a12 = 0%R) by lra.
  assert (Ha13 : a13 = 0%R) by lra.
  assert (Ha14 : a14 = 0%R) by lra.
  assert (Ha15 : a15 = 0%R) by lra.
  assert (Ha16 : a16 = 0%R) by lra.
  subst a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16.
  cbv [sed_zero oct_zero quat_zero].
  apply f_equal2.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
Qed.

Lemma s54_oct_embed_mul : forall a b : CDOct,
  sed_mul (s54_oct_embed a) (s54_oct_embed b) = s54_oct_embed (oct_mul a b).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]] [[b1 b2 b3 b4] [b5 b6 b7 b8]].
  cbv [s54_oct_embed sed_mul sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi oct_neg
       quat_mul quat_add quat_neg quat_conj quat_zero
       qa qb qc qd].
  apply f_equal2.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
Qed.

Lemma s54_oct_embed_mul_y : forall a : CDOct,
  sed_mul (s54_oct_embed a) s54_y = s54_hi_embed a.
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [s54_oct_embed s54_hi_embed s54_y s54_oct_one sed_mul sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi oct_neg
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply f_equal2.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
Qed.

Lemma s54_y_mul_oct_embed : forall a : CDOct,
  sed_mul s54_y (s54_oct_embed a) = s54_hi_embed (oct_conj a).
Proof.
  intros [[a1 a2 a3 a4] [a5 a6 a7 a8]].
  cbv [s54_oct_embed s54_hi_embed s54_y s54_oct_one sed_mul sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi oct_neg
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply f_equal2.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
  - apply f_equal2.
    + apply f_equal4; ring.
    + apply f_equal4; ring.
Qed.

Lemma s54_y_square :
  sed_mul s54_y s54_y = sed_neg sed_one.
Proof.
  cbv [s54_y s54_hi_embed s54_oct_one sed_mul sed_neg sed_one sed_lo sed_hi
       oct_mul oct_conj oct_zero oct_lo oct_hi oct_neg
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

Theorem s54_sed_derivation_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    D sed_zero = sed_zero.
Proof.
  intros D HD.
  destruct HD as [_ Hscale _].
  specialize (Hscale 0%R sed_one).
  rewrite sed_scale_zero in Hscale.
  rewrite sed_scale_zero in Hscale.
  exact Hscale.
Qed.

Theorem s54_sed_derivation_one_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    D sed_one = sed_zero.
Proof.
  intros D HD.
  destruct HD as [_ _ Hmul].
  specialize (Hmul sed_one sed_one).
  rewrite s54_sed_mul_one_left in Hmul.
  rewrite sed_mul_one_right in Hmul.
  rewrite s54_sed_mul_one_left in Hmul.
  apply s54_sed_self_add_eq_zero.
  exact Hmul.
Qed.

Lemma s54_pair_mul_lo_lo :
  forall a b : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (a, oct_zero) (b, oct_zero) =
    (oct_mul a b, oct_zero).
Proof.
  intros a b.
  unfold s54_pair_mul; simpl.
  rewrite s54_oct_conj_zero.
  rewrite oct_mul_zero_left.
  rewrite s54_oct_scale_zero_right.
  rewrite oct_add_zero_right.
  repeat rewrite oct_mul_zero_left.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_pair_mul_any_lo :
  forall x1 x2 b : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (x1, x2) (b, oct_zero) =
    (oct_mul x1 b, oct_mul x2 (oct_conj b)).
Proof.
  intros x1 x2 b.
  unfold s54_pair_mul; simpl.
  rewrite s54_oct_conj_zero.
  rewrite oct_mul_zero_left.
  rewrite s54_oct_scale_zero_right.
  rewrite oct_add_zero_right.
  rewrite oct_mul_zero_left.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_pair_mul_lo_any :
  forall a u v : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (a, oct_zero) (u, v) =
    (oct_mul a u, oct_mul v a).
Proof.
  intros a u v.
  unfold s54_pair_mul; simpl.
  rewrite oct_mul_zero_right.
  rewrite s54_oct_scale_zero_right.
  rewrite oct_add_zero_right.
  rewrite oct_mul_zero_left.
  rewrite oct_add_zero_right.
  reflexivity.
Qed.

Lemma s54_pair_mul_lo_y :
  forall a : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (a, oct_zero) (oct_zero, s54_oct_one) =
    (oct_zero, a).
Proof.
  intro a.
  unfold s54_oct_one.
  rewrite s54_pair_mul_lo_any.
  rewrite oct_mul_zero_right.
  repeat rewrite s54_oct_mul_one_left.
  reflexivity.
Qed.

Lemma s54_pair_mul_y_lo :
  forall a : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (oct_zero, s54_oct_one) (a, oct_zero) =
    (oct_zero, oct_conj a).
Proof.
  intro a.
  unfold s54_oct_one.
  rewrite s54_pair_mul_any_lo.
  rewrite oct_mul_zero_left.
  repeat rewrite s54_oct_mul_one_left.
  reflexivity.
Qed.

Lemma s54_pair_mul_any_y :
  forall aa cc : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (aa, cc) (oct_zero, s54_oct_one) =
    (oct_neg cc, aa).
Proof.
  intros aa cc.
  unfold s54_pair_mul, s54_oct_one; simpl.
  rewrite oct_mul_zero_right.
  rewrite s54_oct_conj_one.
  repeat rewrite s54_oct_mul_one_left.
  rewrite s54_oct_scale_neg_one.
  rewrite oct_add_zero_left.
  rewrite s54_oct_conj_zero.
  rewrite oct_mul_zero_right.
  rewrite oct_add_zero_right.
  reflexivity.
Qed.

Lemma s54_pair_mul_y_any :
  forall aa cc : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (oct_zero, s54_oct_one) (aa, cc) =
    (oct_neg (oct_conj cc), oct_conj aa).
Proof.
  intros aa cc.
  unfold s54_pair_mul, s54_oct_one; simpl.
  rewrite oct_mul_zero_left.
  rewrite oct_mul_one_right.
  rewrite s54_oct_scale_neg_one.
  rewrite oct_add_zero_left.
  rewrite oct_mul_zero_right.
  repeat rewrite s54_oct_mul_one_left.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Definition s54_block_A (D : CDSed -> CDSed) (a : CDOct) : CDOct :=
  sed_lo (D (s54_oct_embed a)).

Definition s54_block_C (D : CDSed -> CDSed) (a : CDOct) : CDOct :=
  sed_hi (D (s54_oct_embed a)).

Definition s54_block_B (D : CDSed -> CDSed) (a : CDOct) : CDOct :=
  sed_lo (D (s54_hi_embed a)).

Definition s54_block_E (D : CDSed -> CDSed) (a : CDOct) : CDOct :=
  sed_hi (D (s54_hi_embed a)).

Definition s54_block_u (D : CDSed -> CDSed) : CDOct :=
  s54_block_B D s54_oct_one.

Definition s54_block_v (D : CDSed -> CDSed) : CDOct :=
  s54_block_E D s54_oct_one.

Theorem s54_block_A_is_derivation :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_A D (oct_mul a b) =
      oct_add (oct_mul (s54_block_A D a) b)
              (oct_mul a (s54_block_A D b)).
Proof.
  intros D HD a b.
  pose proof (schafer1954_sedenion_derivation_to_octonion_pair D HD) as HDp.
  destruct HDp as [_ _ Hpair_mul].
  specialize (Hpair_mul (a, oct_zero) (b, oct_zero)).
  unfold s54_sed_derivation_to_pair, s54_block_A,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_lo_lo in Hpair_mul.
  rewrite s54_pair_mul_any_lo in Hpair_mul.
  rewrite s54_pair_mul_lo_any in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  cbn [fst] in Hlo.
  exact Hlo.
Qed.

Theorem s54_block_A_add :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_A D (oct_add a b) =
      oct_add (s54_block_A D a) (s54_block_A D b).
Proof.
  intros D HD a b.
  destruct HD as [Hadd _ _].
  unfold s54_block_A.
  rewrite s54_oct_embed_add.
  rewrite Hadd.
  reflexivity.
Qed.

Theorem s54_block_A_scale :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall r : R, forall a : CDOct,
      s54_block_A D (oct_scale r a) =
      oct_scale r (s54_block_A D a).
Proof.
  intros D HD r a.
  destruct HD as [_ Hscale _].
  unfold s54_block_A.
  rewrite s54_oct_embed_scale.
  rewrite Hscale.
  reflexivity.
Qed.

Theorem s54_block_C_add :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_C D (oct_add a b) =
      oct_add (s54_block_C D a) (s54_block_C D b).
Proof.
  intros D HD a b.
  destruct HD as [Hadd _ _].
  unfold s54_block_C.
  rewrite s54_oct_embed_add.
  rewrite Hadd.
  reflexivity.
Qed.

Theorem s54_block_C_scale :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall r : R, forall a : CDOct,
      s54_block_C D (oct_scale r a) =
      oct_scale r (s54_block_C D a).
Proof.
  intros D HD r a.
  destruct HD as [_ Hscale _].
  unfold s54_block_C.
  rewrite s54_oct_embed_scale.
  rewrite Hscale.
  reflexivity.
Qed.

Theorem s54_block_B_add :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_B D (oct_add a b) =
      oct_add (s54_block_B D a) (s54_block_B D b).
Proof.
  intros D HD a b.
  destruct HD as [Hadd _ _].
  unfold s54_block_B.
  rewrite s54_hi_embed_add.
  rewrite Hadd.
  reflexivity.
Qed.

Theorem s54_block_B_scale :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall r : R, forall a : CDOct,
      s54_block_B D (oct_scale r a) =
      oct_scale r (s54_block_B D a).
Proof.
  intros D HD r a.
  destruct HD as [_ Hscale _].
  unfold s54_block_B.
  rewrite s54_hi_embed_scale.
  rewrite Hscale.
  reflexivity.
Qed.

Theorem s54_block_E_add :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_E D (oct_add a b) =
      oct_add (s54_block_E D a) (s54_block_E D b).
Proof.
  intros D HD a b.
  destruct HD as [Hadd _ _].
  unfold s54_block_E.
  rewrite s54_hi_embed_add.
  rewrite Hadd.
  reflexivity.
Qed.

Theorem s54_block_E_scale :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall r : R, forall a : CDOct,
      s54_block_E D (oct_scale r a) =
      oct_scale r (s54_block_E D a).
Proof.
  intros D HD r a.
  destruct HD as [_ Hscale _].
  unfold s54_block_E.
  rewrite s54_hi_embed_scale.
  rewrite Hscale.
  reflexivity.
Qed.

Theorem s54_blocks_from_right_generator :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_B D a =
      oct_add (oct_mul a (s54_block_u D)) (oct_neg (s54_block_C D a)) /\
      s54_block_E D a =
      oct_add (s54_block_A D a) (oct_mul (s54_block_v D) a).
Proof.
  intros D HD a.
  pose proof (schafer1954_sedenion_derivation_to_octonion_pair D HD) as HDp.
  destruct HDp as [_ _ Hpair_mul].
  specialize (Hpair_mul (a, oct_zero) (oct_zero, s54_oct_one)).
  unfold s54_sed_derivation_to_pair, s54_block_A, s54_block_B, s54_block_C,
         s54_block_E, s54_block_u, s54_block_v,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_lo_y in Hpair_mul.
  rewrite s54_pair_mul_any_y in Hpair_mul.
  rewrite s54_pair_mul_lo_any in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as HB.
  pose proof (f_equal snd Hpair_mul) as HE.
  cbn [fst snd] in HB, HE.
  rewrite oct_add_comm in HB.
  split; [exact HB | exact HE].
Qed.

Theorem s54_blocks_from_left_generator :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_B D (oct_conj a) =
      oct_add (oct_mul (s54_block_u D) a)
              (oct_neg (oct_conj (s54_block_C D a))) /\
      s54_block_E D (oct_conj a) =
      oct_add (oct_mul (s54_block_v D) (oct_conj a))
              (oct_conj (s54_block_A D a)).
Proof.
  intros D HD a.
  pose proof (schafer1954_sedenion_derivation_to_octonion_pair D HD) as HDp.
  destruct HDp as [_ _ Hpair_mul].
  specialize (Hpair_mul (oct_zero, s54_oct_one) (a, oct_zero)).
  unfold s54_sed_derivation_to_pair, s54_block_A, s54_block_B, s54_block_C,
         s54_block_E, s54_block_u, s54_block_v,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_y_lo in Hpair_mul.
  rewrite s54_pair_mul_any_lo in Hpair_mul.
  rewrite s54_pair_mul_y_any in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as HB.
  pose proof (f_equal snd Hpair_mul) as HE.
  cbn [fst snd] in HB, HE.
  split; [exact HB | exact HE].
Qed.

Theorem s54_block_A_conj :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_A D (oct_conj a) = oct_conj (s54_block_A D a).
Proof.
  intros D HD a.
  destruct (s54_blocks_from_right_generator D HD (oct_conj a)) as [_ Hright].
  destruct (s54_blocks_from_left_generator D HD a) as [_ Hleft].
  rewrite oct_add_comm in Hright.
  rewrite Hleft in Hright.
  symmetry in Hright.
  eapply s54_oct_add_cancel_left.
  exact Hright.
Qed.

Theorem s54_block_A_one_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    s54_block_A D s54_oct_one = oct_zero.
Proof.
  intros D HD.
  unfold s54_block_A, s54_oct_one, s54_oct_embed.
  pose proof (s54_sed_derivation_one_zero D HD) as Hone.
  exact (f_equal sed_lo Hone).
Qed.

Theorem s54_block_C_one_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    s54_block_C D s54_oct_one = oct_zero.
Proof.
  intros D HD.
  unfold s54_block_C, s54_oct_one, s54_oct_embed.
  pose proof (s54_sed_derivation_one_zero D HD) as Hone.
  exact (f_equal sed_hi Hone).
Qed.

Theorem s54_block_A_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    s54_block_A D oct_zero = oct_zero.
Proof.
  intros D HD.
  unfold s54_block_A, s54_oct_embed.
  replace (mkSed oct_zero oct_zero) with sed_zero by reflexivity.
  pose proof (s54_sed_derivation_zero D HD) as Hz.
  exact (f_equal sed_lo Hz).
Qed.

Theorem s54_block_A_is_octonion_derivation :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    schafer1954_octonion_derivation (s54_block_A D).
Proof.
  intros D HD.
  constructor.
  - exact (s54_block_A_add D HD).
  - exact (s54_block_A_scale D HD).
  - exact (s54_block_A_is_derivation D HD).
  - exact (s54_block_A_conj D HD).
Qed.

Theorem s54_blocks_decompose_sedenion_derivation :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      D (mkSed a b) =
      mkSed
        (oct_add (s54_block_A D a) (s54_block_B D b))
        (oct_add (s54_block_C D a) (s54_block_E D b)).
Proof.
  intros D HD a b.
  destruct HD as [Hadd _ _].
  rewrite s54_sed_split.
  rewrite Hadd.
  reflexivity.
Qed.

Theorem s54_concrete_backward_from_block_equalities :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    (forall a : CDOct, s54_block_B D a = oct_zero) ->
    (forall a : CDOct, s54_block_C D a = oct_zero) ->
    (forall a : CDOct, s54_block_E D a = s54_block_A D a) ->
    D = schafer1954_sedenion_extend (s54_block_A D).
Proof.
  intros D HD HB HC HE.
  extensionality x.
  destruct x as [a b].
  rewrite s54_blocks_decompose_sedenion_derivation by exact HD.
  unfold schafer1954_sedenion_extend,
         s54_pair_derivation_to_sed, s54_pair_extend,
         s54_pair_to_sed, s54_sed_to_pair, s54_block_A.
  cbn [fst snd].
  rewrite HB, HC, HE.
  rewrite oct_add_zero_right.
  rewrite oct_add_zero_left.
  reflexivity.
Qed.

Lemma s54_sed_add_assoc : forall x y z : CDSed,
  sed_add x (sed_add y z) = sed_add (sed_add x y) z.
Proof.
  intros [xlo xhi] [ylo yhi] [zlo zhi].
  unfold sed_add; simpl.
  f_equal; apply s54_oct_add_assoc.
Qed.

Lemma s54_sed_scale_add_distr : forall r : R, forall x y : CDSed,
  sed_scale r (sed_add x y) = sed_add (sed_scale r x) (sed_scale r y).
Proof.
  intros r [xlo xhi] [ylo yhi].
  unfold sed_scale, sed_add; simpl.
  f_equal; apply s54_oct_scale_add_distr.
Qed.

Lemma s54_sed_scale_neg : forall r : R, forall x : CDSed,
  sed_scale r (sed_neg x) = sed_neg (sed_scale r x).
Proof.
  intros r [[[a1 a2 a3 a4] [a5 a6 a7 a8]]
            [[b1 b2 b3 b4] [b5 b6 b7 b8]]].
  unfold sed_scale, sed_neg, oct_scale, oct_neg, quat_scale, quat_neg; simpl.
  repeat f_equal; ring.
Qed.

Lemma s54_sed_add_shuffle4 :
  forall a b c d : CDSed,
    sed_add (sed_add a b) (sed_add c d) =
    sed_add (sed_add a c) (sed_add b d).
Proof.
  intros a b c d.
  rewrite <- s54_sed_add_assoc.
  rewrite (s54_sed_add_assoc b c d).
  rewrite (sed_add_comm b c).
  rewrite <- s54_sed_add_assoc.
  rewrite <- s54_sed_add_assoc.
  rewrite s54_sed_add_assoc.
  reflexivity.
Qed.

Lemma s54_sed_sub_add_distr :
  forall a b c d : CDSed,
    sed_sub (sed_add a b) (sed_add c d) =
    sed_add (sed_sub a c) (sed_sub b d).
Proof.
  intros a b c d.
  unfold sed_sub.
  rewrite sed_neg_add.
  apply s54_sed_add_shuffle4.
Qed.

Lemma s54_sed_sub_scale_distr :
  forall r : R, forall x y : CDSed,
    sed_sub (sed_scale r x) (sed_scale r y) =
    sed_scale r (sed_sub x y).
Proof.
  intros r x y.
  unfold sed_sub.
  rewrite <- s54_sed_scale_neg.
  rewrite <- s54_sed_scale_add_distr.
  reflexivity.
Qed.

Lemma s54_sed_sub_mul_left_distr :
  forall x y z : CDSed,
    sed_mul (sed_sub x y) z = sed_sub (sed_mul x z) (sed_mul y z).
Proof.
  intros x y z.
  unfold sed_sub.
  rewrite sed_mul_add_left.
  rewrite sed_neg_mul_left.
  reflexivity.
Qed.

Lemma s54_sed_sub_mul_right_distr :
  forall x y z : CDSed,
    sed_mul x (sed_sub y z) = sed_sub (sed_mul x y) (sed_mul x z).
Proof.
  intros x y z.
  unfold sed_sub.
  rewrite sed_mul_add_right.
  rewrite sed_neg_mul_right.
  reflexivity.
Qed.

Lemma s54_oct_add_neg_zero_implies_eq :
  forall x y : CDOct,
    oct_add x (oct_neg y) = oct_zero ->
    x = y.
Proof.
  intros [[x1 x2 x3 x4] [x5 x6 x7 x8]]
         [[y1 y2 y3 y4] [y5 y6 y7 y8]] H.
  cbv [oct_add oct_neg oct_zero quat_add quat_neg quat_zero
       oct_lo oct_hi qa qb qc qd] in H.
  inversion H; clear H; subst.
  repeat match goal with
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion Hq; clear Hq; subst
  end.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

Lemma s54_oct_neg_injective :
  forall x y : CDOct,
    oct_neg x = oct_neg y ->
    x = y.
Proof.
  intros [[x1 x2 x3 x4] [x5 x6 x7 x8]]
         [[y1 y2 y3 y4] [y5 y6 y7 y8]] Hxy.
  cbv [oct_neg quat_neg oct_lo oct_hi qa qb qc qd] in Hxy.
  inversion Hxy; clear Hxy; subst.
  repeat match goal with
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion Hq; clear Hq; subst
  end.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

Lemma s54_oct_conj_add_self_zero_implies_pure :
  forall u : CDOct,
    oct_add (oct_conj u) u = oct_zero ->
    oct_conj u = oct_neg u.
Proof.
  intros [[u1 u2 u3 u4] [u5 u6 u7 u8]] Hu.
  cbv [oct_add oct_conj oct_neg oct_zero
       quat_add quat_conj quat_neg quat_zero
       oct_lo oct_hi qa qb qc qd] in Hu |- *.
  inversion Hu; clear Hu; subst.
  repeat match goal with
  | Hq : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion Hq; clear Hq; subst
  end.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); lra.
Qed.

Definition s54_sed_residual (D : CDSed -> CDSed) : CDSed -> CDSed :=
  fun x => sed_sub (D x) (schafer1954_sedenion_extend (s54_block_A D) x).

Theorem s54_sed_residual_is_derivation :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    Schafer1954IsSedenionDerivation (s54_sed_residual D).
Proof.
  intros D HD.
  pose proof (s54_block_A_is_octonion_derivation D HD) as HA.
  pose proof (schafer1954_theorem2_octonion_to_sedenion_extension_map
                (s54_block_A D) HA) as HE.
  destruct HD as [Hadd Hscale Hmul].
  destruct HE as [HaddE HscaleE HmulE].
  constructor.
  - intros x y.
    unfold s54_sed_residual.
    rewrite Hadd.
    rewrite HaddE.
    apply s54_sed_sub_add_distr.
  - intros r x.
    unfold s54_sed_residual.
    rewrite Hscale.
    rewrite HscaleE.
    apply s54_sed_sub_scale_distr.
  - intros x y.
    unfold s54_sed_residual.
    rewrite Hmul.
    rewrite HmulE.
    rewrite s54_sed_sub_add_distr.
    rewrite s54_sed_sub_mul_left_distr.
    rewrite s54_sed_sub_mul_right_distr.
    reflexivity.
Qed.

Theorem s54_sed_residual_block_A_zero :
  forall D : CDSed -> CDSed,
    forall a : CDOct,
      s54_block_A (s54_sed_residual D) a = oct_zero.
Proof.
  intros D a.
  unfold s54_block_A, s54_sed_residual, sed_sub,
         schafer1954_sedenion_extend, s54_pair_derivation_to_sed,
         s54_pair_extend, s54_pair_to_sed, s54_sed_to_pair, s54_oct_embed.
  simpl.
  rewrite oct_add_neg_cancel.
  reflexivity.
Qed.

Theorem s54_sed_residual_block_B :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_B (s54_sed_residual D) a = s54_block_B D a.
Proof.
  intros D HD a.
  unfold s54_block_B, s54_sed_residual, sed_sub,
         schafer1954_sedenion_extend, s54_pair_derivation_to_sed,
         s54_pair_extend, s54_pair_to_sed, s54_sed_to_pair, s54_hi_embed.
  simpl.
  rewrite (s54_block_A_zero D HD).
  rewrite oct_neg_zero.
  rewrite oct_add_zero_right.
  reflexivity.
Qed.

Theorem s54_sed_residual_block_C :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_C (s54_sed_residual D) a = s54_block_C D a.
Proof.
  intros D HD a.
  unfold s54_block_C, s54_sed_residual, sed_sub,
         schafer1954_sedenion_extend, s54_pair_derivation_to_sed,
         s54_pair_extend, s54_pair_to_sed, s54_sed_to_pair, s54_oct_embed.
  simpl.
  rewrite (s54_block_A_zero D HD).
  rewrite oct_neg_zero.
  rewrite oct_add_zero_right.
  reflexivity.
Qed.

Theorem s54_sed_residual_block_E :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a : CDOct,
      s54_block_E (s54_sed_residual D) a =
      oct_add (s54_block_E D a) (oct_neg (s54_block_A D a)).
Proof.
  intros D HD a.
  unfold s54_block_E, s54_sed_residual, sed_sub,
         schafer1954_sedenion_extend, s54_pair_derivation_to_sed,
         s54_pair_extend, s54_pair_to_sed, s54_sed_to_pair, s54_hi_embed.
  simpl.
  reflexivity.
Qed.

Theorem s54_residual_block_C_twisted_derivation :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall a b : CDOct,
      s54_block_C (s54_sed_residual D) (oct_mul a b) =
      oct_add
        (oct_mul (s54_block_C (s54_sed_residual D) a) (oct_conj b))
        (oct_mul (s54_block_C (s54_sed_residual D) b) a).
Proof.
  intros D HD a b.
  pose proof
    (schafer1954_sedenion_derivation_to_octonion_pair
       (s54_sed_residual D) (s54_sed_residual_is_derivation D HD))
    as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (a, oct_zero)) (q := (b, oct_zero)).
  unfold s54_sed_derivation_to_pair, s54_block_A, s54_block_C,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_lo_lo in Hpair_mul.
  rewrite s54_pair_mul_any_lo in Hpair_mul.
  rewrite s54_pair_mul_lo_any in Hpair_mul.
  pose proof (f_equal snd Hpair_mul) as Hhi.
  cbn [snd] in Hhi.
  exact Hhi.
Qed.

Theorem s54_concrete_backward_from_residual_blocks :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    (forall a : CDOct, s54_block_B (s54_sed_residual D) a = oct_zero) ->
    (forall a : CDOct, s54_block_C (s54_sed_residual D) a = oct_zero) ->
    (forall a : CDOct, s54_block_E (s54_sed_residual D) a = oct_zero) ->
    D = schafer1954_sedenion_extend (s54_block_A D).
Proof.
  intros D HD HB HC HE.
  apply s54_concrete_backward_from_block_equalities; try exact HD.
  - intro a.
    rewrite <- s54_sed_residual_block_B by exact HD.
    exact (HB a).
  - intro a.
    rewrite <- s54_sed_residual_block_C by exact HD.
    exact (HC a).
  - intro a.
    pose proof (HE a) as HEa.
    rewrite s54_sed_residual_block_E in HEa by exact HD.
    apply s54_oct_add_neg_zero_implies_eq.
    exact HEa.
Qed.

Lemma s54_pair_mul_hi_hi_raw :
  forall a b : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (oct_zero, a) (oct_zero, b) =
    (oct_neg (oct_mul (oct_conj b) a), oct_zero).
Proof.
  intros a b.
  unfold s54_pair_mul; simpl.
  rewrite oct_mul_zero_left.
  rewrite s54_oct_scale_neg_one.
  rewrite s54_oct_conj_zero.
  repeat rewrite oct_mul_zero_right.
  repeat rewrite oct_add_zero_left.
  repeat rewrite oct_add_zero_right.
  reflexivity.
Qed.

Lemma s54_pair_y_square :
  s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
    (oct_zero, s54_oct_one) (oct_zero, s54_oct_one) =
  (oct_neg s54_oct_one, oct_zero).
Proof.
  rewrite s54_pair_mul_hi_hi_raw.
  unfold s54_oct_one.
  apply f_equal2.
  - rewrite s54_oct_conj_one.
    rewrite s54_oct_mul_one_left.
    reflexivity.
  - reflexivity.
Qed.

Lemma s54_sed_derivation_neg_one_pair_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    s54_sed_derivation_to_pair D (oct_neg s54_oct_one, oct_zero) =
    (oct_zero, oct_zero).
Proof.
  intros D HD.
  unfold s54_sed_derivation_to_pair, s54_sed_to_pair, s54_pair_to_sed.
  cbn [fst snd].
  change (sed_lo (D (mkSed (oct_neg s54_oct_one) oct_zero)))
    with (s54_block_A D (oct_neg s54_oct_one)).
  change (sed_hi (D (mkSed (oct_neg s54_oct_one) oct_zero)))
    with (s54_block_C D (oct_neg s54_oct_one)).
  rewrite <- s54_oct_scale_neg_one.
  rewrite (s54_block_A_scale D HD (-1) s54_oct_one).
  rewrite (s54_block_A_one_zero D HD).
  rewrite s54_oct_scale_zero_right.
  rewrite (s54_block_C_scale D HD (-1) s54_oct_one).
  rewrite (s54_block_C_one_zero D HD).
  rewrite s54_oct_scale_zero_right.
  reflexivity.
Qed.

Theorem s54_block_u_pure :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    oct_conj (s54_block_u D) = oct_neg (s54_block_u D).
Proof.
  intros D HD.
  pose proof (schafer1954_sedenion_derivation_to_octonion_pair D HD) as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, s54_oct_one))
                            (q := (oct_zero, s54_oct_one)).
  rewrite s54_pair_y_square in Hpair_mul.
  rewrite (s54_sed_derivation_neg_one_pair_zero D HD) in Hpair_mul.
  unfold s54_sed_derivation_to_pair,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair
    in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_any_y in Hpair_mul.
  rewrite s54_pair_mul_y_any in Hpair_mul.
  pose proof (f_equal snd Hpair_mul) as Hhi.
  cbn [snd] in Hhi.
  change (sed_lo (D (s54_hi_embed s54_oct_one))) with (s54_block_u D) in Hhi.
  change (oct_conj (sed_lo (D (s54_hi_embed s54_oct_one))))
    with (oct_conj (s54_block_u D)) in Hhi.
  symmetry in Hhi.
  rewrite oct_add_comm in Hhi.
  apply s54_oct_conj_add_self_zero_implies_pure.
  exact Hhi.
Qed.

Theorem s54_block_v_pure :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    oct_conj (s54_block_v D) = oct_neg (s54_block_v D).
Proof.
  intros D HD.
  pose proof (schafer1954_sedenion_derivation_to_octonion_pair D HD) as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, s54_oct_one))
                            (q := (oct_zero, s54_oct_one)).
  rewrite s54_pair_y_square in Hpair_mul.
  rewrite (s54_sed_derivation_neg_one_pair_zero D HD) in Hpair_mul.
  unfold s54_sed_derivation_to_pair,
         s54_pair_add, s54_pair_to_sed, s54_sed_to_pair
    in Hpair_mul |- *.
  cbn [fst snd] in Hpair_mul |- *.
  rewrite s54_pair_mul_any_y in Hpair_mul.
  rewrite s54_pair_mul_y_any in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  cbn [fst] in Hlo.
  change (sed_hi (D (s54_hi_embed s54_oct_one))) with (s54_block_v D) in Hlo.
  change (oct_neg (sed_hi (D (s54_hi_embed s54_oct_one))))
    with (oct_neg (s54_block_v D)) in Hlo.
  change (oct_neg (oct_conj (sed_hi (D (s54_hi_embed s54_oct_one)))))
    with (oct_neg (oct_conj (s54_block_v D))) in Hlo.
  symmetry in Hlo.
  apply s54_oct_add_neg_zero_implies_eq in Hlo.
  symmetry.
  exact Hlo.
Qed.

Theorem s54_residual_block_u_pure :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    oct_conj (s54_block_u (s54_sed_residual D)) =
    oct_neg (s54_block_u (s54_sed_residual D)).
Proof.
  intros D HD.
  apply s54_block_u_pure.
  exact (s54_sed_residual_is_derivation D HD).
Qed.

Theorem s54_residual_block_v_pure :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    oct_conj (s54_block_v (s54_sed_residual D)) =
    oct_neg (s54_block_v (s54_sed_residual D)).
Proof.
  intros D HD.
  apply s54_block_v_pure.
  exact (s54_sed_residual_is_derivation D HD).
Qed.

Lemma s54_oct_coords_eq_sum :
  forall r0 r1 r2 r3 r4 r5 r6 r7 : R,
    mkOct (mkQuat r0 r1 r2 r3) (mkQuat r4 r5 r6 r7) =
    oct_add (oct_scale r0 (oct_e 0))
      (oct_add (oct_scale r1 (oct_e 1))
        (oct_add (oct_scale r2 (oct_e 2))
          (oct_add (oct_scale r3 (oct_e 3))
            (oct_add (oct_scale r4 (oct_e 4))
              (oct_add (oct_scale r5 (oct_e 5))
                (oct_add (oct_scale r6 (oct_e 6))
                  (oct_scale r7 (oct_e 7)))))))).
Proof.
  intros r0 r1 r2 r3 r4 r5 r6 r7.
  unfold oct_add, oct_scale.
  cbn [oct_e oct_lo oct_hi quat_one quat_zero].
  unfold quat_add, quat_scale.
  simpl.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Record Schafer1954ResidualCoordinateSurface
    (D : CDSed -> CDSed) : Type := {
  s54_res_coord_mu : Schafer1954Basis7 -> R;
  s54_res_coord_eq36 :
    s54_block_C (s54_sed_residual D) s54_oct_one = oct_zero;
  s54_res_coord_eq50 :
    forall i : Schafer1954Basis7,
      s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
      oct_scale (s54_res_coord_mu i) (s54_basis7_oct i);
  s54_res_coord_eq51 :
    forall i j k : Schafer1954Basis7,
      schafer1954_eq52_cayley_triple i j k ->
      (s54_res_coord_mu i +
       s54_res_coord_mu j +
       s54_res_coord_mu k)%R = 0%R;
  s54_res_coord_C_eq_A :
    forall a : CDOct,
      s54_block_B (s54_sed_residual D) a =
      s54_block_A (s54_sed_residual D) a;
}.

Section Schafer1954ConcreteTheorem3Instantiation.
  Variable s54_eq30_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq31_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq32_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq37_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq38_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq39_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq40_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq42_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq46_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq47_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq48_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq49_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq50_formula : (Schafer1954Basis7 -> R) -> (CDOct -> CDOct) -> Prop.
  Variable s54_eq52_formula :
    (Schafer1954Basis7 -> Schafer1954Basis7 -> Schafer1954Basis7 -> Prop) -> Prop.

  Hypothesis s54_eq50_formula_concrete :
    forall mu B,
      s54_eq50_formula mu B ->
      forall i : Schafer1954Basis7,
        B (s54_basis7_oct i) = oct_scale (mu i) (s54_basis7_oct i).

  Definition s54_t3_concrete_coord_t : Type :=
    Schafer1954Theorem3CoordinateSurface
      CDOct oct_add oct_mul oct_zero s54_oct_one
      s54_eq30_formula s54_eq31_formula s54_eq32_formula
      s54_eq37_formula s54_eq38_formula s54_eq39_formula
      s54_eq40_formula s54_eq42_formula s54_eq46_formula
      s54_eq47_formula s54_eq48_formula s54_eq49_formula
      Schafer1954Basis7 schafer1954_basis7_tracked
      schafer1954_eq52_cayley_triple
      s54_eq50_formula s54_eq52_formula.

  Definition s54_t3_concrete_A (S : s54_t3_concrete_coord_t) : CDOct -> CDOct :=
    s54_t3_block_A_map
      CDOct oct_add oct_mul oct_zero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula
      (s54_t3_coord_block
         CDOct oct_add oct_mul oct_zero s54_oct_one
         s54_eq30_formula s54_eq31_formula s54_eq32_formula
         s54_eq37_formula s54_eq38_formula s54_eq39_formula
         s54_eq40_formula s54_eq42_formula s54_eq46_formula
         s54_eq47_formula s54_eq48_formula s54_eq49_formula
         Schafer1954Basis7 schafer1954_basis7_tracked
         schafer1954_eq52_cayley_triple
         s54_eq50_formula s54_eq52_formula S).

  Definition s54_t3_concrete_B (S : s54_t3_concrete_coord_t) : CDOct -> CDOct :=
    s54_t3_block_B_map
      CDOct oct_add oct_mul oct_zero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula
      (s54_t3_coord_block
         CDOct oct_add oct_mul oct_zero s54_oct_one
         s54_eq30_formula s54_eq31_formula s54_eq32_formula
         s54_eq37_formula s54_eq38_formula s54_eq39_formula
         s54_eq40_formula s54_eq42_formula s54_eq46_formula
         s54_eq47_formula s54_eq48_formula s54_eq49_formula
         Schafer1954Basis7 schafer1954_basis7_tracked
         schafer1954_eq52_cayley_triple
         s54_eq50_formula s54_eq52_formula S).

  Definition s54_t3_concrete_C (S : s54_t3_concrete_coord_t) : CDOct -> CDOct :=
    s54_t3_block_C_map
      CDOct oct_add oct_mul oct_zero
      s54_eq30_formula s54_eq31_formula s54_eq32_formula
      (s54_t3_coord_block
         CDOct oct_add oct_mul oct_zero s54_oct_one
         s54_eq30_formula s54_eq31_formula s54_eq32_formula
         s54_eq37_formula s54_eq38_formula s54_eq39_formula
         s54_eq40_formula s54_eq42_formula s54_eq46_formula
         s54_eq47_formula s54_eq48_formula s54_eq49_formula
         Schafer1954Basis7 schafer1954_basis7_tracked
         schafer1954_eq52_cayley_triple
         s54_eq50_formula s54_eq52_formula S).

  Definition s54_t3_concrete_mu (S : s54_t3_concrete_coord_t) :
      Schafer1954Basis7 -> R :=
    s54_t3_diag_scalar
      CDOct oct_add oct_mul oct_zero s54_oct_one
      s54_eq30_formula s54_eq31_formula s54_eq32_formula
      s54_eq37_formula s54_eq38_formula s54_eq39_formula
      s54_eq40_formula s54_eq42_formula s54_eq46_formula
      s54_eq47_formula s54_eq48_formula s54_eq49_formula
      Schafer1954Basis7 schafer1954_basis7_tracked
      schafer1954_eq52_cayley_triple
      s54_eq50_formula s54_eq52_formula S.

  Record Schafer1954ConcreteTheorem3InstantiationSurface
      (D : CDSed -> CDSed) : Type := {
    s54_t3_concrete_surface : s54_t3_concrete_coord_t;
    s54_t3_concrete_A_matches :
      forall a : CDOct,
        s54_t3_concrete_A s54_t3_concrete_surface a =
        s54_block_A (s54_sed_residual D) a;
    s54_t3_concrete_B_matches :
      forall a : CDOct,
        s54_t3_concrete_B s54_t3_concrete_surface a =
        s54_block_C (s54_sed_residual D) a;
    s54_t3_concrete_C_matches :
      forall a : CDOct,
        s54_t3_concrete_C s54_t3_concrete_surface a =
        s54_block_B (s54_sed_residual D) a;
  }.

  Definition s54_residual_coordinate_surface_of_theorem3_instantiation
      (D : CDSed -> CDSed)
      (S : Schafer1954ConcreteTheorem3InstantiationSurface D) :
      Schafer1954ResidualCoordinateSurface D.
  Proof.
    refine
      {| s54_res_coord_mu := s54_t3_concrete_mu (s54_t3_concrete_surface D S);
         s54_res_coord_eq36 := _;
         s54_res_coord_eq50 := _;
         s54_res_coord_eq51 := _;
         s54_res_coord_C_eq_A := _ |}.
    - rewrite <- (s54_t3_concrete_B_matches D S s54_oct_one).
      exact
        (s54_t3_eq36
           CDOct oct_add oct_mul oct_zero s54_oct_one
           s54_eq30_formula s54_eq31_formula s54_eq32_formula
           s54_eq37_formula s54_eq38_formula s54_eq39_formula
           s54_eq40_formula s54_eq42_formula s54_eq46_formula
           s54_eq47_formula s54_eq48_formula s54_eq49_formula
           Schafer1954Basis7 schafer1954_basis7_tracked
           schafer1954_eq52_cayley_triple
           s54_eq50_formula s54_eq52_formula
           (s54_t3_concrete_surface D S)).
    - intro i.
      rewrite <- (s54_t3_concrete_B_matches D S (s54_basis7_oct i)).
      apply s54_eq50_formula_concrete.
      exact
        (s54_t3_eq50
           CDOct oct_add oct_mul oct_zero s54_oct_one
           s54_eq30_formula s54_eq31_formula s54_eq32_formula
           s54_eq37_formula s54_eq38_formula s54_eq39_formula
           s54_eq40_formula s54_eq42_formula s54_eq46_formula
           s54_eq47_formula s54_eq48_formula s54_eq49_formula
           Schafer1954Basis7 schafer1954_basis7_tracked
           schafer1954_eq52_cayley_triple
           s54_eq50_formula s54_eq52_formula
           (s54_t3_concrete_surface D S)).
    - exact
        (s54_t3_eq51
           CDOct oct_add oct_mul oct_zero s54_oct_one
           s54_eq30_formula s54_eq31_formula s54_eq32_formula
           s54_eq37_formula s54_eq38_formula s54_eq39_formula
           s54_eq40_formula s54_eq42_formula s54_eq46_formula
           s54_eq47_formula s54_eq48_formula s54_eq49_formula
           Schafer1954Basis7 schafer1954_basis7_tracked
           schafer1954_eq52_cayley_triple
           s54_eq50_formula s54_eq52_formula
           (s54_t3_concrete_surface D S)).
    - intro a.
      rewrite <- (s54_t3_concrete_C_matches D S a).
      rewrite <- (s54_t3_concrete_A_matches D S a).
      exact
        (s54_t3_coordinate_surface_forces_C_eq_A
           CDOct oct_add oct_mul oct_zero s54_oct_one
           oct_add_zero_right oct_mul_zero_left
           s54_eq30_formula s54_eq31_formula s54_eq32_formula
           s54_eq37_formula s54_eq38_formula s54_eq39_formula
           s54_eq40_formula s54_eq42_formula s54_eq46_formula
           s54_eq47_formula s54_eq48_formula s54_eq49_formula
           Schafer1954Basis7 schafer1954_basis7_tracked
           schafer1954_eq52_cayley_triple
           s54_eq50_formula s54_eq52_formula
           (s54_t3_concrete_surface D S) a).
  Defined.

End Schafer1954ConcreteTheorem3Instantiation.

Definition s54_concrete_eq30_formula (_ : CDOct -> CDOct) (_ : CDOct) : Prop := True.
Definition s54_concrete_eq31_formula (_ : CDOct -> CDOct) (_ : CDOct) : Prop := True.
Definition s54_concrete_eq32_formula (_ : CDOct -> CDOct) (_ : CDOct) : Prop := True.
Definition s54_concrete_eq37_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq38_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq39_formula (_ : CDOct -> CDOct) (_ : CDOct) : Prop := True.
Definition s54_concrete_eq40_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq42_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq46_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq47_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq48_formula (_ : CDOct -> CDOct) : Prop := True.
Definition s54_concrete_eq49_formula (_ : CDOct -> CDOct) : Prop := True.

Definition s54_concrete_eq50_formula
    (mu : Schafer1954Basis7 -> R) (B : CDOct -> CDOct) : Prop :=
  forall i : Schafer1954Basis7,
    B (s54_basis7_oct i) = oct_scale (mu i) (s54_basis7_oct i).

Definition s54_concrete_eq52_formula
    (_ : Schafer1954Basis7 -> Schafer1954Basis7 -> Schafer1954Basis7 -> Prop) :
    Prop := True.

Lemma s54_concrete_eq50_formula_concrete :
  forall mu B,
    s54_concrete_eq50_formula mu B ->
    forall i : Schafer1954Basis7,
      B (s54_basis7_oct i) = oct_scale (mu i) (s54_basis7_oct i).
Proof.
  intros mu B Hmu i.
  exact (Hmu i).
Qed.

Ltac s54_invert_oct_equalities :=
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ =>
      inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion H; clear H; subst
  end.

Theorem s54_residual_block_C_basis_zero_direct :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall i : Schafer1954Basis7,
      s54_block_C (s54_sed_residual D) (s54_basis7_oct i) = oct_zero.
Proof.
  intros D HD i.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b1)) as C1 eqn:HC1.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b2)) as C2 eqn:HC2.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b3)) as C3 eqn:HC3.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b4)) as C4 eqn:HC4.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b5)) as C5 eqn:HC5.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b6)) as C6 eqn:HC6.
  remember (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b7)) as C7 eqn:HC7.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b1) (s54_basis7_oct s54_b2)) as H12.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b2) (s54_basis7_oct s54_b3)) as H23.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b3) (s54_basis7_oct s54_b1)) as H31.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b1) (s54_basis7_oct s54_b4)) as H14.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b4) (s54_basis7_oct s54_b5)) as H45.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b5) (s54_basis7_oct s54_b1)) as H51.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b1) (s54_basis7_oct s54_b7)) as H17.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b7) (s54_basis7_oct s54_b6)) as H76.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b6) (s54_basis7_oct s54_b1)) as H61.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b2) (s54_basis7_oct s54_b4)) as H24.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b4) (s54_basis7_oct s54_b6)) as H46.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b6) (s54_basis7_oct s54_b2)) as H62.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b2) (s54_basis7_oct s54_b5)) as H25.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b5) (s54_basis7_oct s54_b7)) as H57.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b7) (s54_basis7_oct s54_b2)) as H72.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b3) (s54_basis7_oct s54_b4)) as H34.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b4) (s54_basis7_oct s54_b7)) as H47.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b7) (s54_basis7_oct s54_b3)) as H73.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b3) (s54_basis7_oct s54_b6)) as H36.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b6) (s54_basis7_oct s54_b5)) as H65.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct s54_b5) (s54_basis7_oct s54_b3)) as H53.
  rewrite (schafer1954_eq52_basis_mul s54_b1 s54_b2 s54_b3 s54_eq52_123) in H12.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b1 s54_b2 s54_b3 s54_eq52_123) in H23.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b1 s54_b2 s54_b3 s54_eq52_123) in H31.
  rewrite (schafer1954_eq52_basis_mul s54_b1 s54_b4 s54_b5 s54_eq52_145) in H14.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b1 s54_b4 s54_b5 s54_eq52_145) in H45.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b1 s54_b4 s54_b5 s54_eq52_145) in H51.
  rewrite (schafer1954_eq52_basis_mul s54_b1 s54_b7 s54_b6 s54_eq52_167) in H17.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b1 s54_b7 s54_b6 s54_eq52_167) in H76.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b1 s54_b7 s54_b6 s54_eq52_167) in H61.
  rewrite (schafer1954_eq52_basis_mul s54_b2 s54_b4 s54_b6 s54_eq52_246) in H24.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b2 s54_b4 s54_b6 s54_eq52_246) in H46.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b2 s54_b4 s54_b6 s54_eq52_246) in H62.
  rewrite (schafer1954_eq52_basis_mul s54_b2 s54_b5 s54_b7 s54_eq52_257) in H25.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b2 s54_b5 s54_b7 s54_eq52_257) in H57.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b2 s54_b5 s54_b7 s54_eq52_257) in H72.
  rewrite (schafer1954_eq52_basis_mul s54_b3 s54_b4 s54_b7 s54_eq52_347) in H34.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b3 s54_b4 s54_b7 s54_eq52_347) in H47.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b3 s54_b4 s54_b7 s54_eq52_347) in H73.
  rewrite (schafer1954_eq52_basis_mul s54_b3 s54_b6 s54_b5 s54_eq52_356) in H36.
  rewrite (schafer1954_eq52_basis_mul_cyclic_l s54_b3 s54_b6 s54_b5 s54_eq52_356) in H65.
  rewrite (schafer1954_eq52_basis_mul_cyclic_r s54_b3 s54_b6 s54_b5 s54_eq52_356) in H53.
  repeat rewrite schafer1954_basis7_conj_neg in
    H12, H23, H31, H14, H45, H51, H17, H76, H61,
    H24, H46, H62, H25, H57, H72, H34, H47, H73,
    H36, H65, H53.
  repeat rewrite <- HC1 in H23, H45, H76.
  repeat rewrite <- HC2 in H31, H46, H57.
  repeat rewrite <- HC3 in H12, H47, H65.
  repeat rewrite <- HC4 in H51, H62, H73.
  repeat rewrite <- HC5 in H14, H72, H36.
  repeat rewrite <- HC6 in H17, H24, H53.
  repeat rewrite <- HC7 in H61, H25, H34.
  destruct C1 as [[c10 c11 c12 c13] [c14 c15 c16 c17]].
  destruct C2 as [[c20 c21 c22 c23] [c24 c25 c26 c27]].
  destruct C3 as [[c30 c31 c32 c33] [c34 c35 c36 c37]].
  destruct C4 as [[c40 c41 c42 c43] [c44 c45 c46 c47]].
  destruct C5 as [[c50 c51 c52 c53] [c54 c55 c56 c57]].
  destruct C6 as [[c60 c61 c62 c63] [c64 c65 c66 c67]].
  destruct C7 as [[c70 c71 c72 c73] [c74 c75 c76 c77]].
  cbv [oct_add oct_mul oct_neg oct_conj oct_zero oct_e
       oct_lo oct_hi quat_add quat_mul quat_neg quat_conj
       qa qb qc qd quat_zero quat_one] in
    H12, H23, H31, H14, H45, H51, H17, H76, H61,
    H24, H46, H62, H25, H57, H72, H34, H47, H73,
    H36, H65, H53 |- *.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ =>
      inversion H; clear H
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ =>
      inversion H; clear H
  end.
  cbv [s54_basis7_oct oct_e oct_lo oct_hi qa qb qc qd quat_zero quat_one] in *.
  repeat match goal with
  | H : _ |- _ => ring_simplify in H
  end.
  destruct i;
    [rewrite <- HC1 | rewrite <- HC2 | rewrite <- HC3
    |rewrite <- HC4 | rewrite <- HC5 | rewrite <- HC6 | rewrite <- HC7];
    clear HC1 HC2 HC3 HC4 HC5 HC6 HC7;
    cbv [oct_zero];
    apply (f_equal2 mkOct);
    apply (f_equal4 mkQuat);
    nra.
Qed.

Theorem s54_residual_eq50_builder_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    { mu : Schafer1954Basis7 -> R &
      forall i : Schafer1954Basis7,
        s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
        oct_scale (mu i) (s54_basis7_oct i) }.
Proof.
  intros D HD.
  exists (fun _ : Schafer1954Basis7 => 0%R).
  intro i.
  rewrite (s54_residual_block_C_basis_zero_direct D HD i).
  apply oct_scale_zero.
Qed.

Theorem s54_residual_block_C_eq51_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      forall i j k : Schafer1954Basis7,
        schafer1954_eq52_cayley_triple i j k ->
        (mu i + mu j + mu k)%R = 0%R.
Proof.
  intros D HD mu Hdiag i j k Htr.
  pose proof
    (s54_residual_block_C_twisted_derivation
       D HD (s54_basis7_oct i) (s54_basis7_oct j)) as HC.
  rewrite (schafer1954_eq52_basis_mul i j k Htr) in HC.
  rewrite (Hdiag i) in HC.
  rewrite (Hdiag j) in HC.
  rewrite (Hdiag k) in HC.
  rewrite (schafer1954_basis7_conj_neg j) in HC.
  rewrite <- s54_oct_scale_neg_one in HC.
  rewrite oct_mul_scale_left in HC.
  rewrite oct_mul_scale_right in HC.
  rewrite oct_mul_scale_left in HC.
  rewrite (schafer1954_eq52_basis_mul i j k Htr) in HC.
  rewrite (schafer1954_eq52_basis_mul_rev i j k Htr) in HC.
  pose proof (f_equal (s54_basis7_coeff k) HC) as Hcoeff.
  destruct Htr;
    vm_compute in Hcoeff;
    ring_simplify in Hcoeff;
    lra.
Qed.

Theorem s54_residual_block_C_basis_zero_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      forall i : Schafer1954Basis7,
        s54_block_C (s54_sed_residual D) (s54_basis7_oct i) = oct_zero.
Proof.
  intros D HD mu Hdiag i.
  assert (Hmu0 : mu i = 0%R).
  {
    apply (schafer1954_eq52_real_solution_zero mu).
    intros j k l Htr.
    exact (s54_residual_block_C_eq51_of_eq50 D HD mu Hdiag j k l Htr).
  }
  rewrite (Hdiag i).
  rewrite Hmu0.
  apply oct_scale_zero.
Qed.

Theorem s54_residual_block_C_zero_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      forall a : CDOct,
        s54_block_C (s54_sed_residual D) a = oct_zero.
Proof.
  intros D HD mu Hdiag [[r0 r1 r2 r3] [r4 r5 r6 r7]].
  rewrite s54_oct_coords_eq_sum.
  repeat rewrite s54_block_C_add by exact (s54_sed_residual_is_derivation D HD).
  repeat rewrite s54_block_C_scale by exact (s54_sed_residual_is_derivation D HD).
  change (s54_block_C (s54_sed_residual D) (oct_e 0)) with
    (s54_block_C (s54_sed_residual D) s54_oct_one).
  change (s54_block_C (s54_sed_residual D) (oct_e 1)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b1)).
  change (s54_block_C (s54_sed_residual D) (oct_e 2)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b2)).
  change (s54_block_C (s54_sed_residual D) (oct_e 3)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b3)).
  change (s54_block_C (s54_sed_residual D) (oct_e 4)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b4)).
  change (s54_block_C (s54_sed_residual D) (oct_e 5)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b5)).
  change (s54_block_C (s54_sed_residual D) (oct_e 6)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b6)).
  change (s54_block_C (s54_sed_residual D) (oct_e 7)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b7)).
  rewrite (s54_block_C_one_zero (s54_sed_residual D)
             (s54_sed_residual_is_derivation D HD)).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b1).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b2).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b3).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b4).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b5).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b6).
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag s54_b7).
  repeat rewrite s54_oct_scale_zero_right.
  repeat rewrite oct_add_zero_left.
  reflexivity.
Qed.

Theorem s54_residual_block_u_anticomm_basis_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      forall i : Schafer1954Basis7,
        oct_mul (s54_block_u (s54_sed_residual D)) (s54_basis7_oct i) =
        oct_neg
          (oct_mul (s54_basis7_oct i) (s54_block_u (s54_sed_residual D))).
Proof.
  intros D HD mu Hdiag i.
  destruct (s54_blocks_from_right_generator
              (s54_sed_residual D)
              (s54_sed_residual_is_derivation D HD)
              (s54_basis7_oct i)) as [Hright _].
  destruct (s54_blocks_from_left_generator
              (s54_sed_residual D)
              (s54_sed_residual_is_derivation D HD)
              (s54_basis7_oct i)) as [Hleft _].
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag i) in Hright.
  rewrite (s54_residual_block_C_basis_zero_of_eq50 D HD mu Hdiag i) in Hleft.
  rewrite oct_neg_zero in Hright.
  rewrite s54_oct_conj_zero in Hleft.
  rewrite oct_neg_zero in Hleft.
  rewrite oct_add_zero_right in Hright.
  rewrite oct_add_zero_right in Hleft.
  rewrite (schafer1954_basis7_conj_neg i) in Hleft.
  rewrite <- s54_oct_scale_neg_one in Hleft.
  rewrite (s54_block_B_scale
             (s54_sed_residual D)
             (s54_sed_residual_is_derivation D HD)
             (-1) (s54_basis7_oct i)) in Hleft.
  rewrite s54_oct_scale_neg_one in Hleft.
  rewrite Hright in Hleft.
  symmetry.
  exact Hleft.
Qed.

Lemma s54_oct_pure_anticomm_basis_zero_raw :
  forall v : CDOct,
    oct_conj v = oct_neg v ->
    oct_mul v (oct_e 1) = oct_neg (oct_mul (oct_e 1) v) ->
    oct_mul v (oct_e 2) = oct_neg (oct_mul (oct_e 2) v) ->
    oct_mul v (oct_e 3) = oct_neg (oct_mul (oct_e 3) v) ->
    oct_mul v (oct_e 4) = oct_neg (oct_mul (oct_e 4) v) ->
    oct_mul v (oct_e 5) = oct_neg (oct_mul (oct_e 5) v) ->
    oct_mul v (oct_e 6) = oct_neg (oct_mul (oct_e 6) v) ->
    oct_mul v (oct_e 7) = oct_neg (oct_mul (oct_e 7) v) ->
    v = oct_zero.
Proof.
  intros [[v0 v1 v2 v3] [v4 v5 v6 v7]] Hpure H1 H2 H3 H4 H5 H6 H7.
  cbv [oct_conj oct_neg oct_mul oct_e oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd] in Hpure, H1, H2, H3, H4, H5, H6, H7 |- *.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ => inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion H; clear H; subst
  end.
  assert (Hv0 : v0 = 0%R) by lra.
  assert (Hv1 : v1 = 0%R) by lra.
  assert (Hv2 : v2 = 0%R) by lra.
  assert (Hv3 : v3 = 0%R) by lra.
  assert (Hv4 : v4 = 0%R) by lra.
  assert (Hv5 : v5 = 0%R) by lra.
  assert (Hv6 : v6 = 0%R) by lra.
  assert (Hv7 : v7 = 0%R) by lra.
  subst.
  cbv [oct_zero quat_zero].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Theorem s54_residual_block_u_zero_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      s54_block_u (s54_sed_residual D) = oct_zero.
Proof.
  intros D HD mu Hdiag.
  apply s54_oct_pure_anticomm_basis_zero_raw.
  - exact (s54_residual_block_u_pure D HD).
  - change (s54_basis7_oct s54_b1) with (oct_e 1).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b1).
  - change (s54_basis7_oct s54_b2) with (oct_e 2).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b2).
  - change (s54_basis7_oct s54_b3) with (oct_e 3).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b3).
  - change (s54_basis7_oct s54_b4) with (oct_e 4).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b4).
  - change (s54_basis7_oct s54_b5) with (oct_e 5).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b5).
  - change (s54_basis7_oct s54_b6) with (oct_e 6).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b6).
  - change (s54_basis7_oct s54_b7) with (oct_e 7).
    exact (s54_residual_block_u_anticomm_basis_of_eq50 D HD mu Hdiag s54_b7).
Qed.

Theorem s54_residual_block_B_eq_A_of_eq50 :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall mu : Schafer1954Basis7 -> R,
      (forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i)) ->
      forall a : CDOct,
        s54_block_B (s54_sed_residual D) a =
        s54_block_A (s54_sed_residual D) a.
Proof.
  intros D HD mu Hdiag a.
  destruct (s54_blocks_from_right_generator
              (s54_sed_residual D)
              (s54_sed_residual_is_derivation D HD)
              a) as [HB _].
  rewrite (s54_residual_block_u_zero_of_eq50 D HD mu Hdiag) in HB.
  rewrite oct_mul_zero_right in HB.
  rewrite (s54_residual_block_C_zero_of_eq50 D HD mu Hdiag a) in HB.
  rewrite oct_neg_zero in HB.
  rewrite oct_add_zero_left in HB.
  rewrite (s54_sed_residual_block_A_zero D a).
  exact HB.
Qed.

Record Schafer1954ConcreteResidualTheorem3Facts
    (D : CDSed -> CDSed) : Type := {
  s54_concrete_residual_mu : Schafer1954Basis7 -> R;
  s54_concrete_residual_eq50 :
    forall i : Schafer1954Basis7,
      s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
      oct_scale (s54_concrete_residual_mu i) (s54_basis7_oct i);
  s54_concrete_residual_eq51 :
    forall i j k : Schafer1954Basis7,
      schafer1954_eq52_cayley_triple i j k ->
      (s54_concrete_residual_mu i +
       s54_concrete_residual_mu j +
       s54_concrete_residual_mu k)%R = 0%R;
  s54_concrete_residual_B_eq_A :
    forall a : CDOct,
      s54_block_B (s54_sed_residual D) a =
      s54_block_A (s54_sed_residual D) a;
}.

Record Schafer1954ConcreteResidualDiagonalFacts
    (D : CDSed -> CDSed) : Type := {
  s54_concrete_residual_diag_mu : Schafer1954Basis7 -> R;
  s54_concrete_residual_diag_eq50 :
    forall i : Schafer1954Basis7,
      s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
      oct_scale (s54_concrete_residual_diag_mu i) (s54_basis7_oct i);
  s54_concrete_residual_diag_B_eq_A :
    forall a : CDOct,
      s54_block_B (s54_sed_residual D) a =
      s54_block_A (s54_sed_residual D) a;
}.

Definition s54_concrete_residual_diagonal_facts_of_eq50
    (D : CDSed -> CDSed)
    (HD : Schafer1954IsSedenionDerivation D)
    (mu : Schafer1954Basis7 -> R)
    (Hdiag : forall i : Schafer1954Basis7,
        s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
        oct_scale (mu i) (s54_basis7_oct i)) :
    Schafer1954ConcreteResidualDiagonalFacts D.
Proof.
  refine
    {| s54_concrete_residual_diag_mu := mu;
       s54_concrete_residual_diag_eq50 := Hdiag;
       s54_concrete_residual_diag_B_eq_A := _ |}.
  intro a.
  exact (s54_residual_block_B_eq_A_of_eq50 D HD mu Hdiag a).
Defined.

Definition s54_concrete_residual_theorem3_facts_of_diagonal_facts
    (D : CDSed -> CDSed)
    (HD : Schafer1954IsSedenionDerivation D)
    (F : Schafer1954ConcreteResidualDiagonalFacts D) :
    Schafer1954ConcreteResidualTheorem3Facts D.
Proof.
  refine
    {| s54_concrete_residual_mu := s54_concrete_residual_diag_mu D F;
       s54_concrete_residual_eq50 := s54_concrete_residual_diag_eq50 D F;
       s54_concrete_residual_eq51 := _;
       s54_concrete_residual_B_eq_A :=
         s54_concrete_residual_diag_B_eq_A D F |}.
  intros i j k Htr.
  exact
    (s54_residual_block_C_eq51_of_eq50
       D HD (s54_concrete_residual_diag_mu D F)
       (s54_concrete_residual_diag_eq50 D F)
       i j k Htr).
Defined.

Definition s54_concrete_residual_theorem3_facts_of_residual_coordinates
    (D : CDSed -> CDSed)
    (S : Schafer1954ResidualCoordinateSurface D) :
    Schafer1954ConcreteResidualTheorem3Facts D.
Proof.
  refine
    {| s54_concrete_residual_mu := s54_res_coord_mu D S;
       s54_concrete_residual_eq50 := s54_res_coord_eq50 D S;
       s54_concrete_residual_eq51 := s54_res_coord_eq51 D S;
       s54_concrete_residual_B_eq_A := s54_res_coord_C_eq_A D S |}.
Defined.

Definition s54_concrete_theorem3_instantiation_of_residual_facts
    (D : CDSed -> CDSed)
    (HD : Schafer1954IsSedenionDerivation D)
    (F : Schafer1954ConcreteResidualTheorem3Facts D) :
    Schafer1954ConcreteTheorem3InstantiationSurface
      s54_concrete_eq30_formula s54_concrete_eq31_formula s54_concrete_eq32_formula
      s54_concrete_eq37_formula s54_concrete_eq38_formula s54_concrete_eq39_formula
      s54_concrete_eq40_formula s54_concrete_eq42_formula s54_concrete_eq46_formula
      s54_concrete_eq47_formula s54_concrete_eq48_formula s54_concrete_eq49_formula
      s54_concrete_eq50_formula s54_concrete_eq52_formula
      D.
Proof.
  pose (Block :=
    ({| s54_t3_A := s54_block_A (s54_sed_residual D);
       s54_t3_B := s54_block_C (s54_sed_residual D);
       s54_t3_C := s54_block_A (s54_sed_residual D);
       s54_t3_c := oct_zero;
       s54_t3_eq29 :=
         s54_block_A_is_derivation (s54_sed_residual D)
           (s54_sed_residual_is_derivation D HD);
       s54_t3_eq30 := fun _ => I;
       s54_t3_eq31 := fun _ => I;
       s54_t3_eq32 := fun _ => I;
       s54_t3_eq33 :=
         s54_block_A_is_derivation (s54_sed_residual D)
           (s54_sed_residual_is_derivation D HD);
       s54_t3_eq34 := fun x =>
         eq_sym
           (eq_trans
              (f_equal (oct_add (s54_block_A (s54_sed_residual D) x))
                       (oct_mul_zero_left x))
              (oct_add_zero_right (s54_block_A (s54_sed_residual D) x)));
       s54_t3_eq35 := eq_refl |}
      : Schafer1954Theorem3BlockSurface
          CDOct oct_add oct_mul oct_zero
          s54_concrete_eq30_formula s54_concrete_eq31_formula s54_concrete_eq32_formula)).
  pose (Coord :=
    ({| s54_t3_coord_block := Block;
       s54_t3_eq36 :=
         s54_block_C_one_zero (s54_sed_residual D)
           (s54_sed_residual_is_derivation D HD);
       s54_t3_eq37 := I;
       s54_t3_eq38 := I;
       s54_t3_eq39 := fun _ => I;
       s54_t3_eq40 := I;
       s54_t3_eq42 := I;
       s54_t3_eq46 := I;
       s54_t3_eq47 := I;
       s54_t3_eq48 := I;
       s54_t3_eq49 := I;
       s54_t3_diag_scalar := s54_concrete_residual_mu D F;
       s54_t3_eq50 := s54_concrete_residual_eq50 D F;
       s54_t3_eq51 := s54_concrete_residual_eq51 D F;
       s54_t3_eq52 := I;
       s54_t3_eq51_eq52_force_zero := fun i _ =>
         schafer1954_eq52_real_solution_zero (s54_concrete_residual_mu D F)
           (fun j k l Htrl => s54_concrete_residual_eq51 D F j k l Htrl) i |}
      : Schafer1954Theorem3CoordinateSurface
          CDOct oct_add oct_mul oct_zero s54_oct_one
          s54_concrete_eq30_formula s54_concrete_eq31_formula s54_concrete_eq32_formula
          s54_concrete_eq37_formula s54_concrete_eq38_formula s54_concrete_eq39_formula
          s54_concrete_eq40_formula s54_concrete_eq42_formula s54_concrete_eq46_formula
          s54_concrete_eq47_formula s54_concrete_eq48_formula s54_concrete_eq49_formula
          Schafer1954Basis7 schafer1954_basis7_tracked
          schafer1954_eq52_cayley_triple
          s54_concrete_eq50_formula s54_concrete_eq52_formula)).
  refine
    {| s54_t3_concrete_surface := Coord;
       s54_t3_concrete_A_matches := _;
       s54_t3_concrete_B_matches := _;
       s54_t3_concrete_C_matches := _ |}.
  - intro a. reflexivity.
  - intro a. reflexivity.
  - intro a.
    rewrite (s54_concrete_residual_B_eq_A D F a).
    reflexivity.
Defined.

Theorem s54_residual_coordinate_mu_zero :
  forall D : CDSed -> CDSed,
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall i : Schafer1954Basis7,
      s54_res_coord_mu D S i = 0%R.
Proof.
  intros D S i.
  apply (schafer1954_eq52_real_solution_zero (s54_res_coord_mu D S)).
  intros j k l Htrl.
  exact (s54_res_coord_eq51 D S j k l Htrl).
Qed.

Theorem s54_residual_coordinate_C_basis_zero :
  forall D : CDSed -> CDSed,
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall i : Schafer1954Basis7,
      s54_block_C (s54_sed_residual D) (s54_basis7_oct i) = oct_zero.
Proof.
  intros D S i.
  rewrite (s54_res_coord_eq50 D S i).
  rewrite (s54_residual_coordinate_mu_zero D S i).
  apply oct_scale_zero.
Qed.

Theorem s54_residual_coordinate_B_zero :
  forall D : CDSed -> CDSed,
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_block_B (s54_sed_residual D) a = oct_zero.
Proof.
  intros D S a.
  rewrite (s54_res_coord_C_eq_A D S a).
  apply s54_sed_residual_block_A_zero.
Qed.

Theorem s54_residual_coordinate_C_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_block_C (s54_sed_residual D) a = oct_zero.
Proof.
  intros D HD S [[r0 r1 r2 r3] [r4 r5 r6 r7]].
  rewrite s54_oct_coords_eq_sum.
  repeat rewrite s54_block_C_add by exact (s54_sed_residual_is_derivation D HD).
  repeat rewrite s54_block_C_scale by exact (s54_sed_residual_is_derivation D HD).
  change (s54_block_C (s54_sed_residual D) (oct_e 0)) with
    (s54_block_C (s54_sed_residual D) s54_oct_one).
  change (s54_block_C (s54_sed_residual D) (oct_e 1)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b1)).
  change (s54_block_C (s54_sed_residual D) (oct_e 2)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b2)).
  change (s54_block_C (s54_sed_residual D) (oct_e 3)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b3)).
  change (s54_block_C (s54_sed_residual D) (oct_e 4)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b4)).
  change (s54_block_C (s54_sed_residual D) (oct_e 5)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b5)).
  change (s54_block_C (s54_sed_residual D) (oct_e 6)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b6)).
  change (s54_block_C (s54_sed_residual D) (oct_e 7)) with
    (s54_block_C (s54_sed_residual D) (s54_basis7_oct s54_b7)).
  rewrite (s54_res_coord_eq36 D S).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b1).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b2).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b3).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b4).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b5).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b6).
  rewrite (s54_residual_coordinate_C_basis_zero D S s54_b7).
  repeat rewrite s54_oct_scale_zero_right.
  repeat rewrite oct_add_zero_left.
  reflexivity.
Qed.

Theorem s54_residual_coordinate_E_formula :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_block_E (s54_sed_residual D) a =
      oct_mul (s54_block_v (s54_sed_residual D)) a.
Proof.
  intros D HD S a.
  destruct (s54_blocks_from_right_generator
              (s54_sed_residual D)
              (s54_sed_residual_is_derivation D HD)
              a) as [_ HE].
  rewrite (s54_sed_residual_block_A_zero D a) in HE.
  rewrite oct_add_zero_left in HE.
  exact HE.
Qed.

Lemma s54_pair_mul_hi_hi :
  forall a b : CDOct,
    s54_pair_mul CDOct oct_add oct_mul oct_scale oct_conj (-1)
      (oct_zero, a) (oct_zero, b) =
    (oct_neg (oct_mul (oct_conj b) a), oct_zero).
Proof.
  intros a b.
  unfold s54_pair_mul; simpl.
  rewrite oct_mul_zero_left.
  rewrite s54_oct_scale_neg_one.
  rewrite s54_oct_conj_zero.
  repeat rewrite oct_mul_zero_right.
  repeat rewrite oct_add_zero_left.
  repeat rewrite oct_add_zero_right.
  reflexivity.
Qed.

Theorem s54_residual_coordinate_pair_lo_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_sed_derivation_to_pair (s54_sed_residual D) (a, oct_zero) =
      (oct_zero, oct_zero).
Proof.
  intros D HD S a.
  unfold s54_sed_derivation_to_pair, s54_pair_to_sed.
  cbn [fst snd].
  rewrite s54_blocks_decompose_sedenion_derivation
    with (D := s54_sed_residual D)
         (a := a) (b := oct_zero).
  - rewrite (s54_sed_residual_block_A_zero D a).
    rewrite (s54_residual_coordinate_B_zero D S oct_zero).
    rewrite (s54_residual_coordinate_C_zero D HD S a).
    rewrite (s54_residual_coordinate_E_formula D HD S oct_zero).
    unfold s54_sed_to_pair.
    simpl.
    rewrite oct_mul_zero_right.
    repeat rewrite oct_add_zero_left.
    repeat rewrite oct_add_zero_right.
    reflexivity.
  - exact (s54_sed_residual_is_derivation D HD).
Qed.

Theorem s54_residual_coordinate_pair_hi :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_sed_derivation_to_pair (s54_sed_residual D) (oct_zero, a) =
      (oct_zero, oct_mul (s54_block_v (s54_sed_residual D)) a).
Proof.
  intros D HD S a.
  unfold s54_sed_derivation_to_pair, s54_pair_to_sed.
  cbn [fst snd].
  rewrite s54_blocks_decompose_sedenion_derivation
    with (D := s54_sed_residual D)
         (a := oct_zero) (b := a).
  - rewrite (s54_sed_residual_block_A_zero D oct_zero).
    rewrite (s54_residual_coordinate_B_zero D S a).
    rewrite (s54_residual_coordinate_C_zero D HD S oct_zero).
    rewrite (s54_residual_coordinate_E_formula D HD S a).
    unfold s54_sed_to_pair.
    simpl.
    repeat rewrite oct_add_zero_left.
    repeat rewrite oct_add_zero_right.
    reflexivity.
  - exact (s54_sed_residual_is_derivation D HD).
Qed.

Theorem s54_residual_coordinate_v_pure :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      oct_conj (s54_block_v (s54_sed_residual D)) =
      oct_neg (s54_block_v (s54_sed_residual D)).
Proof.
  intros D HD S.
  pose proof
    (schafer1954_sedenion_derivation_to_octonion_pair
       (s54_sed_residual D) (s54_sed_residual_is_derivation D HD))
    as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, s54_oct_one))
                            (q := (oct_zero, s54_oct_one)).
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_lo_zero
             D HD S
             (oct_neg (oct_mul (oct_conj s54_oct_one) s54_oct_one))) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S s54_oct_one) in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  unfold s54_pair_add in Hlo.
  cbn [fst snd] in Hlo.
  repeat rewrite s54_oct_mul_one_left in Hlo.
  repeat rewrite oct_mul_one_right in Hlo.
  symmetry in Hlo.
  apply s54_oct_add_neg_zero_implies_eq in Hlo.
  unfold s54_oct_one in Hlo.
  simpl in Hlo.
  assert (Hone :
    oct_conj {| oct_lo := quat_one; oct_hi := quat_zero |} =
    {| oct_lo := quat_one; oct_hi := quat_zero |}).
  {
    cbv [oct_conj quat_conj quat_neg quat_zero quat_one
         oct_lo oct_hi qa qb qc qd].
    apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
  }
  rewrite Hone in Hlo.
  repeat rewrite s54_oct_mul_one_left in Hlo.
  repeat rewrite oct_mul_one_right in Hlo.
  symmetry.
  exact Hlo.
Qed.

Theorem s54_residual_coordinate_middle_nucleus_21_raw :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      oct_mul (oct_mul (oct_conj (oct_e 2))
                       (s54_block_v (s54_sed_residual D)))
              (oct_e 1) =
      oct_mul (oct_conj (oct_e 2))
              (oct_mul (s54_block_v (s54_sed_residual D)) (oct_e 1)).
Proof.
  intros D HD S.
  pose proof
    (schafer1954_sedenion_derivation_to_octonion_pair
       (s54_sed_residual D) (s54_sed_residual_is_derivation D HD))
    as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, oct_e 1))
                            (q := (oct_zero, oct_e 2)).
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_lo_zero
             D HD S
             (oct_neg (oct_mul (oct_conj (oct_e 2)) (oct_e 1)))) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 1)) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 2)) in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  unfold s54_pair_add in Hlo.
  cbn [fst snd] in Hlo.
  symmetry in Hlo.
  apply s54_oct_add_neg_zero_implies_eq in Hlo.
  rewrite oct_conj_antimorphism in Hlo.
  rewrite (s54_residual_coordinate_v_pure D HD S) in Hlo.
  rewrite oct_neg_mul_right in Hlo.
  rewrite oct_neg_mul_left in Hlo.
  apply s54_oct_neg_injective in Hlo.
  symmetry.
  exact Hlo.
Qed.

Theorem s54_residual_coordinate_middle_nucleus_41_raw :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      oct_mul (oct_mul (oct_conj (oct_e 4))
                       (s54_block_v (s54_sed_residual D)))
              (oct_e 1) =
      oct_mul (oct_conj (oct_e 4))
              (oct_mul (s54_block_v (s54_sed_residual D)) (oct_e 1)).
Proof.
  intros D HD S.
  pose proof
    (schafer1954_sedenion_derivation_to_octonion_pair
       (s54_sed_residual D) (s54_sed_residual_is_derivation D HD))
    as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, oct_e 1))
                            (q := (oct_zero, oct_e 4)).
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_lo_zero
             D HD S
             (oct_neg (oct_mul (oct_conj (oct_e 4)) (oct_e 1)))) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 1)) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 4)) in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  unfold s54_pair_add in Hlo.
  cbn [fst snd] in Hlo.
  symmetry in Hlo.
  apply s54_oct_add_neg_zero_implies_eq in Hlo.
  rewrite oct_conj_antimorphism in Hlo.
  rewrite (s54_residual_coordinate_v_pure D HD S) in Hlo.
  rewrite oct_neg_mul_right in Hlo.
  rewrite oct_neg_mul_left in Hlo.
  apply s54_oct_neg_injective in Hlo.
  symmetry.
  exact Hlo.
Qed.

Theorem s54_residual_coordinate_middle_nucleus_24_raw :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      oct_mul (oct_mul (oct_conj (oct_e 2))
                       (s54_block_v (s54_sed_residual D)))
              (oct_e 4) =
      oct_mul (oct_conj (oct_e 2))
              (oct_mul (s54_block_v (s54_sed_residual D)) (oct_e 4)).
Proof.
  intros D HD S.
  pose proof
    (schafer1954_sedenion_derivation_to_octonion_pair
       (s54_sed_residual D) (s54_sed_residual_is_derivation D HD))
    as Hpair.
  destruct Hpair as [_ _ Hpair_mul].
  specialize Hpair_mul with (p := (oct_zero, oct_e 4))
                            (q := (oct_zero, oct_e 2)).
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_lo_zero
             D HD S
             (oct_neg (oct_mul (oct_conj (oct_e 2)) (oct_e 4)))) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 4)) in Hpair_mul.
  rewrite (s54_residual_coordinate_pair_hi D HD S (oct_e 2)) in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  rewrite s54_pair_mul_hi_hi in Hpair_mul.
  pose proof (f_equal fst Hpair_mul) as Hlo.
  unfold s54_pair_add in Hlo.
  cbn [fst snd] in Hlo.
  symmetry in Hlo.
  apply s54_oct_add_neg_zero_implies_eq in Hlo.
  rewrite oct_conj_antimorphism in Hlo.
  rewrite (s54_residual_coordinate_v_pure D HD S) in Hlo.
  rewrite oct_neg_mul_right in Hlo.
  rewrite oct_neg_mul_left in Hlo.
  apply s54_oct_neg_injective in Hlo.
  symmetry.
  exact Hlo.
Qed.

Lemma s54_oct_pure_raw_component0 :
  forall v0 v1 v2 v3 v4 v5 v6 v7 : R,
    let v := mkOct (mkQuat v0 v1 v2 v3) (mkQuat v4 v5 v6 v7) in
    oct_conj v = oct_neg v ->
    v0 = 0%R.
Proof.
  intros v0 v1 v2 v3 v4 v5 v6 v7 v Hpure.
  cbv [v oct_conj oct_neg oct_lo oct_hi
       quat_conj quat_neg qa qb qc qd] in Hpure.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ => inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion H; clear H; subst
  end.
  lra.
Qed.

Lemma s54_oct_middle_raw_21_components :
  forall v0 v1 v2 v3 v4 v5 v6 v7 : R,
    let v := mkOct (mkQuat v0 v1 v2 v3) (mkQuat v4 v5 v6 v7) in
    oct_mul (oct_mul (oct_conj (oct_e 2)) v) (oct_e 1) =
    oct_mul (oct_conj (oct_e 2)) (oct_mul v (oct_e 1)) ->
    v4 = 0%R /\ v5 = 0%R /\ v6 = 0%R /\ v7 = 0%R.
Proof.
  intros v0 v1 v2 v3 v4 v5 v6 v7 v H21.
  cbv [v oct_conj oct_mul oct_e oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd] in H21.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ => inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion H; clear H; subst
  end.
  repeat split; lra.
Qed.

Lemma s54_oct_middle_raw_41_components :
  forall v0 v1 v2 v3 v4 v5 v6 v7 : R,
    let v := mkOct (mkQuat v0 v1 v2 v3) (mkQuat v4 v5 v6 v7) in
    oct_mul (oct_mul (oct_conj (oct_e 4)) v) (oct_e 1) =
    oct_mul (oct_conj (oct_e 4)) (oct_mul v (oct_e 1)) ->
    v2 = 0%R /\ v3 = 0%R /\ v6 = 0%R /\ v7 = 0%R.
Proof.
  intros v0 v1 v2 v3 v4 v5 v6 v7 v H41.
  cbv [v oct_conj oct_mul oct_e oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd] in H41.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ => inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion H; clear H; subst
  end.
  repeat split; lra.
Qed.

Lemma s54_oct_middle_raw_24_components :
  forall v0 v1 v2 v3 v4 v5 v6 v7 : R,
    let v := mkOct (mkQuat v0 v1 v2 v3) (mkQuat v4 v5 v6 v7) in
    oct_mul (oct_mul (oct_conj (oct_e 2)) v) (oct_e 4) =
    oct_mul (oct_conj (oct_e 2)) (oct_mul v (oct_e 4)) ->
    v1 = 0%R /\ v3 = 0%R /\ v5 = 0%R /\ v7 = 0%R.
Proof.
  intros v0 v1 v2 v3 v4 v5 v6 v7 v H24.
  cbv [v oct_conj oct_mul oct_e oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd] in H24.
  repeat match goal with
  | H : mkOct _ _ = mkOct _ _ |- _ => inversion H; clear H; subst
  | H : mkQuat _ _ _ _ = mkQuat _ _ _ _ |- _ => inversion H; clear H; subst
  end.
  repeat split; lra.
Qed.

Theorem s54_oct_middle_nucleus_pure_zero_raw :
  forall v : CDOct,
    oct_conj v = oct_neg v ->
    oct_mul (oct_mul (oct_conj (oct_e 2)) v) (oct_e 1) =
    oct_mul (oct_conj (oct_e 2)) (oct_mul v (oct_e 1)) ->
    oct_mul (oct_mul (oct_conj (oct_e 4)) v) (oct_e 1) =
    oct_mul (oct_conj (oct_e 4)) (oct_mul v (oct_e 1)) ->
    oct_mul (oct_mul (oct_conj (oct_e 2)) v) (oct_e 4) =
    oct_mul (oct_conj (oct_e 2)) (oct_mul v (oct_e 4)) ->
    v = oct_zero.
Proof.
  intros [[v0 v1 v2 v3] [v4 v5 v6 v7]] Hpure H21 H41 H24.
  assert (Hv0 : v0 = 0%R).
  { eapply s54_oct_pure_raw_component0; exact Hpure. }
  destruct (s54_oct_middle_raw_21_components
              v0 v1 v2 v3 v4 v5 v6 v7 H21)
    as [Hv4 [Hv5 [Hv6 Hv7]]].
  destruct (s54_oct_middle_raw_41_components
              v0 v1 v2 v3 v4 v5 v6 v7 H41)
    as [Hv2 [Hv3 [_ _]]].
  destruct (s54_oct_middle_raw_24_components
              v0 v1 v2 v3 v4 v5 v6 v7 H24)
    as [Hv1 [_ [_ _]]].
  subst.
  reflexivity.
Qed.

Theorem s54_residual_coordinate_E_zero :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
    forall a : CDOct,
      s54_block_E (s54_sed_residual D) a = oct_zero.
Proof.
  intros D HD S a.
  rewrite (s54_residual_coordinate_E_formula D HD S a).
  assert
    (Hv :
      s54_block_v (s54_sed_residual D) = oct_zero).
  {
    apply s54_oct_middle_nucleus_pure_zero_raw.
    - exact (s54_residual_coordinate_v_pure D HD S).
    - exact (s54_residual_coordinate_middle_nucleus_21_raw D HD S).
    - exact (s54_residual_coordinate_middle_nucleus_41_raw D HD S).
    - exact (s54_residual_coordinate_middle_nucleus_24_raw D HD S).
  }
  rewrite Hv.
  apply oct_mul_zero_left.
Qed.

Theorem s54_concrete_backward_from_residual_coordinates :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      D = schafer1954_sedenion_extend (s54_block_A D).
Proof.
  intros D HD S.
  apply s54_concrete_backward_from_residual_blocks; try exact HD.
  - exact (s54_residual_coordinate_B_zero D S).
  - exact (s54_residual_coordinate_C_zero D HD S).
  - exact (s54_residual_coordinate_E_zero D HD S).
Qed.

Theorem s54_octonion_sedenion_backward_from_residual_coordinates :
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    forall S : Schafer1954ResidualCoordinateSurface D,
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A.
Proof.
  intros D HD S.
  exists (s54_block_A D).
  split.
  - exact (s54_block_A_is_octonion_derivation D HD).
  - exact (s54_concrete_backward_from_residual_coordinates D HD S).
Qed.

Theorem s54_octonion_sedenion_backward_from_residual_coordinate_builder :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ResidualCoordinateSurface D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hcoords D HD.
  exact
    (s54_octonion_sedenion_backward_from_residual_coordinates
       D HD (Hcoords D HD)).
Qed.

Theorem s54_octonion_sedenion_backward_from_theorem3_instantiation :
  forall s54_eq30_formula : (CDOct -> CDOct) -> CDOct -> Prop,
  forall s54_eq31_formula : (CDOct -> CDOct) -> CDOct -> Prop,
  forall s54_eq32_formula : (CDOct -> CDOct) -> CDOct -> Prop,
  forall s54_eq37_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq38_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq39_formula : (CDOct -> CDOct) -> CDOct -> Prop,
  forall s54_eq40_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq42_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq46_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq47_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq48_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq49_formula : (CDOct -> CDOct) -> Prop,
  forall s54_eq50_formula : (Schafer1954Basis7 -> R) -> (CDOct -> CDOct) -> Prop,
  forall s54_eq52_formula :
    (Schafer1954Basis7 -> Schafer1954Basis7 -> Schafer1954Basis7 -> Prop) -> Prop,
  (forall mu B,
      s54_eq50_formula mu B ->
      forall i : Schafer1954Basis7,
        B (s54_basis7_oct i) = oct_scale (mu i) (s54_basis7_oct i)) ->
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteTheorem3InstantiationSurface
        s54_eq30_formula s54_eq31_formula s54_eq32_formula
        s54_eq37_formula s54_eq38_formula s54_eq39_formula
        s54_eq40_formula s54_eq42_formula s54_eq46_formula
        s54_eq47_formula s54_eq48_formula s54_eq49_formula
        s54_eq50_formula s54_eq52_formula
        D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros s54_eq30_formula s54_eq31_formula s54_eq32_formula
         s54_eq37_formula s54_eq38_formula s54_eq39_formula
         s54_eq40_formula s54_eq42_formula s54_eq46_formula
         s54_eq47_formula s54_eq48_formula s54_eq49_formula
         s54_eq50_formula s54_eq52_formula H50 Hcoord D HD.
  exact
    (s54_octonion_sedenion_backward_from_residual_coordinate_builder
       (fun D' HD' =>
          s54_residual_coordinate_surface_of_theorem3_instantiation
            s54_eq30_formula s54_eq31_formula s54_eq32_formula
            s54_eq37_formula s54_eq38_formula s54_eq39_formula
            s54_eq40_formula s54_eq42_formula s54_eq46_formula
            s54_eq47_formula s54_eq48_formula s54_eq49_formula
            s54_eq50_formula s54_eq52_formula H50
       D' (Hcoord D' HD'))
       D HD).
Qed.

Theorem s54_octonion_sedenion_backward_from_residual_theorem3_facts :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteResidualTheorem3Facts D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hfacts D HD.
  exact
    (s54_octonion_sedenion_backward_from_theorem3_instantiation
       s54_concrete_eq30_formula s54_concrete_eq31_formula s54_concrete_eq32_formula
       s54_concrete_eq37_formula s54_concrete_eq38_formula s54_concrete_eq39_formula
       s54_concrete_eq40_formula s54_concrete_eq42_formula s54_concrete_eq46_formula
       s54_concrete_eq47_formula s54_concrete_eq48_formula s54_concrete_eq49_formula
       s54_concrete_eq50_formula s54_concrete_eq52_formula
       s54_concrete_eq50_formula_concrete
       (fun D' HD' =>
          s54_concrete_theorem3_instantiation_of_residual_facts
            D' HD' (Hfacts D' HD'))
       D HD).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_identification_from_residual_theorem3_facts :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteResidualTheorem3Facts D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D <->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hfacts D.
  split.
  - exact (s54_octonion_sedenion_backward_from_residual_theorem3_facts Hfacts D).
  - intros [A [HA ->]].
    exact (schafer1954_theorem2_octonion_to_sedenion_extension_map A HA).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_type_g_from_residual_theorem3_facts :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteResidualTheorem3Facts D) ->
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A) /\
  dim_g2 = 14%nat /\
  dim_stabilizer = 8%nat /\
  (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
Proof.
  intro Hfacts.
  split.
  - exact
      (schafer1954_theorem4_octonion_sedenion_identification_from_residual_theorem3_facts
         Hfacts).
  - exact schafer1954_theorem4_type_g_support.
Qed.

Theorem s54_octonion_sedenion_backward_from_residual_diagonal_facts :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteResidualDiagonalFacts D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hfacts D HD.
  exact
    (s54_octonion_sedenion_backward_from_residual_theorem3_facts
       (fun D' HD' =>
          s54_concrete_residual_theorem3_facts_of_diagonal_facts
            D' HD' (Hfacts D' HD'))
       D HD).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_identification_from_residual_diagonal_facts :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteResidualDiagonalFacts D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D <->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hfacts D.
  split.
  - exact (s54_octonion_sedenion_backward_from_residual_diagonal_facts Hfacts D).
  - intros [A [HA ->]].
    exact (schafer1954_theorem2_octonion_to_sedenion_extension_map A HA).
Qed.

Theorem s54_octonion_sedenion_backward_from_residual_eq50_builder :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      { mu : Schafer1954Basis7 -> R &
        forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i) }) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D ->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Heq50 D HD.
  apply (s54_octonion_sedenion_backward_from_residual_diagonal_facts
           (fun D' HD' =>
              let (mu, Hdiag) := Heq50 D' HD' in
              s54_concrete_residual_diagonal_facts_of_eq50 D' HD' mu Hdiag)
           D HD).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_identification_from_residual_eq50_builder :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      { mu : Schafer1954Basis7 -> R &
        forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i) }) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D <->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Heq50 D.
  split.
  - exact (s54_octonion_sedenion_backward_from_residual_eq50_builder Heq50 D).
  - intros [A [HA ->]].
    exact (schafer1954_theorem2_octonion_to_sedenion_extension_map A HA).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_type_g_from_residual_eq50_builder :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      { mu : Schafer1954Basis7 -> R &
        forall i : Schafer1954Basis7,
          s54_block_C (s54_sed_residual D) (s54_basis7_oct i) =
          oct_scale (mu i) (s54_basis7_oct i) }) ->
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A) /\
  dim_g2 = 14%nat /\
  dim_stabilizer = 8%nat /\
  (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
Proof.
  intro Heq50.
  split.
  - exact
      (schafer1954_theorem4_octonion_sedenion_identification_from_residual_eq50_builder
         Heq50).
  - exact schafer1954_theorem4_type_g_support.
Qed.

Record Schafer1954OctonionSedenionConverseSurface := {
  s54_octsed_converse_eq52_discharge :
    forall p : Schafer1954Basis7 -> R,
      (forall i j k : Schafer1954Basis7,
          schafer1954_eq52_cayley_triple i j k ->
          (p i + p j + p k)%R = 0%R) ->
      forall i : Schafer1954Basis7, p i = 0%R;
  s54_octsed_converse_backward :
    forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A;
}.

Definition schafer1954_octonion_sedenion_converse_surface_of_backward
    (Hbackward :
      forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D ->
        exists A : CDOct -> CDOct,
          schafer1954_octonion_derivation A /\
          D = schafer1954_sedenion_extend A) :
    Schafer1954OctonionSedenionConverseSurface.
Proof.
  refine
    {| s54_octsed_converse_eq52_discharge :=
         schafer1954_eq52_real_solution_zero;
       s54_octsed_converse_backward := Hbackward |}.
Defined.

Definition schafer1954_octonion_sedenion_converse_surface_of_residual_coordinates
    (Hcoords :
      forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D ->
        Schafer1954ResidualCoordinateSurface D) :
    Schafer1954OctonionSedenionConverseSurface :=
  schafer1954_octonion_sedenion_converse_surface_of_backward
    (s54_octonion_sedenion_backward_from_residual_coordinate_builder Hcoords).

Section Schafer1954ConcreteTheorem4FromTheorem3.
  Variable s54_eq30_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq31_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq32_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq37_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq38_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq39_formula : (CDOct -> CDOct) -> CDOct -> Prop.
  Variable s54_eq40_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq42_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq46_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq47_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq48_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq49_formula : (CDOct -> CDOct) -> Prop.
  Variable s54_eq50_formula : (Schafer1954Basis7 -> R) -> (CDOct -> CDOct) -> Prop.
  Variable s54_eq52_formula :
    (Schafer1954Basis7 -> Schafer1954Basis7 -> Schafer1954Basis7 -> Prop) -> Prop.

  Hypothesis s54_eq50_formula_concrete :
    forall mu B,
      s54_eq50_formula mu B ->
      forall i : Schafer1954Basis7,
        B (s54_basis7_oct i) = oct_scale (mu i) (s54_basis7_oct i).

  Hypothesis s54_theorem3_instantiation_builder :
    forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ConcreteTheorem3InstantiationSurface
        s54_eq30_formula s54_eq31_formula s54_eq32_formula
        s54_eq37_formula s54_eq38_formula s54_eq39_formula
        s54_eq40_formula s54_eq42_formula s54_eq46_formula
        s54_eq47_formula s54_eq48_formula s54_eq49_formula
        s54_eq50_formula s54_eq52_formula
        D.

  Definition schafer1954_octonion_sedenion_converse_surface_of_theorem3_instantiation :
      Schafer1954OctonionSedenionConverseSurface :=
    schafer1954_octonion_sedenion_converse_surface_of_backward
      (s54_octonion_sedenion_backward_from_theorem3_instantiation
         s54_eq30_formula s54_eq31_formula s54_eq32_formula
         s54_eq37_formula s54_eq38_formula s54_eq39_formula
         s54_eq40_formula s54_eq42_formula s54_eq46_formula
         s54_eq47_formula s54_eq48_formula s54_eq49_formula
         s54_eq50_formula s54_eq52_formula
         s54_eq50_formula_concrete s54_theorem3_instantiation_builder).

  Theorem schafer1954_theorem4_octonion_sedenion_identification_from_theorem3_instantiation :
    forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A.
  Proof.
    intro D.
    split.
    - exact
        (s54_octsed_converse_backward
           schafer1954_octonion_sedenion_converse_surface_of_theorem3_instantiation
           D).
    - intros [A [HA ->]].
      exact (schafer1954_theorem2_octonion_to_sedenion_extension_map A HA).
  Qed.

  Theorem schafer1954_theorem4_octonion_sedenion_type_g_from_theorem3_instantiation :
    (forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D <->
        exists A : CDOct -> CDOct,
          schafer1954_octonion_derivation A /\
          D = schafer1954_sedenion_extend A) /\
    dim_g2 = 14%nat /\
    dim_stabilizer = 8%nat /\
    (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
  Proof.
    split.
    - exact schafer1954_theorem4_octonion_sedenion_identification_from_theorem3_instantiation.
    - exact schafer1954_theorem4_type_g_support.
  Qed.

  Record Schafer1954Theorem4FromTheorem3Surface := {
    s54_t4_from_t3_converse_surface_data :
      Schafer1954OctonionSedenionConverseSurface;
    s54_t4_from_t3_identification :
      forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D <->
        exists A : CDOct -> CDOct,
          schafer1954_octonion_derivation A /\
          D = schafer1954_sedenion_extend A;
    s54_t4_from_t3_type_g :
      (forall D : CDSed -> CDSed,
          Schafer1954IsSedenionDerivation D <->
          exists A : CDOct -> CDOct,
            schafer1954_octonion_derivation A /\
            D = schafer1954_sedenion_extend A) /\
      dim_g2 = 14%nat /\
      dim_stabilizer = 8%nat /\
      (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168);
  }.

  Definition schafer1954_theorem4_from_theorem3_surface :
      Schafer1954Theorem4FromTheorem3Surface.
  Proof.
    refine
      {| s54_t4_from_t3_converse_surface_data :=
           schafer1954_octonion_sedenion_converse_surface_of_theorem3_instantiation;
         s54_t4_from_t3_identification := _;
         s54_t4_from_t3_type_g := _ |}.
    - exact schafer1954_theorem4_octonion_sedenion_identification_from_theorem3_instantiation.
    - exact schafer1954_theorem4_octonion_sedenion_type_g_from_theorem3_instantiation.
  Defined.
End Schafer1954ConcreteTheorem4FromTheorem3.

Theorem schafer1954_theorem4_octonion_sedenion_identification :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D <->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intros Hbackward D.
  split.
  - exact (Hbackward D).
  - intros [A [HA ->]].
    exact (schafer1954_theorem2_octonion_to_sedenion_extension_map A HA).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_identification_from_converse_surface :
  forall S : Schafer1954OctonionSedenionConverseSurface,
    forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A.
Proof.
  intros S.
  apply schafer1954_theorem4_octonion_sedenion_identification.
  exact (s54_octsed_converse_backward S).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_identification_from_residual_coordinates :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      Schafer1954ResidualCoordinateSurface D) ->
  forall D : CDSed -> CDSed,
    Schafer1954IsSedenionDerivation D <->
    exists A : CDOct -> CDOct,
      schafer1954_octonion_derivation A /\
      D = schafer1954_sedenion_extend A.
Proof.
  intro Hcoords.
  apply schafer1954_theorem4_octonion_sedenion_identification.
  exact
    (s54_octonion_sedenion_backward_from_residual_coordinate_builder Hcoords).
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_type_g :
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D ->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A) ->
  (forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A) /\
  dim_g2 = 14%nat /\
  dim_stabilizer = 8%nat /\
  (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
Proof.
  intro Hbackward.
  split.
  - exact (schafer1954_theorem4_octonion_sedenion_identification Hbackward).
  - exact schafer1954_theorem4_type_g_support.
Qed.

Theorem schafer1954_theorem4_octonion_sedenion_type_g_from_converse_surface :
  forall S : Schafer1954OctonionSedenionConverseSurface,
    (forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D <->
        exists A : CDOct -> CDOct,
          schafer1954_octonion_derivation A /\
          D = schafer1954_sedenion_extend A) /\
    dim_g2 = 14%nat /\
    dim_stabilizer = 8%nat /\
    (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168).
Proof.
  intros S.
  split.
  - exact
      (schafer1954_theorem4_octonion_sedenion_identification_from_converse_surface S).
  - exact schafer1954_theorem4_type_g_support.
Qed.

Record Schafer1954Theorem4OctonionSedenionSurface := {
  s54_t4_octsed_converse_surface_data :
    Schafer1954OctonionSedenionConverseSurface;
  s54_t4_octsed_identification :
    forall D : CDSed -> CDSed,
      Schafer1954IsSedenionDerivation D <->
      exists A : CDOct -> CDOct,
        schafer1954_octonion_derivation A /\
        D = schafer1954_sedenion_extend A;
  s54_t4_octsed_type_g :
    (forall D : CDSed -> CDSed,
        Schafer1954IsSedenionDerivation D <->
        exists A : CDOct -> CDOct,
          schafer1954_octonion_derivation A /\
          D = schafer1954_sedenion_extend A) /\
    dim_g2 = 14%nat /\
    dim_stabilizer = 8%nat /\
    (21 - 7 = 14 /\ 14 - 6 = 8 /\ (2^3 - 1) * (2^3 - 2) * (2^3 - 4) = 168);
}.

Definition schafer1954_theorem4_octonion_sedenion_surface
    (S : Schafer1954OctonionSedenionConverseSurface) :
    Schafer1954Theorem4OctonionSedenionSurface.
Proof.
  refine
    {| s54_t4_octsed_converse_surface_data := S;
       s54_t4_octsed_identification := _;
       s54_t4_octsed_type_g := _ |}.
  - exact
      (schafer1954_theorem4_octonion_sedenion_identification_from_converse_surface S).
  - exact
      (schafer1954_theorem4_octonion_sedenion_type_g_from_converse_surface S).
Defined.

(** Paper-order scope checkpoint:
    - Theorem 2 is now landed abstractly as the generic extension map above.
    - Theorem 3 now has both the block-restriction bridge `(29)`-`(35)` and a
      coordinate normalization surface for `(36)`-`(52)`.
    - Theorem 4 now has both an abstract identification surface expressing the
      `D(M_t) = D(C)` equivalence and the concrete real-octonion `(51)`-`(52)`
      discharge, plus a canonical octonion/sedenion converse surface, a
      residual-coordinate backward bridge, a concrete theorem-4-from-
      theorem-3 package, and a smaller residual-theorem-3-facts package; the
      remaining gap is to prove those concrete residual theorem-3 facts from
      the full paper-specific uniqueness argument. *)
Theorem schafer1954_theorems2_to_4_scope_summary :
  True.
Proof. exact I. Qed.

Theorem Schafer1954_lane_compiles : True.
Proof. exact I. Qed.
