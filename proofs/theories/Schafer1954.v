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

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
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
        (s54_t3_diag_scalar i + s54_t3_diag_scalar j)%R =
        s54_t3_diag_scalar k;
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

(** Paper-order scope checkpoint:
    - Theorem 2 is now landed abstractly as the generic extension map above.
    - Theorem 3 now has both the block-restriction bridge `(29)`-`(35)` and a
      coordinate normalization surface for `(36)`-`(52)`.
    - Theorem 4 now has an abstract identification surface expressing the
      `D(M_t) = D(C)` equivalence, while the concrete coordinate uniqueness
      discharge and the final paper-specific type-G theorem still remain open. *)
Theorem schafer1954_theorems2_to_4_scope_summary :
  True.
Proof. exact I. Qed.

Theorem Schafer1954_lane_compiles : True.
Proof. exact I. Qed.
