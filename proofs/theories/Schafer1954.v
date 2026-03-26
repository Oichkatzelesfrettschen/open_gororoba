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
    - Theorem 4, pp. 445-446:
      derivation algebras `D(M_t) = D(C)` with 14-dimensional type-G surface.
      This remains open as a dedicated paper theorem lane; local G2 files are
      support material, not yet a faithful formalization of Schafer's theorem.

    Current Rocq companion map:
    - CDPowerAssociative.v : flexibility / power-associativity anchors
    - CDPropertyTower.v    : low-dimensional property-loss summary

    Remaining Schafer 1954 backlog:
    - dedicated basis-law lane matching Lemma 4 and the restricted basis claims
    - explicit derivation-algebra formalization for Theorem 4
    - a tighter bridge from the G2 support files to Schafer's derivation language
*)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Export
  CDPowerAssociative
  CDPropertyTower.

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

Theorem Schafer1954_lane_compiles : True.
Proof. exact I. Qed.
