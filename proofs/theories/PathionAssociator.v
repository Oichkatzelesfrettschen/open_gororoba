(** * PathionAssociator: Associator computation at dim=32.

    The pathion associator [a,b,c] := (a*b)*c - a*(b*c) measures
    departure from associativity at dim=32.

    Key results:
    - pathion_assoc is well-defined via pathion_sub of two triple products
    - pathion_assoc_norm_sq computes |[a,b,c]|^2
    - Constructive witness: non-zero associator for (e1, e2, e4)
    - All 32 basis elements have unit norm

    Mirrors: crates/algebra_experimental/src/higher_cd.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
From OpenGororoba Require Import Pathion.

(** * Pathion associator: [a,b,c] = (a*b)*c - a*(b*c). *)

Definition pathion_assoc (a b c : CDPathion) : CDPathion :=
  pathion_sub (pathion_mul (pathion_mul a b) c) (pathion_mul a (pathion_mul b c)).

(** * Pathion associator norm squared. *)

Definition pathion_assoc_norm_sq (a b c : CDPathion) : R :=
  pathion_norm_sq (pathion_assoc a b c).

(** * Full cbv whitelist for pathion-level computation.

    Includes all projection and arithmetic functions from pathion
    down through sedenion, octonion, and quaternion layers.
    Using ring_simplify + lra instead of ring alone gives ~30x speedup
    at dim=32 (avoids exponential polynomial expansion). *)

Ltac pathion_norm_sq_4 :=
  unfold pathion_assoc_norm_sq, pathion_assoc, pathion_sub, pathion_add, pathion_neg;
  cbv [pathion_norm_sq pathion_mul pathion_e pathion_zero pathion_lo pathion_hi
       sed_norm_sq sed_mul sed_add sed_sub sed_neg sed_conj sed_e sed_zero
       oct_norm_sq oct_mul oct_add oct_sub oct_neg oct_conj oct_e oct_zero
       quat_norm_sq quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  ring_simplify; lra.

(** * Pathion basis element norms. *)

(** Every standard basis element e_i (i < 32) has unit norm. *)
Theorem pathion_basis_unit_norm : forall i : nat,
  (i < 32)%nat -> pathion_norm_sq (pathion_e i) = 1.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|[|[|[|[|[|[|[|[|[|[|[|[|
    [|[|[|[|[|[|[|[|[|[|[|[|[|[|[|[|n]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]];
    try (cbv [pathion_e pathion_norm_sq pathion_zero
              sed_norm_sq sed_e sed_zero oct_norm_sq oct_e oct_zero
              quat_norm_sq quat_zero quat_one
              pathion_lo pathion_hi sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
         ring);
    lia.
Qed.

(** * Constructive witness: the pathion associator is non-zero for (e1, e2, e4).
    This proves non-associativity at dim=32 via an explicit triple that
    already witnesses it at dim=16 (sedenion subspace). *)

Theorem pathion_assoc_nonzero_e1_e2_e4 :
  pathion_assoc (pathion_e 1) (pathion_e 2) (pathion_e 4) <> pathion_zero.
Proof.
  intro H.
  apply (f_equal (fun p => qd (oct_hi (sed_lo (pathion_lo p))))) in H.
  cbv [pathion_assoc pathion_sub pathion_add pathion_neg pathion_mul
       pathion_e pathion_zero pathion_lo pathion_hi pathion_conj
       sed_mul sed_add sed_sub sed_neg sed_conj sed_e sed_zero sed_norm_sq
       oct_mul oct_add oct_sub oct_neg oct_conj oct_e oct_zero oct_norm_sq
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one quat_norm_sq
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd] in H.
  ring_simplify in H. lra.
Qed.
