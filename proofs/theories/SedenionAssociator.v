(** * SedenionAssociator: Associator computation at dim=16.

    The sedenion associator [a,b,c] := (a*b)*c - a*(b*c) measures
    departure from associativity.

    Key results:
    - sed_assoc is well-defined via sed_sub of two triple products
    - The associator is NON-ZERO for specific triples (constructive witness)
    - Sedenion basis elements all have unit norm
    - sed_assoc_norm_sq computes |[a,b,c]|^2

    Mirrors: crates/algebra_experimental/src/majorana_braiding.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** * Sedenion associator: [a,b,c] = (a*b)*c - a*(b*c). *)

Definition sed_assoc (a b c : CDSed) : CDSed :=
  sed_sub (sed_mul (sed_mul a b) c) (sed_mul a (sed_mul b c)).

(** * Sedenion associator norm squared. *)

Definition sed_assoc_norm_sq (a b c : CDSed) : R :=
  sed_norm_sq (sed_assoc a b c).

(** * Bulletproof tactic for sedenion component extraction.

    Bridges the axiomatic R gap: cbv unfolds the CD structure,
    ring_simplify crushes symbolic R arithmetic (0*1 + (-1)*0 -> -1),
    lra closes the numerical contradiction. *)

Ltac extract_and_smash proj H :=
  apply (f_equal proj) in H;
  cbv [sed_assoc sed_mul sed_add sed_sub sed_neg
       oct_mul oct_add oct_sub oct_neg oct_conj
       quat_mul quat_add quat_neg quat_conj
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd
       sed_e oct_e sed_zero oct_zero quat_zero quat_one] in H;
  ring_simplify in H;
  try lra;
  try discriminate.

(** * Sedenion basis element norms. *)

(** Every standard basis element e_i (i < 16) has unit norm. *)
Theorem sed_basis_unit_norm : forall i : nat,
  (i < 16)%nat -> sed_norm_sq (sed_e i) = 1.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|[|[|[|[|[|[|[|[|[|[|[|[|n]]]]]]]]]]]]]]]];
    try (cbv [sed_e oct_e sed_norm_sq oct_norm_sq quat_norm_sq
              sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero quat_one
              qa qb qc qd]; ring);
    lia.
Qed.

(** * Constructive witness: the associator is non-zero for (e1, e2, e4).
    This proves non-associativity at dim=16 via an explicit triple. *)

Theorem sed_assoc_nonzero_e1_e2_e4 :
  sed_assoc (sed_e 1) (sed_e 2) (sed_e 4) <> sed_zero.
Proof.
  intro H.
  extract_and_smash (fun s => qd (oct_hi (sed_lo s))) H.
Qed.

(** * Constructive witness: associator (e1, e9, e2) is non-zero.
    This witnesses a ZD-channel-crossing associator. *)

Theorem sed_assoc_nonzero_e1_e9_e2 :
  sed_assoc (sed_e 1) (sed_e 9) (sed_e 2) <> sed_zero.
Proof.
  intro H.
  extract_and_smash (fun s => qc (oct_lo (sed_hi s))) H.
Qed.
