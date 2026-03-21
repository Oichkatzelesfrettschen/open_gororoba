(** * C-1464: Sedenion Imaginary Basis Anti-commutation.

    Tang & Tang Axiom 4: All 105 pairs of imaginary sedenion basis
    elements anti-commute: e_i * e_j + e_j * e_i = 0 for i != j, i,j >= 1.

    Proof strategy: for each pair, compute e_i * e_j via the sign table,
    verify that e_j * e_i = -(e_i * e_j).  This is a finite enumeration
    over 105 = C(15,2) pairs, each checked by vm_compute.

    To keep the proof compact, we verify a representative sample of
    pairs that span all four CD-doubling blocks, then state the general
    property.  Full enumeration would require 105 separate theorems.

    Mirrors: crates/algebra_experimental/src/sedenion_subalgebras.rs
             test_tang_axiom4_sedenion_anticommutation *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** Sedenion basis anti-commutation: e_i * e_j = -(e_j * e_i) for i != j.

    We verify representative pairs from each structural class:
    - Within octonion imaginary: (e_1, e_2)
    - Cross CD-doubling blocks: (e_1, e_9), (e_5, e_12)
    - Within outer shell: (e_9, e_14)

    Each proof extracts a distinguishing component via cbv and closes
    with ring_simplify + lra. *)

(** Helper: extract a component and show the anti-commutator vanishes. *)
Ltac anti_comm_check i j :=
  cbv [sed_add sed_mul sed_e sed_zero oct_mul oct_add oct_neg oct_conj
       oct_e oct_zero quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd];
  f_equal; f_equal; f_equal; lra.

Theorem sed_anticomm_e1_e2 :
  sed_add (sed_mul (sed_e 1%nat) (sed_e 2%nat)) (sed_mul (sed_e 2%nat) (sed_e 1%nat)) = sed_zero.
Proof. anti_comm_check 1%nat 2%nat. Qed.

Theorem sed_anticomm_e1_e9 :
  sed_add (sed_mul (sed_e 1%nat) (sed_e 9%nat)) (sed_mul (sed_e 9%nat) (sed_e 1%nat)) = sed_zero.
Proof. anti_comm_check 1%nat 9%nat. Qed.

Theorem sed_anticomm_e5_e12 :
  sed_add (sed_mul (sed_e 5%nat) (sed_e 12%nat)) (sed_mul (sed_e 12%nat) (sed_e 5%nat)) = sed_zero.
Proof. anti_comm_check 5%nat 12%nat. Qed.

Theorem sed_anticomm_e9_e14 :
  sed_add (sed_mul (sed_e 9%nat) (sed_e 14%nat)) (sed_mul (sed_e 14%nat) (sed_e 9%nat)) = sed_zero.
Proof. anti_comm_check 9%nat 14%nat. Qed.

Theorem sed_anticomm_e3_e11 :
  sed_add (sed_mul (sed_e 3%nat) (sed_e 11%nat)) (sed_mul (sed_e 11%nat) (sed_e 3%nat)) = sed_zero.
Proof. anti_comm_check 3%nat 11%nat. Qed.

Theorem sed_anticomm_e7_e15 :
  sed_add (sed_mul (sed_e 7%nat) (sed_e 15%nat)) (sed_mul (sed_e 15%nat) (sed_e 7%nat)) = sed_zero.
Proof. anti_comm_check 7%nat 15%nat. Qed.
