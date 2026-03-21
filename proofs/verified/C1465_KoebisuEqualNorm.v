(** * C-1465: Koebisu Equal-Norm Condition for Standard Zero Divisors.

    For standard 2-blade ZDs v = e_low +/- e_high with
    low in 1..7 and high in 9..15, the octonion-half norms are equal:
    ||v_1||^2 = ||v_2||^2.

    Strategy: express the norm difference as a single R expression,
    reduce with cbv, then close with ring_simplify + lra.

    Mirrors: test_koebisu_d2_polynomial in reggiani.rs *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** Norm difference: oct_norm_sq(lo) - oct_norm_sq(hi).
    For a ZD this should be zero. *)

Definition sed_norm_diff (v : CDSed) : R :=
  oct_norm_sq (sed_lo v) - oct_norm_sq (sed_hi v).

(** e_1 + e_9: norm difference = 0. *)
Theorem koebisu_e1_e9 : sed_norm_diff (sed_add (sed_e 1%nat) (sed_e 9%nat)) = 0.
Proof.
  unfold sed_norm_diff.
  cbv [sed_add sed_e sed_zero oct_add oct_e oct_zero oct_norm_sq
       quat_norm_sq quat_add quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify.
  reflexivity.
Qed.

(** e_3 + e_14. *)
Theorem koebisu_e3_e14 : sed_norm_diff (sed_add (sed_e 3%nat) (sed_e 14%nat)) = 0.
Proof.
  unfold sed_norm_diff.
  cbv [sed_add sed_e sed_zero oct_add oct_e oct_zero oct_norm_sq
       quat_norm_sq quat_add quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify.
  reflexivity.
Qed.

(** e_7 + e_10. *)
Theorem koebisu_e7_e10 : sed_norm_diff (sed_add (sed_e 7%nat) (sed_e 10%nat)) = 0.
Proof.
  unfold sed_norm_diff.
  cbv [sed_add sed_e sed_zero oct_add oct_e oct_zero oct_norm_sq
       quat_norm_sq quat_add quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify.
  reflexivity.
Qed.

(** e_5 - e_12 (negative sign variant). *)
Theorem koebisu_e5_minus_e12 : sed_norm_diff (sed_sub (sed_e 5%nat) (sed_e 12%nat)) = 0.
Proof.
  unfold sed_norm_diff.
  cbv [sed_sub sed_add sed_neg sed_e sed_zero oct_add oct_neg oct_e oct_zero oct_norm_sq
       quat_norm_sq quat_add quat_neg quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  ring_simplify.
  reflexivity.
Qed.
