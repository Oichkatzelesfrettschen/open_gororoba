(** * C-909: Octonions are NOT associative.

    Explicit witness: (e1 * e2) * e4 <> e1 * (e2 * e4).
    Computed: (e1*e2)*e4 = e7, e1*(e2*e4) = -e7.
    These differ in the qd component of oct_hi: 1 vs -1. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Definition oe1 : CDOct := oct_e 1.  (* (0,1,0,0 | 0,0,0,0) *)
Definition oe2 : CDOct := oct_e 2.  (* (0,0,1,0 | 0,0,0,0) *)
Definition oe4 : CDOct := oct_e 4.  (* (0,0,0,0 | 1,0,0,0) *)

(** (e1 * e2) * e4 <> e1 * (e2 * e4). *)
Theorem C909_octonion_non_associative :
  oct_mul (oct_mul oe1 oe2) oe4 <> oct_mul oe1 (oct_mul oe2 oe4).
Proof.
  unfold oe1, oe2, oe4, oct_e, oct_mul, oct_conj,
         quat_mul, quat_add, quat_neg, quat_conj, quat_zero, quat_one.
  simpl. intro H.
  assert (Hhi := f_equal oct_hi H). simpl in Hhi.
  assert (Hd := f_equal qd Hhi). simpl in Hd.
  lra.
Qed.
