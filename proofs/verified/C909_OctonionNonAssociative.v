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
  intro H.
  (* Project directly to the distinguishing scalar to avoid unfolding the
     full 8-component octonion equality.  The two sides differ only in the
     qd component of oct_hi: +1 versus -1. *)
  assert (Hd := f_equal (fun x => qd (oct_hi x)) H).
  unfold oe1, oe2, oe4, oct_e in Hd.
  cbv [oct_mul oct_conj oct_hi qd
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd oct_lo] in Hd.
  lra.
Qed.
