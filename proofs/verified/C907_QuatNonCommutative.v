(** * C-907: Quaternions are NOT commutative.

    Explicit witness: i*j = k but j*i = -k.
    This is the first Cayley-Dickson level where commutativity fails. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Definition qi : CDQuat := mkQuat 0 1 0 0.
Definition qj : CDQuat := mkQuat 0 0 1 0.
Definition qk : CDQuat := mkQuat 0 0 0 1.

(** i*j = k. *)
Theorem ij_eq_k : quat_mul qi qj = qk.
Proof. unfold qi, qj, qk, quat_mul; simpl; f_equal; ring. Qed.

(** j*i = -k. *)
Theorem ji_eq_neg_k : quat_mul qj qi = mkQuat 0 0 0 (-1).
Proof. unfold qi, qj, quat_mul; simpl; f_equal; ring. Qed.

(** Quaternions are not commutative. *)
Theorem C907_quat_noncommutative : quat_mul qi qj <> quat_mul qj qi.
Proof.
  unfold qi, qj, quat_mul; simpl.
  intro H. injection H. lra.
Qed.
