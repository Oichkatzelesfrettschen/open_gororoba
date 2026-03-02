(** * C-902: Pure imaginary quaternion squares to negative norm.

    For q with Re(q)=0: q^2 = -|q|^2.
    Generalizes i^2 = j^2 = k^2 = -1 to arbitrary pure quaternions. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C902_quat_imaginary_square : forall q,
  qa q = 0 ->
  quat_mul q q = quat_scale (- quat_norm_sq q) quat_one.
Proof. exact quat_imaginary_square. Qed.
