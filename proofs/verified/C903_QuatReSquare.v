(** * C-903: Real part of quaternion square.

    Re(q^2) = 2*Re(q)^2 - |q|^2. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C903_quat_re_square : forall q,
  qa (quat_mul q q) = 2 * (qa q)^2 - quat_norm_sq q.
Proof. exact quat_re_square. Qed.
