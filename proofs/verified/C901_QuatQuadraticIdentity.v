(** * C-901: Quaternion quadratic identity.

    q^2 = 2*Re(q)*q - |q|^2.
    Every quaternion satisfies a monic quadratic over R.
    This is the key structural identity for quaternion algebras. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C901_quat_quadratic_identity : forall q,
  quat_mul q q = quat_add (quat_scale (2 * qa q) q)
                           (quat_scale (- quat_norm_sq q) quat_one).
Proof. exact quat_quadratic_identity. Qed.
