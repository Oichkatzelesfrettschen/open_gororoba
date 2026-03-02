(** * C-900: Quaternion real-part commutativity.

    Re(p*q) = Re(q*p) for all p, q in H.
    Scalar-part commutativity despite full non-commutativity. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

Theorem C900_quat_re_comm : forall p q,
  qa (quat_mul p q) = qa (quat_mul q p).
Proof. exact quat_re_mul_comm. Qed.
