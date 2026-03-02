(** * C-905: Jordan identity at dim=4.

    (a^2*b)*a = a^2*(b*a) for all a, b in H.
    Holds in all alternative algebras (including octonions). *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra CDAssociator.

Theorem C905_quat_jordan_identity : forall a b,
  quat_mul (quat_mul (quat_mul a a) b) a =
  quat_mul (quat_mul a a) (quat_mul b a).
Proof. exact quat_jordan. Qed.
