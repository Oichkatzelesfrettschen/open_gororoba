(** * OctonionInverse: two-sided inverse at dim 8.

    Octonion inverse oct_inv(x) = conj(x) / |x|^2, the dim-8 analogue of the
    quaternion CDInverse.v.  The two-sided inverse property (x * inv(x) = 1 and
    inv(x) * x = 1 for |x|^2 <> 0) is already established in Brown1972ChapterV.v
    (brown1972_oct_inv_mul_left/right); this module restates it under the canonical
    names oct_inv / oct_mul_inv_r / oct_mul_inv_l so the ODIV/ODIV_L catalog ground
    matches the quaternion CDInverse naming.  Valid for every nonzero octonion: dim
    8 is a division algebra (no zero divisors), so |x|^2 > 0 unless x = 0. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
     OctonionNorm Brown1972ChapterIII Brown1972ChapterV.

Open Scope R_scope.

(** Octonion inverse: conj(x) scaled by the reciprocal squared norm. *)
Definition oct_inv (x : CDOct) : CDOct :=
  oct_scale (/ oct_norm_sq x) (oct_conj x).

(** Octonion one (the multiplicative identity (1,0,0,0,0,0,0,0)). *)
Definition oct_one : CDOct := brown1972_oct_one.

(** Right inverse: x * inv(x) = 1. *)
Theorem oct_mul_inv_r : forall x : CDOct,
  oct_norm_sq x <> 0 -> oct_mul x (oct_inv x) = oct_one.
Proof.
  intros x Hnz.
  unfold oct_inv, oct_one.
  exact (brown1972_oct_inv_mul_right x Hnz).
Qed.

(** Left inverse: inv(x) * x = 1. *)
Theorem oct_mul_inv_l : forall x : CDOct,
  oct_norm_sq x <> 0 -> oct_mul (oct_inv x) x = oct_one.
Proof.
  intros x Hnz.
  unfold oct_inv, oct_one.
  exact (brown1972_oct_inv_mul_left x Hnz).
Qed.
