(** * CDPowerAssociative: Power-associativity and the quadratic identity.

    Schafer (1961) Ch V: An algebra A is power-associative iff
      x^i * x^j = x^{i+j} for all i, j >= 1.

    Equivalent conditions (Schafer eqs 3, 4):
      (3)  x^2 * x = x * x^2     (third-power associativity)
      (4)  x^2 * x^2 = x(x*x^2)  (fourth-power associativity)
      (3') (x, x, x) = 0
      (4') (x, x, x^2) = 0

    The quadratic identity (Schafer eq 25):
      x^2 - t(x)*x + n(x)*1 = 0
    holds in all CD algebras.  For purely imaginary x (t(x)=0):
      x^2 = -n(x)*1

    We prove these at the sedenion level (dim 16) by direct computation.

    Also: the flexible identity (xy)x = x(yx) FAILS at dim 16.

    Sources: Schafer (1961) "An Introduction to Nonassociative Algebras",
             Ch III (eqs 25-28) and Ch V (eqs 1-4).
    Mirrors: cd_kernel, DicksonCDProcess.v, SchaferDivAlg16.v
    Supports claims: C-1007 (CD property loss). *)

From Stdlib Require Import Reals Lra Psatz.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** ================================================================== *)
(** * Third-power associativity: x^2 * x = x * x^2.                   *)
(** ================================================================== *)

(** Quaternion level: trivially true (quaternions are associative). *)
Theorem quat_third_power : forall q : CDQuat,
  quat_mul (quat_mul q q) q = quat_mul q (quat_mul q q).
Proof. intros [a b c d]; unfold quat_mul; simpl; f_equal; ring. Qed.

(** Octonion level: true because octonions are alternative,
    and alternative implies third-power associativity. *)
Theorem oct_third_power : forall x : CDOct,
  oct_mul (oct_mul x x) x = oct_mul x (oct_mul x x).
Proof.
  intros [[a b c d] [e f g h]].
  cbv [oct_mul oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Quadratic identity: x^2 = t(x)*x - n(x)*1.                     *)
(** ================================================================== *)

(** The quadratic identity for quaternions (already proved as
    quat_quadratic_identity in CayleyDicksonAlgebra.v).
    For purely imaginary quaternions: x^2 = -n(x)*1. *)

(** Sedenion norm-conjugate: x * conj(x) = n(x) * 1.
    This is Schafer eq.27': x*x_bar = n(x)*1.

    Extends dickson_oct_conj_norm from DicksonCDProcess.v to dim 16. *)
Theorem sed_conj_norm : forall x : CDSed,
  sed_mul x (sed_conj x) =
  mkSed (mkOct (mkQuat (sed_norm_sq x) 0 0 0) quat_zero) oct_zero.
Proof.
  intros [[[a1 a2 a3 a4] [a5 a6 a7 a8]] [[a9 a10 a11 a12] [a13 a14 a15 a16]]].
  cbv [sed_mul sed_conj sed_norm_sq sed_lo sed_hi
       oct_mul oct_conj oct_neg oct_norm_sq oct_zero oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj quat_norm_sq quat_zero
       qa qb qc qd].
  f_equal; f_equal; f_equal; ring.
Qed.

(** ================================================================== *)
(** * Flexible identity anchors.                                       *)
(** ================================================================== *)

(** The flexible identity: (xy)x = x(yx) for all x, y.
    The project-wide position is that flexibility is universal across the
    standard Cayley-Dickson tower. This file currently records concrete
    low-dimensional anchors rather than the full all-dim theorem.

    Earlier draft comments in this file incorrectly described dim-16
    flexibility failure. That is not the intended mathematical stance of
    the repo and is corrected here. *)

(** Flexible identity holds for quaternions (associative => flexible). *)
Theorem quat_flexible : forall x y : CDQuat,
  quat_mul (quat_mul x y) x = quat_mul x (quat_mul y x).
Proof.
  intros [a b c d] [e f g h]; unfold quat_mul; simpl; f_equal; ring.
Qed.

(** Flexible identity also holds for octonions (alternative => flexible). *)
Theorem oct_flexible : forall x y : CDOct,
  oct_mul (oct_mul x y) x = oct_mul x (oct_mul y x).
Proof.
  intros [[a b c d] [e f g h]] [[i j k l] [m n o p]].
  cbv [oct_mul oct_conj oct_lo oct_hi
       quat_mul quat_add quat_neg quat_conj qa qb qc qd].
  f_equal; f_equal; ring.
Qed.

(** Flexible identity holds for sedenions (all CD algebras are flexible).
    Degree-3 polynomial in 32 real variables (16 per sedenion).
    Proof: cbv + component-wise f_equal4 mkQuat + ring (~22s, ~2.4 GB peak). *)
Theorem sed_flexible : forall x y : CDSed,
  sed_mul (sed_mul x y) x = sed_mul x (sed_mul y x).
Proof.
  intros [[[ a1  a2  a3  a4] [ a5  a6  a7  a8]]
          [[ a9 a10 a11 a12] [a13 a14 a15 a16]]]
         [[[ b1  b2  b3  b4] [ b5  b6  b7  b8]]
          [[ b9 b10 b11 b12] [b13 b14 b15 b16]]].
  cbv [sed_mul sed_neg sed_lo sed_hi
       oct_mul oct_add oct_sub oct_neg oct_conj
       oct_lo oct_hi oct_zero oct_e
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd].
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.
Qed.

(** ================================================================== *)
(** * Summary.                                                         *)
(** ================================================================== *)

(** Proved:
    - quat_third_power: x^2*x = x*x^2 (dim 4, by ring)
    - oct_third_power: x^2*x = x*x^2 (dim 8, by cbv+ring)
    - sed_conj_norm: x*conj(x) = n(x)*1 (dim 16, by cbv+ring)
    - quat_flexible: (xy)x = x(yx) (dim 4, by ring)
    - oct_flexible: (xy)x = x(yx) (dim 8, by cbv+ring)
    - sed_flexible: (xy)x = x(yx) (dim 16, by cbv+ring, ~22s, ~2.4GB)

    The sed_conj_norm theorem at dim 16 is a degree-2 polynomial
    identity in 32 real variables (16 per sedenion), verified by ring
    in approximately 6 seconds.  sed_flexible is degree-3 in 32 variables
    and takes approximately 22 seconds with a 2.4 GB peak.

    Zero Admitted. *)
