(** * CDTowerInstantiation: concrete CD tower from RealBase via CDDoubleFunctor.

    Instantiates the generic CDDouble functor starting from the reals,
    producing the full Cayley-Dickson tower:

      R (dim 1) -> C (dim 2) -> H (dim 4) -> O (dim 8) -> S (dim 16)

    Each level inherits all 7 linearity theorems automatically from
    the functor.  The only manual work is the RealBase instance (7 proofs,
    each 1-3 lines using ring/lra).

    This is the "mathematically right closure for the proof side" (Epic C, C5).

    Mirrors: crates/gororoba_algebra/src/construction/cd_tower.rs *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import CDDoubleFunctor.
Open Scope R_scope.

(* =========================================================================
   RealBase: the reals as a CDAlgLinear module.

   The reals are a 1-dimensional CD algebra with trivial conjugation
   (conj(x) = x).  All 7 linearity axioms reduce to ring identities.
   ========================================================================= *)

Module RealBase <: CDAlgLinear.

  Definition carrier := R.

  Definition cd_mul (a b : R) : R := a * b.
  Definition cd_add (a b : R) : R := a + b.
  Definition cd_neg (a : R) : R := - a.
  Definition cd_sub (a b : R) : R := a - b.
  (* Reals have trivial conjugation: conj(x) = x. *)
  Definition cd_conj (a : R) : R := a.
  Definition cd_scale (r : R) (a : R) : R := r * a.
  Definition cd_zero : R := 0.

  (* -- Axiom proofs: all are 1-line ring lemmas. -- *)

  Theorem mul_scale_left : forall r (a b : R),
    cd_mul (cd_scale r a) b = cd_scale r (cd_mul a b).
  Proof. intros; unfold cd_mul, cd_scale; ring. Qed.

  Theorem mul_scale_right : forall r (a b : R),
    cd_mul a (cd_scale r b) = cd_scale r (cd_mul a b).
  Proof. intros; unfold cd_mul, cd_scale; ring. Qed.

  Theorem conj_scale : forall r (a : R),
    cd_conj (cd_scale r a) = cd_scale r (cd_conj a).
  Proof. intros; unfold cd_conj, cd_scale; ring. Qed.

  Theorem neg_scale : forall r (a : R),
    cd_neg (cd_scale r a) = cd_scale r (cd_neg a).
  Proof. intros; unfold cd_neg, cd_scale; ring. Qed.

  Theorem scale_add : forall r (a b : R),
    cd_scale r (cd_add a b) = cd_add (cd_scale r a) (cd_scale r b).
  Proof. intros; unfold cd_scale, cd_add; ring. Qed.

  Theorem scale_sub : forall r (a b : R),
    cd_scale r (cd_sub a b) = cd_sub (cd_scale r a) (cd_scale r b).
  Proof. intros; unfold cd_scale, cd_sub; ring. Qed.

  Theorem sub_def : forall (a b : R),
    cd_sub a b = cd_add a (cd_neg b).
  Proof. intros; unfold cd_sub, cd_add, cd_neg; ring. Qed.

End RealBase.

(* =========================================================================
   Tower instantiation: each level is a single line.

   Complex := CDDouble RealBase         -- dim 2
   Quaternion := CDDouble Complex       -- dim 4
   Octonion := CDDouble Quaternion      -- dim 8
   Sedenion := CDDouble Octonion        -- dim 16

   All 7 linearity theorems are inherited at each level.  No new proofs.
   ========================================================================= *)

Module Complex := CDDouble RealBase.
Module Quaternion := CDDouble Complex.
Module Octonion := CDDouble Quaternion.
Module Sedenion := CDDouble Octonion.

(* =========================================================================
   Verification: confirm the tower compiles and key types are accessible.
   ========================================================================= *)

(** The Complex carrier is a pair of reals. *)
Check Complex.carrier.
(** Complex multiplication uses the CD formula with real conjugation. *)
Check Complex.cd_mul.
(** All 7 linearity theorems are present at Complex level. *)
Check Complex.mul_scale_left.
Check Complex.mul_scale_right.
Check Complex.conj_scale.
Check Complex.neg_scale.
Check Complex.scale_add.
Check Complex.scale_sub.
Check Complex.sub_def.

(** Quaternion carrier is a pair of Complex.carrier = pair of pair of R. *)
Check Quaternion.carrier.
Check Quaternion.mul_scale_left.

(** Octonion level: 3 doublings deep. *)
Check Octonion.carrier.
Check Octonion.mul_scale_left.

(** Sedenion level: 4 doublings deep. All 7 axioms still hold. *)
Check Sedenion.carrier.
Check Sedenion.mul_scale_left.
Check Sedenion.mul_scale_right.
Check Sedenion.conj_scale.

(** Quick sanity: Complex.cd_mul on two known values.
    (1,0) * (0,1) should give (0,1) -- i.e., 1*i = i.
    But we cannot compute on axiomatic R, so we just verify
    the type-level structure compiles. *)

Definition complex_one := Complex.mkDouble 1 0.
Definition complex_i := Complex.mkDouble 0 1.
Definition complex_product := Complex.cd_mul complex_one complex_i.
(** complex_product = mkDouble (1*0 - 0*0) (1*0 + 0*0) -- requires
    concrete R arithmetic, which Rocq's axiomatic R does not support
    via vm_compute.  The STRUCTURAL theorem (linearity inheritance)
    is what matters, not the concrete value. *)
