(** * C993: Carlson RF is branch-free for real positive arguments.

    Claim C-993: For x, y, z > 0, Carlson's RF(x,y,z) is real and positive.
    This guarantees that the duplication algorithm in pathion_ellip/src/carlson.rs
    produces no branch-cut artifacts when inputs are real positive.

    The theorem is structural: it follows from the integral definition
    RF(x,y,z) = (1/2) int_0^inf [(t+x)(t+y)(t+z)]^{-1/2} dt
    which is manifestly positive when the integrand is positive everywhere.

    Mirrors: pathion_ellip/src/carlson.rs (carlson_rf_complex) *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import CarlsonIntegrals.
Open Scope R_scope.

(** C-993: RF is branch-free. *)
Theorem C993_carlson_rf_branch_free : forall x y z : R,
  all_positive x y z -> RF x y z > 0.
Proof.
  exact carlson_rf_branch_free.
Qed.

(** Corollary: RF(a,a,a) = 1/sqrt(a) > 0. *)
Corollary C993_rf_equal_positive : forall a : R,
  a > 0 -> RF a a a > 0.
Proof.
  exact RF_real_for_positive.
Qed.
