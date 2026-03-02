(** * BinaryEntropy: Shannon binary entropy function.

    H(p) = -(p*ln(p) + (1-p)*ln(1-p)) for p in (0,1).

    Properties:
    - Symmetric: H(p) = H(1-p)
    - Maximum at p = 1/2: H(1/2) = ln(2)
    - Nonnegative on (0,1)

    Mirrors: vacuum_frustration/src/immirzi_bridge.rs *)

From Stdlib Require Import Reals Lra Psatz Rpower.
Open Scope R_scope.

(** Binary entropy function. *)
Definition binary_entropy (p : R) : R :=
  - (p * ln p + (1 - p) * ln (1 - p)).

(** Symmetry: H(p) = H(1-p). *)
Theorem binary_entropy_symmetric : forall p,
  0 < p < 1 -> binary_entropy p = binary_entropy (1 - p).
Proof.
  intros p [Hp0 Hp1].
  unfold binary_entropy.
  replace (1 - (1 - p)) with p by lra.
  f_equal. apply Rplus_comm.
Qed.

(** H(1/2) = ln(2). *)
Theorem binary_entropy_max_value :
  binary_entropy (1/2) = ln 2.
Proof.
  unfold binary_entropy.
  replace (1 - 1 / 2) with (1 / 2) by lra.
  unfold Rdiv.
  rewrite !Rmult_1_l.
  rewrite !ln_Rinv by lra.
  lra.
Qed.

(** Helper: ln(x) < 0 for 0 < x < 1. *)
Lemma ln_neg : forall x, 0 < x -> x < 1 -> ln x < 0.
Proof.
  intros x Hx0 Hx1.
  rewrite <- ln_1.
  apply ln_increasing; assumption.
Qed.

(** Helper: positive * negative < 0. *)
Lemma pos_times_neg_lt_0 : forall a b,
  a > 0 -> b < 0 -> a * b < 0.
Proof.
  intros a b Ha Hb.
  replace 0 with (a * 0) by ring.
  apply Rmult_lt_compat_l; lra.
Qed.

(** H(p) >= 0 for p in (0, 1). *)
Theorem binary_entropy_nonneg : forall p,
  0 < p < 1 -> binary_entropy p >= 0.
Proof.
  intros p [Hp0 Hp1].
  unfold binary_entropy.
  assert (Hln_p : ln p < 0) by (apply ln_neg; lra).
  assert (Hln_1mp : ln (1 - p) < 0) by (apply ln_neg; lra).
  assert (H1 : p * ln p < 0) by (apply pos_times_neg_lt_0; lra).
  assert (H2 : (1 - p) * ln (1 - p) < 0) by (apply pos_times_neg_lt_0; lra).
  lra.
Qed.
