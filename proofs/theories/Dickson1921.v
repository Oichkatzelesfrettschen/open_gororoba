(** * Dickson1921: Paper-scoped Rocq companion for Dickson (1921).

    Source:
      L.E. Dickson, "Quaternions and Their Generalizations,"
      Proc. Natl. Acad. Sci. USA 7(4), 1921, pp. 109-114.

    This short note complements the 1919 Dickson construction paper. Its main
    role in the repo is not to replace `DicksonCDProcess.v`, but to surface the
    companion viewpoint of deriving quaternion and direct-generalization
    structure from:
    - quadratic identities,
    - conjugation and norm,
    - norm composition,
    - and the recovery of the quaternion table without assuming associativity.

    Current Rocq landing:
    - `CayleyDicksonAlgebra.v`  : conjugation, norm, quadratic identities
    - `DicksonCDProcess.v`      : 1919 construction lane reused here
    - `HurwitzTheorem.v`        : dimension restriction for composition laws
    - this file                 : 1921 paper surface, parameterized generalized
                                  quaternion family, generalized/canonical
                                  matrix lanes, generalized basis-generator
                                  relations, and canonical infinitesimal lane

    Remaining Dickson 1921 backlog:
    - a fuller field-abstract lift beyond the current real-parameter
      surrogate used by the repo
    - the full infinitesimal-transformation reduction in paper order beyond the
      canonical n = 4 quaternion lane
    - the broader n > 4 exclusion lane internal to the note, rather than only
      by reference out to Hurwitz
*)

From Stdlib Require Import Arith Lia Lra Psatz Reals.
From OpenGororoba Require Export
  CayleyDicksonAlgebra DicksonCDProcess HurwitzTheorem FloatAxioms FloatQuaternion.

Open Scope R_scope.

(** ================================================================== *)
(** * Abstract paper surface.                                          *)
(** ================================================================== *)

(** Dickson's note works over a general field F and studies a linear algebra
    with principal unit, conjugation, norm, and a multiplicative norm law.
    The current repo is still real-number based, but we can already expose the
    paper's structural interface explicitly instead of keeping the file as a
    pure consequence wrapper.

    Paper alignment:
    - [d1921_mul], [d1921_scale], [d1921_one] model the linear algebra and
      principal unit from sections 2-4.
    - [d1921_conj] is Dickson's x' from equation (3).
    - [d1921_norm] is Dickson's o(x) from equations (3) and (4).
    - [d1921_norm_conj_right] and [d1921_norm_conj_left] encode xx' = o(x)
      and x'x = o(x) from equation (3).
    - [d1921_norm_multiplicative] encodes o(X) = o(x) o(xi) from equation (4).
*)
Record Dickson1921Surface (A : Type) := {
  d1921_mul : A -> A -> A;
  d1921_scale : R -> A -> A;
  d1921_one : A;
  d1921_conj : A -> A;
  d1921_norm : A -> R;
  d1921_mul_one_left : forall x, d1921_mul d1921_one x = x;
  d1921_mul_one_right : forall x, d1921_mul x d1921_one = x;
  d1921_conj_involution : forall x, d1921_conj (d1921_conj x) = x;
  d1921_norm_conj_right : forall x,
    d1921_mul x (d1921_conj x) = d1921_scale (d1921_norm x) d1921_one;
  d1921_norm_conj_left : forall x,
    d1921_mul (d1921_conj x) x = d1921_scale (d1921_norm x) d1921_one;
  d1921_norm_conj_preserved : forall x,
    d1921_norm (d1921_conj x) = d1921_norm x;
  d1921_norm_multiplicative : forall x y,
    d1921_norm (d1921_mul x y) = d1921_norm x * d1921_norm y
}.

Arguments d1921_mul {A} _ _ _.
Arguments d1921_scale {A} _ _ _.
Arguments d1921_one {A} _.
Arguments d1921_conj {A} _ _.
Arguments d1921_norm {A} _ _.
Arguments d1921_mul_one_left {A} _ _.
Arguments d1921_mul_one_right {A} _ _.
Arguments d1921_conj_involution {A} _ _.
Arguments d1921_norm_conj_right {A} _ _.
Arguments d1921_norm_conj_left {A} _ _.
Arguments d1921_norm_conj_preserved {A} _ _.
Arguments d1921_norm_multiplicative {A} _ _ _.

Record Dickson1921Section5Coords := {
  d1921_s5_scalar : R;
  d1921_s5_tail : nat -> R
}.

Definition d1921_s5_coord (x : Dickson1921Section5Coords) (i : nat) : R :=
  match i with
  | 0%nat => d1921_s5_scalar x
  | S j => d1921_s5_tail x (S j)
  end.

Definition d1921_s5_single_tail (k : nat) (eps : R) : nat -> R :=
  fun i => if Nat.eqb i k then eps else 0.

Definition d1921_s5_near_identity (k : nat) (eps : R) : Dickson1921Section5Coords :=
  {| d1921_s5_scalar := 1;
     d1921_s5_tail := d1921_s5_single_tail k eps |}.

Section Dickson1921Derived.
  Context {A : Type} (S : Dickson1921Surface A).

  Theorem d1921_eq3_right : forall x,
    d1921_mul S x (d1921_conj S x) =
    d1921_scale S (d1921_norm S x) (d1921_one S).
  Proof.
    exact (d1921_norm_conj_right S).
  Qed.

  Theorem d1921_eq3_left : forall x,
    d1921_mul S (d1921_conj S x) x =
    d1921_scale S (d1921_norm S x) (d1921_one S).
  Proof.
    exact (d1921_norm_conj_left S).
  Qed.

  Theorem d1921_eq4 : forall x y,
    d1921_norm S (d1921_mul S x y) =
    d1921_norm S x * d1921_norm S y.
  Proof.
    exact (d1921_norm_multiplicative S).
  Qed.

  Theorem d1921_norm_product_with_right_conjugate : forall x y,
    d1921_norm S (d1921_mul S x (d1921_conj S y)) =
    d1921_norm S x * d1921_norm S y.
  Proof.
    intros x y.
    rewrite d1921_eq4.
    rewrite d1921_norm_conj_preserved.
    reflexivity.
  Qed.

  Theorem d1921_norm_product_with_left_conjugate : forall x y,
    d1921_norm S (d1921_mul S (d1921_conj S x) y) =
    d1921_norm S x * d1921_norm S y.
  Proof.
    intros x y.
    rewrite d1921_eq4.
    rewrite d1921_norm_conj_preserved.
    reflexivity.
  Qed.
End Dickson1921Derived.

(** ================================================================== *)
(** * FLOAT_OPS-based coefficient lift.                                *)
(** ================================================================== *)

(** The repo already has a generic coefficient interface [`FLOAT_OPS`] and a
    functorized quaternion lane [`QuatOps`]. Dickson 1921 can start its
    field-abstract lift there without inventing a second scalar hierarchy.

    This module lands the low-cost section-3 structural fragment abstractly:
    principal unit, conjugation involution, and norm preservation under
    conjugation. The heavier norm-conjugate and norm-multiplicative paper
    consequences still remain in the concrete real-number lane for now. *)
Module Dickson1921FloatSection3 (F : FLOAT_OPS).
  Module Q := QuatOps F.

  Theorem d1921_float_eq3_unit_left : forall q : Q.Quat,
    Q.quat_mul Q.quat_one q = q.
  Proof.
    exact Q.quat_mul_one_left.
  Qed.

  Theorem d1921_float_eq3_unit_right : forall q : Q.Quat,
    Q.quat_mul q Q.quat_one = q.
  Proof.
    exact Q.quat_mul_one_right.
  Qed.

  Theorem d1921_float_conj_involution : forall q : Q.Quat,
    Q.quat_conj (Q.quat_conj q) = q.
  Proof.
    exact Q.quat_conj_involution.
  Qed.

  Theorem d1921_float_norm_conj_preserved : forall q : Q.Quat,
    Q.quat_norm_sq (Q.quat_conj q) = Q.quat_norm_sq q.
  Proof.
    exact Q.quat_norm_sq_conj.
  Qed.

  Theorem d1921_float_section3_summary :
    (forall q : Q.Quat, Q.quat_mul Q.quat_one q = q) /\
    (forall q : Q.Quat, Q.quat_mul q Q.quat_one = q) /\
    (forall q : Q.Quat, Q.quat_conj (Q.quat_conj q) = q) /\
    (forall q : Q.Quat, Q.quat_norm_sq (Q.quat_conj q) = Q.quat_norm_sq q).
  Proof.
    repeat split.
    - exact d1921_float_eq3_unit_left.
    - exact d1921_float_eq3_unit_right.
    - exact d1921_float_conj_involution.
    - exact d1921_float_norm_conj_preserved.
  Qed.
End Dickson1921FloatSection3.

Module Dickson1921FloatSection4 (F : FLOAT_OPS).
  Import F.
  Module Q := QuatOps F.

  Theorem d1921_float_eq3_right : forall q : Q.Quat,
    Q.quat_mul q (Q.quat_conj q) = Q.quat_scale (Q.quat_norm_sq q) Q.quat_one.
  Proof.
    exact Q.quat_norm_conjugate.
  Qed.

  Theorem d1921_float_eq3_left : forall q : Q.Quat,
    Q.quat_mul (Q.quat_conj q) q = Q.quat_scale (Q.quat_norm_sq q) Q.quat_one.
  Proof.
    exact Q.quat_conj_norm_left.
  Qed.

  Theorem d1921_float_eq4 : forall p q : Q.Quat,
    Q.quat_norm_sq (Q.quat_mul p q) = Q.quat_norm_sq p * Q.quat_norm_sq q.
  Proof.
    exact Q.quat_norm_mul.
  Qed.

  Theorem d1921_float_mul_assoc : forall x y z : Q.Quat,
    Q.quat_mul (Q.quat_mul x y) z = Q.quat_mul x (Q.quat_mul y z).
  Proof.
    exact Q.quat_mul_assoc.
  Qed.

  Theorem d1921_float_section4_summary :
    (forall q : Q.Quat,
        Q.quat_mul q (Q.quat_conj q) =
        Q.quat_scale (Q.quat_norm_sq q) Q.quat_one) /\
    (forall q : Q.Quat,
        Q.quat_mul (Q.quat_conj q) q =
        Q.quat_scale (Q.quat_norm_sq q) Q.quat_one) /\
    (forall p q : Q.Quat,
        Q.quat_norm_sq (Q.quat_mul p q) = Q.quat_norm_sq p * Q.quat_norm_sq q) /\
    (forall x y z : Q.Quat,
        Q.quat_mul (Q.quat_mul x y) z = Q.quat_mul x (Q.quat_mul y z)).
  Proof.
    repeat split.
    - exact d1921_float_eq3_right.
    - exact d1921_float_eq3_left.
    - exact d1921_float_eq4.
    - exact d1921_float_mul_assoc.
  Qed.
End Dickson1921FloatSection4.

Module Dickson1921FloatQuaternionCase (F : FLOAT_OPS).
  Import F.
  Module Q := QuatOps F.

  Theorem d1921_float_quaternion_table :
    Q.quat_mul Q.qi Q.qi = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qj Q.qj = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qk Q.qk = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qi Q.qj = Q.qk /\
    Q.quat_mul Q.qj Q.qk = Q.qi /\
    Q.quat_mul Q.qk Q.qi = Q.qj /\
    Q.quat_mul Q.qj Q.qi = Q.quat_neg Q.qk /\
    Q.quat_mul Q.qk Q.qj = Q.quat_neg Q.qi /\
    Q.quat_mul Q.qi Q.qk = Q.quat_neg Q.qj.
  Proof.
    exact Q.quat_basis_table.
  Qed.

  Theorem d1921_float_quaternion_case_summary :
    (forall x y z : Q.Quat,
        Q.quat_mul (Q.quat_mul x y) z = Q.quat_mul x (Q.quat_mul y z)) /\
    Q.quat_mul Q.qi Q.qi = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qj Q.qj = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qk Q.qk = Q.quat_scale (opp one) Q.quat_one /\
    Q.quat_mul Q.qi Q.qj = Q.qk /\
    Q.quat_mul Q.qj Q.qk = Q.qi /\
    Q.quat_mul Q.qk Q.qi = Q.qj /\
    Q.quat_mul Q.qj Q.qi = Q.quat_neg Q.qk /\
    Q.quat_mul Q.qk Q.qj = Q.quat_neg Q.qi /\
    Q.quat_mul Q.qi Q.qk = Q.quat_neg Q.qj.
  Proof.
    split.
    - exact Q.quat_mul_assoc.
    - exact d1921_float_quaternion_table.
  Qed.
End Dickson1921FloatQuaternionCase.

Module Dickson1921FloatMatrixLane (F : FLOAT_OPS).
  Import F.
  Module Q := QuatOps F.

  Theorem d1921_float_matrix_action :
    forall x xi : Q.Quat, Q.quat_hamilton_apply x xi = Q.quat_mul x xi.
  Proof.
    exact Q.quat_hamilton_apply_eq_mul.
  Qed.

  Theorem d1921_float_eq13_dim4 :
    forall x xi : Q.Quat,
      Q.quat_norm_sq (Q.quat_hamilton_apply x xi) =
      Q.quat_norm_sq x * Q.quat_norm_sq xi.
  Proof.
    exact Q.quat_eq13_dim4.
  Qed.

  Theorem d1921_float_eq15_dim4 :
    forall x : Q.Quat, forall i j : nat,
      (i < 4)%nat ->
      (j < 4)%nat ->
      Q.quat_hamilton_gram_entry x i j =
      Q.quat_norm_sq x * Q.quat_identity_entry i j.
  Proof.
    exact Q.quat_eq15_dim4.
  Qed.

  Theorem d1921_float_eq16_basis_skew :
    forall k i j : nat,
      (1 <= k <= 3)%nat ->
      (i < 4)%nat ->
      (j < 4)%nat ->
      Q.quat_matrix_transpose (Q.quat_hamilton_basis_entry k) i j =
      opp (Q.quat_hamilton_basis_entry k i j).
  Proof.
    exact Q.quat_eq16_basis_skew.
  Qed.

  Theorem d1921_float_eq16_basis_square :
    forall k i j : nat,
      (1 <= k <= 3)%nat ->
      (i < 4)%nat ->
      (j < 4)%nat ->
      Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry k)
                              (Q.quat_hamilton_basis_entry k) i j =
      opp (Q.quat_identity_entry i j).
  Proof.
    exact Q.quat_eq16_basis_square.
  Qed.

  Theorem d1921_float_eq16_basis_anticommute :
    forall k l i j : nat,
      (1 <= k <= 3)%nat ->
      (1 <= l <= 3)%nat ->
      k <> l ->
      (i < 4)%nat ->
      (j < 4)%nat ->
      Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry k)
                              (Q.quat_hamilton_basis_entry l) i j +
      Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry l)
                              (Q.quat_hamilton_basis_entry k) i j = zero.
  Proof.
    exact Q.quat_eq16_basis_anticommute.
  Qed.

  Theorem d1921_float_matrix_summary :
    (forall x xi : Q.Quat, Q.quat_hamilton_apply x xi = Q.quat_mul x xi) /\
    (forall x xi : Q.Quat,
        Q.quat_norm_sq (Q.quat_hamilton_apply x xi) =
        Q.quat_norm_sq x * Q.quat_norm_sq xi) /\
    (forall x : Q.Quat, forall i j : nat,
        (i < 4)%nat ->
        (j < 4)%nat ->
        Q.quat_hamilton_gram_entry x i j =
        Q.quat_norm_sq x * Q.quat_identity_entry i j) /\
    (forall k i j : nat,
        (1 <= k <= 3)%nat ->
        (i < 4)%nat ->
        (j < 4)%nat ->
        Q.quat_matrix_transpose (Q.quat_hamilton_basis_entry k) i j =
        opp (Q.quat_hamilton_basis_entry k i j)) /\
    (forall k i j : nat,
        (1 <= k <= 3)%nat ->
        (i < 4)%nat ->
        (j < 4)%nat ->
        Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry k)
                                (Q.quat_hamilton_basis_entry k) i j =
        opp (Q.quat_identity_entry i j)) /\
    (forall k l i j : nat,
        (1 <= k <= 3)%nat ->
        (1 <= l <= 3)%nat ->
        k <> l ->
        (i < 4)%nat ->
        (j < 4)%nat ->
        Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry k)
                                (Q.quat_hamilton_basis_entry l) i j +
        Q.quat_matrix_mul_entry (Q.quat_hamilton_basis_entry l)
                                (Q.quat_hamilton_basis_entry k) i j = zero).
  Proof.
    repeat split.
    - exact d1921_float_matrix_action.
    - exact d1921_float_eq13_dim4.
    - exact d1921_float_eq15_dim4.
    - exact d1921_float_eq16_basis_skew.
    - exact d1921_float_eq16_basis_square.
    - exact d1921_float_eq16_basis_anticommute.
  Qed.
End Dickson1921FloatMatrixLane.

Module Dickson1921FloatParamLane (F : FLOAT_OPS).
  Import F.
  Module Q := QuatOps F.

  Theorem d1921_float_param_norm_summary :
    forall c2 c3 : t,
      (forall x, Q.quat_param_mul c2 c3 Q.quat_one x = x) /\
      (forall x, Q.quat_param_mul c2 c3 x Q.quat_one = x) /\
      (forall x,
          Q.quat_param_mul c2 c3 x (Q.quat_conj x) =
          Q.quat_scale (Q.quat_param_norm c2 c3 x) Q.quat_one) /\
      (forall x,
          Q.quat_param_mul c2 c3 (Q.quat_conj x) x =
          Q.quat_scale (Q.quat_param_norm c2 c3 x) Q.quat_one) /\
      (forall x,
          Q.quat_param_norm c2 c3 (Q.quat_conj x) =
          Q.quat_param_norm c2 c3 x) /\
      (forall x y,
          Q.quat_param_norm c2 c3 (Q.quat_param_mul c2 c3 x y) =
          Q.quat_param_norm c2 c3 x * Q.quat_param_norm c2 c3 y).
  Proof.
    intros c2 c3.
    repeat split.
    - exact (Q.quat_param_mul_one_left c2 c3).
    - exact (Q.quat_param_mul_one_right c2 c3).
    - exact (Q.quat_param_norm_conj_right c2 c3).
    - exact (Q.quat_param_norm_conj_left c2 c3).
    - exact (Q.quat_param_norm_conj_preserved c2 c3).
    - exact (Q.quat_param_norm_mul c2 c3).
  Qed.

  Theorem d1921_float_param_table_summary :
    forall c2 c3 : t,
      Q.quat_param_mul c2 c3 Q.qi Q.qi = Q.quat_scale c2 Q.quat_one /\
      Q.quat_param_mul c2 c3 Q.qj Q.qj = Q.quat_scale c3 Q.quat_one /\
      Q.quat_param_mul c2 c3 Q.qk Q.qk =
        Q.quat_scale (opp c2 * c3) Q.quat_one /\
      Q.quat_param_mul c2 c3 Q.qi Q.qj = Q.qk /\
      Q.quat_param_mul c2 c3 Q.qj Q.qi = Q.quat_neg Q.qk /\
      Q.quat_param_mul c2 c3 Q.qi Q.qk = Q.quat_scale c2 Q.qj /\
      Q.quat_param_mul c2 c3 Q.qk Q.qi = Q.quat_scale (opp c2) Q.qj /\
      Q.quat_param_mul c2 c3 Q.qj Q.qk = Q.quat_scale (opp c3) Q.qi /\
      Q.quat_param_mul c2 c3 Q.qk Q.qj = Q.quat_scale c3 Q.qi.
  Proof.
    intros c2 c3.
    repeat split;
      [ exact (Q.quat_param_qi_sq c2 c3)
      | exact (Q.quat_param_qj_sq c2 c3)
      | exact (Q.quat_param_qk_sq c2 c3)
      | exact (Q.quat_param_qi_qj c2 c3)
      | exact (Q.quat_param_qj_qi c2 c3)
      | exact (Q.quat_param_qi_qk c2 c3)
      | exact (Q.quat_param_qk_qi c2 c3)
      | exact (Q.quat_param_qj_qk c2 c3)
      | exact (Q.quat_param_qk_qj c2 c3) ].
  Qed.

  Theorem d1921_float_param_matrix_summary :
    forall c2 c3 : t,
      (forall x xi,
          Q.quat_param_matrix_apply c2 c3 x xi =
          Q.quat_param_mul c2 c3 x xi) /\
      (forall x xi,
          Q.quat_param_norm c2 c3 (Q.quat_param_matrix_apply c2 c3 x xi) =
          Q.quat_param_norm c2 c3 x * Q.quat_param_norm c2 c3 xi) /\
      (forall x i j,
          (i < 4)%nat ->
          (j < 4)%nat ->
          Q.quat_param_gram_entry c2 c3 x i j =
          Q.quat_param_norm c2 c3 x * Q.quat_param_form_entry c2 c3 i j).
  Proof.
    intros c2 c3.
    repeat split.
    - exact (Q.quat_param_matrix_apply_eq_mul c2 c3).
    - exact (Q.quat_param_eq13_dim4 c2 c3).
    - exact (Q.quat_param_eq15_dim4 c2 c3).
  Qed.

  Theorem d1921_float_param_infinitesimal_summary :
    forall c2 c3 : t,
      (forall x k eps i,
          (1 <= k <= 3)%nat ->
          (i < 4)%nat ->
          Q.quat_coord (Q.quat_param_mul c2 c3 x (Q.quat_near_identity k eps)) i =
          Q.quat_coord x i + eps * Q.quat_param_eq6_delta_coord c2 c3 k x i) /\
      (forall k eps,
          (1 <= k <= 3)%nat ->
          Q.quat_param_norm c2 c3 (Q.quat_near_identity k eps) =
          one + Q.quat_param_near_identity_quadratic_factor c2 c3 k * (eps * eps)) /\
      (forall x k eps,
          (1 <= k <= 3)%nat ->
          Q.quat_param_norm c2 c3
            (Q.quat_param_mul c2 c3 x (Q.quat_near_identity k eps)) -
          Q.quat_param_norm c2 c3 x =
          (eps * eps) * Q.quat_param_near_identity_quadratic_factor c2 c3 k *
            Q.quat_param_norm c2 c3 x) /\
      (forall x k,
          (1 <= k <= 3)%nat ->
          Q.quat_param_eq7_linear_form c2 c3 k x = zero).
  Proof.
    intros c2 c3.
    repeat split.
    - exact (Q.quat_param_eq6_dim4 c2 c3).
    - exact (Q.quat_param_near_identity_norm c2 c3).
    - intros x k eps Hk.
      exact (Q.quat_param_infinitesimal_no_linear_term c2 c3 k eps x Hk).
    - intros x k Hk.
      exact (Q.quat_param_eq7_linear_form_vanish c2 c3 k x Hk).
  Qed.
End Dickson1921FloatParamLane.

(** The canonical quaternion lane inhabits Dickson's paper surface inside the
    current real-number formalization. *)
Theorem quat_mul_one_left : forall q : CDQuat,
  quat_mul quat_one q = q.
Proof.
  intros [a b c d].
  unfold quat_mul, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem quat_mul_one_right : forall q : CDQuat,
  quat_mul q quat_one = q.
Proof.
  intros [a b c d].
  unfold quat_mul, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem quat_norm_conj_preserved : forall q : CDQuat,
  quat_norm_sq (quat_conj q) = quat_norm_sq q.
Proof.
  intros [a b c d].
  unfold quat_norm_sq, quat_conj.
  simpl.
  ring.
Qed.

Definition dickson1921_quat_surface : Dickson1921Surface CDQuat.
Proof.
  refine
    {| d1921_mul := quat_mul;
       d1921_scale := quat_scale;
       d1921_one := quat_one;
       d1921_conj := quat_conj;
       d1921_norm := quat_norm_sq |}.
  - exact quat_mul_one_left.
  - exact quat_mul_one_right.
  - exact quat_conj_involution.
  - intros x.
    unfold quat_scale, quat_one.
    simpl.
    replace
      (mkQuat (quat_norm_sq x * 1) (quat_norm_sq x * 0)
              (quat_norm_sq x * 0) (quat_norm_sq x * 0))
      with (mkQuat (quat_norm_sq x) 0 0 0) by (f_equal; ring).
    exact (quat_norm_conjugate x).
  - intros x.
    unfold quat_scale, quat_one.
    simpl.
    replace
      (mkQuat (quat_norm_sq x * 1) (quat_norm_sq x * 0)
              (quat_norm_sq x * 0) (quat_norm_sq x * 0))
      with (mkQuat (quat_norm_sq x) 0 0 0) by (f_equal; ring).
    exact (quat_conj_norm_left x).
  - exact quat_norm_conj_preserved.
  - exact quat_norm_mul.
Defined.

(** ================================================================== *)
(** * Parameterized generalized quaternion family.                     *)
(** ================================================================== *)

(** Dickson's 1921 note derives a direct generalization of quaternions over a
    field F.  The current repo is still real-number based, but we can model
    the same algebraic pattern over parameters c2, c3 in R.  This gives an
    honest "general-field-shaped" surrogate without pretending we already have
    a full abstract field library in the proof tree. *)

Definition dickson1921_param_mul (c2 c3 : R) (p q : CDQuat) : CDQuat :=
  mkQuat
    (qa p * qa q + c2 * qb p * qb q + c3 * qc p * qc q - c2 * c3 * qd p * qd q)
    (qa p * qb q + qb p * qa q - c3 * qc p * qd q + c3 * qd p * qc q)
    (qa p * qc q + qc p * qa q + c2 * qb p * qd q - c2 * qd p * qb q)
    (qa p * qd q + qd p * qa q + qb p * qc q - qc p * qb q).

Definition dickson1921_param_norm (c2 c3 : R) (q : CDQuat) : R :=
  (qa q)^2 - c2 * (qb q)^2 - c3 * (qc q)^2 + c2 * c3 * (qd q)^2.

Theorem dickson1921_param_mul_one_left : forall c2 c3 q,
  dickson1921_param_mul c2 c3 quat_one q = q.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_param_mul, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_mul_one_right : forall c2 c3 q,
  dickson1921_param_mul c2 c3 q quat_one = q.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_param_mul, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_conj_involution : forall q,
  quat_conj (quat_conj q) = q.
Proof.
  exact quat_conj_involution.
Qed.

Theorem dickson1921_param_norm_conj_preserved : forall c2 c3 q,
  dickson1921_param_norm c2 c3 (quat_conj q) =
  dickson1921_param_norm c2 c3 q.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_param_norm, quat_conj.
  simpl.
  ring.
Qed.

Theorem dickson1921_param_norm_conj_right : forall c2 c3 q,
  dickson1921_param_mul c2 c3 q (quat_conj q) =
  quat_scale (dickson1921_param_norm c2 c3 q) quat_one.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_param_mul, dickson1921_param_norm,
         quat_conj, quat_scale, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_norm_conj_left : forall c2 c3 q,
  dickson1921_param_mul c2 c3 (quat_conj q) q =
  quat_scale (dickson1921_param_norm c2 c3 q) quat_one.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_param_mul, dickson1921_param_norm,
         quat_conj, quat_scale, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_norm_mul : forall c2 c3 p q,
  dickson1921_param_norm c2 c3 (dickson1921_param_mul c2 c3 p q) =
  dickson1921_param_norm c2 c3 p * dickson1921_param_norm c2 c3 q.
Proof.
  intros c2 c3 [a b c d] [e f g h].
  unfold dickson1921_param_norm, dickson1921_param_mul.
  simpl.
  ring.
Qed.

Theorem dickson1921_param_mul_assoc : forall c2 c3 x y z,
  dickson1921_param_mul c2 c3 (dickson1921_param_mul c2 c3 x y) z =
  dickson1921_param_mul c2 c3 x (dickson1921_param_mul c2 c3 y z).
Proof.
  intros c2 c3 [a b c d] [e f g h] [i j k l].
  unfold dickson1921_param_mul.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_i_sq : forall c2 c3,
  dickson1921_param_mul c2 c3 qi qi = quat_scale c2 quat_one.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, quat_scale, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_j_sq : forall c2 c3,
  dickson1921_param_mul c2 c3 qj qj = quat_scale c3 quat_one.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qj, quat_scale, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_k_sq : forall c2 c3,
  dickson1921_param_mul c2 c3 qk qk = quat_scale (- c2 * c3) quat_one.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qk, quat_scale, quat_one.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_ij : forall c2 c3,
  dickson1921_param_mul c2 c3 qi qj = qk.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qj, qk.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_ji : forall c2 c3,
  dickson1921_param_mul c2 c3 qj qi = quat_neg qk.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qj, qk, quat_neg.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_ik : forall c2 c3,
  dickson1921_param_mul c2 c3 qi qk = quat_scale c2 qj.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qk, qj, quat_scale.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_ki : forall c2 c3,
  dickson1921_param_mul c2 c3 qk qi = quat_scale (- c2) qj.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qk, qj, quat_scale.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_jk : forall c2 c3,
  dickson1921_param_mul c2 c3 qj qk = quat_scale (- c3) qi.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qj, qk, quat_scale.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_kj : forall c2 c3,
  dickson1921_param_mul c2 c3 qk qj = quat_scale c3 qi.
Proof.
  intros c2 c3.
  unfold dickson1921_param_mul, qi, qj, qk, quat_scale.
  simpl.
  f_equal; ring.
Qed.

Definition dickson1921_param_surface (c2 c3 : R) : Dickson1921Surface CDQuat.
Proof.
  refine
    {| d1921_mul := dickson1921_param_mul c2 c3;
       d1921_scale := quat_scale;
       d1921_one := quat_one;
       d1921_conj := quat_conj;
       d1921_norm := dickson1921_param_norm c2 c3 |}.
  - exact (dickson1921_param_mul_one_left c2 c3).
  - exact (dickson1921_param_mul_one_right c2 c3).
  - exact dickson1921_param_conj_involution.
  - exact (dickson1921_param_norm_conj_right c2 c3).
  - exact (dickson1921_param_norm_conj_left c2 c3).
  - exact (dickson1921_param_norm_conj_preserved c2 c3).
  - exact (dickson1921_param_norm_mul c2 c3).
Defined.

Theorem dickson1921_param_specializes_to_standard_mul : forall p q,
  dickson1921_param_mul (-1) (-1) p q = quat_mul p q.
Proof.
  intros [a b c d] [e f g h].
  unfold dickson1921_param_mul, quat_mul.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_specializes_to_standard_norm : forall q,
  dickson1921_param_norm (-1) (-1) q = quat_norm_sq q.
Proof.
  intros [a b c d].
  unfold dickson1921_param_norm, quat_norm_sq.
  simpl.
  ring.
Qed.

(** This is the paper's generalized quaternion family in the repo's current
    real-parameter surrogate. *)
Theorem dickson1921_generalized_quaternion_family : forall (c2 c3 : R),
  Dickson1921Surface CDQuat.
Proof.
  intros c2 c3.
  exact (dickson1921_param_surface c2 c3).
Qed.

(** ================================================================== *)
(** * Canonical quaternion consequences.                               *)
(** ================================================================== *)

(** Every quaternion satisfies a quadratic over the base field. *)
Theorem dickson1921_quaternion_quadratic_identity : forall q : CDQuat,
  quat_mul q q =
  quat_add (quat_scale (2 * qa q) q)
           (quat_scale (- quat_norm_sq q) quat_one).
Proof.
  exact quat_quadratic_identity.
Qed.

(** Product with the conjugate equals the norm. *)
Theorem dickson1921_quaternion_norm_conjugate : forall q : CDQuat,
  quat_mul q (quat_conj q) = mkQuat (quat_norm_sq q) 0 0 0.
Proof.
  exact quat_norm_conjugate.
Qed.

(** The norm of a product is the product of the norms. *)
Theorem dickson1921_quaternion_norm_multiplicative : forall p q : CDQuat,
  quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q.
Proof.
  exact quat_norm_mul.
Qed.

(** The standard quaternion table is recovered without taking associativity as
    a starting axiom in the paper. The concrete multiplication witnesses here
    are inherited from the 1919 construction lane. *)
Theorem dickson1921_quaternion_table :
  quat_mul qi qj = qk /\
  quat_mul qj qk = qi /\
  quat_mul qk qi = qj /\
  quat_mul qj qi = mkQuat 0 0 0 (-1) /\
  quat_mul qk qj = mkQuat 0 (-1) 0 0 /\
  quat_mul qi qk = mkQuat 0 0 (-1) 0.
Proof.
  repeat split;
    [ exact dickson_eq3_ij
    | exact dickson_eq3_jk
    | exact dickson_eq3_ki
    | exact dickson_eq3_ji
    | exact dickson_eq3_kj
    | exact dickson_eq3_ik ].
Qed.

(** The note points back to Hurwitz's composition-dimension restriction. *)
Theorem dickson1921_composition_dimensions_tracked :
  forall n : nat,
  tracked_cd_tower_dimension n ->
  (hurwitz_radon n = n <-> hurwitz_square_dimension n).
Proof.
  exact hurwitz_cd_tower_classification.
Qed.

(** ================================================================== *)
(** * Matrix language for the n = 4 quaternion lane.                   *)
(** ================================================================== *)

(** Dickson's sections 6-7 switch to the matrix language

      X = M(x) * xi,

    and then impose M(x)^T * M(x) = o(x) * I_4.  We formalize this in the
    canonical quaternion case as an explicit 4x4 Hamilton matrix. *)

Definition quat_coord (q : CDQuat) (i : nat) : R :=
  match i with
  | 0 => qa q
  | 1 => qb q
  | 2 => qc q
  | 3 => qd q
  | _ => 0
  end.

Definition dickson1921_s5_of_quat (q : CDQuat) : Dickson1921Section5Coords :=
  {| d1921_s5_scalar := qa q;
     d1921_s5_tail := fun i => quat_coord q i |}.

Theorem dickson1921_s5_of_quat_coord : forall q i,
  d1921_s5_coord (dickson1921_s5_of_quat q) i = quat_coord q i.
Proof.
  intros q [|i]; reflexivity.
Qed.

(** Dickson's equation (5), specialized to the dim-4 parameterized family.
    Repo indexing is 0-based, so paper coordinates x_1,...,x_4 correspond to
    [quat_coord] indices 0,1,2,3 respectively. *)
Theorem dickson1921_param_eq5_dim4 : forall c2 c3 x xi,
  quat_coord (dickson1921_param_mul c2 c3 x xi) 0%nat =
    quat_coord x 0%nat * quat_coord xi 0%nat +
    c2 * quat_coord x 1%nat * quat_coord xi 1%nat +
    c3 * quat_coord x 2%nat * quat_coord xi 2%nat -
    c2 * c3 * quat_coord x 3%nat * quat_coord xi 3%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 1%nat =
    quat_coord x 0%nat * quat_coord xi 1%nat +
    quat_coord x 1%nat * quat_coord xi 0%nat -
    c3 * quat_coord x 2%nat * quat_coord xi 3%nat +
    c3 * quat_coord x 3%nat * quat_coord xi 2%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 2%nat =
    quat_coord x 0%nat * quat_coord xi 2%nat +
    quat_coord x 2%nat * quat_coord xi 0%nat +
    c2 * quat_coord x 1%nat * quat_coord xi 3%nat -
    c2 * quat_coord x 3%nat * quat_coord xi 1%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 3%nat =
    quat_coord x 0%nat * quat_coord xi 3%nat +
    quat_coord x 3%nat * quat_coord xi 0%nat +
    quat_coord x 1%nat * quat_coord xi 2%nat -
    quat_coord x 2%nat * quat_coord xi 1%nat.
Proof.
  intros c2 c3 [a b c d] [e f g h].
  unfold dickson1921_param_mul, quat_coord.
  simpl.
  repeat split; ring.
Qed.

Definition quat_identity_entry (i j : nat) : R :=
  if Nat.eqb i j then 1 else 0.

Definition quat_hamilton_basis_entry (k i j : nat) : R :=
  match k, i, j with
  | 0, 0, 0 => 1 | 0, 1, 1 => 1 | 0, 2, 2 => 1 | 0, 3, 3 => 1
  | 1, 0, 1 => -1 | 1, 1, 0 => 1 | 1, 2, 3 => -1 | 1, 3, 2 => 1
  | 2, 0, 2 => -1 | 2, 1, 3 => 1 | 2, 2, 0 => 1 | 2, 3, 1 => -1
  | 3, 0, 3 => -1 | 3, 1, 2 => -1 | 3, 2, 1 => 1 | 3, 3, 0 => 1
  | _, _, _ => 0
  end.

Definition quat_hamilton_entry (x : CDQuat) (i j : nat) : R :=
  qa x * quat_hamilton_basis_entry 0%nat i j +
  qb x * quat_hamilton_basis_entry 1%nat i j +
  qc x * quat_hamilton_basis_entry 2%nat i j +
  qd x * quat_hamilton_basis_entry 3%nat i j.

Definition quat_hamilton_apply (x xi : CDQuat) : CDQuat :=
  mkQuat
    (quat_hamilton_entry x 0%nat 0%nat * quat_coord xi 0%nat +
     quat_hamilton_entry x 0%nat 1%nat * quat_coord xi 1%nat +
     quat_hamilton_entry x 0%nat 2%nat * quat_coord xi 2%nat +
     quat_hamilton_entry x 0%nat 3%nat * quat_coord xi 3%nat)
    (quat_hamilton_entry x 1%nat 0%nat * quat_coord xi 0%nat +
     quat_hamilton_entry x 1%nat 1%nat * quat_coord xi 1%nat +
     quat_hamilton_entry x 1%nat 2%nat * quat_coord xi 2%nat +
     quat_hamilton_entry x 1%nat 3%nat * quat_coord xi 3%nat)
    (quat_hamilton_entry x 2%nat 0%nat * quat_coord xi 0%nat +
     quat_hamilton_entry x 2%nat 1%nat * quat_coord xi 1%nat +
     quat_hamilton_entry x 2%nat 2%nat * quat_coord xi 2%nat +
     quat_hamilton_entry x 2%nat 3%nat * quat_coord xi 3%nat)
    (quat_hamilton_entry x 3%nat 0%nat * quat_coord xi 0%nat +
     quat_hamilton_entry x 3%nat 1%nat * quat_coord xi 1%nat +
     quat_hamilton_entry x 3%nat 2%nat * quat_coord xi 2%nat +
     quat_hamilton_entry x 3%nat 3%nat * quat_coord xi 3%nat).

Definition quat_matrix_transpose (m : nat -> nat -> R) (i j : nat) : R :=
  m j i.

Definition quat_matrix_mul_entry (m n : nat -> nat -> R) (i j : nat) : R :=
  m i 0%nat * n 0%nat j +
  m i 1%nat * n 1%nat j +
  m i 2%nat * n 2%nat j +
  m i 3%nat * n 3%nat j.

Definition quat_hamilton_gram_entry (x : CDQuat) (i j : nat) : R :=
  quat_matrix_mul_entry (quat_matrix_transpose (quat_hamilton_entry x))
                        (quat_hamilton_entry x) i j.

(** For the parameterized family, Dickson's quadratic form is diagonal:
      Q = diag(1, -c2, -c3, c2*c3).
    The left-multiplication matrix satisfies the generalized matrix identity
      M(x)^T Q M(x) = N(x) Q. *)

Definition dickson1921_param_form_entry (c2 c3 : R) (i j : nat) : R :=
  match i, j with
  | 0%nat, 0%nat => 1
  | 1%nat, 1%nat => - c2
  | 2%nat, 2%nat => - c3
  | 3%nat, 3%nat => c2 * c3
  | _, _ => 0
  end.

Definition dickson1921_param_matrix_entry (c2 c3 : R) (x : CDQuat) (i j : nat) : R :=
  match i, j with
  | 0%nat, 0%nat => qa x
  | 0%nat, 1%nat => c2 * qb x
  | 0%nat, 2%nat => c3 * qc x
  | 0%nat, 3%nat => - c2 * c3 * qd x
  | 1%nat, 0%nat => qb x
  | 1%nat, 1%nat => qa x
  | 1%nat, 2%nat => c3 * qd x
  | 1%nat, 3%nat => - c3 * qc x
  | 2%nat, 0%nat => qc x
  | 2%nat, 1%nat => - c2 * qd x
  | 2%nat, 2%nat => qa x
  | 2%nat, 3%nat => c2 * qb x
  | 3%nat, 0%nat => qd x
  | 3%nat, 1%nat => - qc x
  | 3%nat, 2%nat => qb x
  | 3%nat, 3%nat => qa x
  | _, _ => 0
  end.

Definition dickson1921_param_matrix_apply (c2 c3 : R) (x xi : CDQuat) : CDQuat :=
  mkQuat
    (dickson1921_param_matrix_entry c2 c3 x 0%nat 0%nat * quat_coord xi 0%nat +
     dickson1921_param_matrix_entry c2 c3 x 0%nat 1%nat * quat_coord xi 1%nat +
     dickson1921_param_matrix_entry c2 c3 x 0%nat 2%nat * quat_coord xi 2%nat +
     dickson1921_param_matrix_entry c2 c3 x 0%nat 3%nat * quat_coord xi 3%nat)
    (dickson1921_param_matrix_entry c2 c3 x 1%nat 0%nat * quat_coord xi 0%nat +
     dickson1921_param_matrix_entry c2 c3 x 1%nat 1%nat * quat_coord xi 1%nat +
     dickson1921_param_matrix_entry c2 c3 x 1%nat 2%nat * quat_coord xi 2%nat +
     dickson1921_param_matrix_entry c2 c3 x 1%nat 3%nat * quat_coord xi 3%nat)
    (dickson1921_param_matrix_entry c2 c3 x 2%nat 0%nat * quat_coord xi 0%nat +
     dickson1921_param_matrix_entry c2 c3 x 2%nat 1%nat * quat_coord xi 1%nat +
     dickson1921_param_matrix_entry c2 c3 x 2%nat 2%nat * quat_coord xi 2%nat +
     dickson1921_param_matrix_entry c2 c3 x 2%nat 3%nat * quat_coord xi 3%nat)
    (dickson1921_param_matrix_entry c2 c3 x 3%nat 0%nat * quat_coord xi 0%nat +
     dickson1921_param_matrix_entry c2 c3 x 3%nat 1%nat * quat_coord xi 1%nat +
     dickson1921_param_matrix_entry c2 c3 x 3%nat 2%nat * quat_coord xi 2%nat +
     dickson1921_param_matrix_entry c2 c3 x 3%nat 3%nat * quat_coord xi 3%nat).

Definition dickson1921_param_gram_entry (c2 c3 : R) (x : CDQuat) (i j : nat) : R :=
  dickson1921_param_form_entry c2 c3 0%nat 0%nat *
    dickson1921_param_matrix_entry c2 c3 x 0%nat i *
    dickson1921_param_matrix_entry c2 c3 x 0%nat j +
  dickson1921_param_form_entry c2 c3 1%nat 1%nat *
    dickson1921_param_matrix_entry c2 c3 x 1%nat i *
    dickson1921_param_matrix_entry c2 c3 x 1%nat j +
  dickson1921_param_form_entry c2 c3 2%nat 2%nat *
    dickson1921_param_matrix_entry c2 c3 x 2%nat i *
    dickson1921_param_matrix_entry c2 c3 x 2%nat j +
  dickson1921_param_form_entry c2 c3 3%nat 3%nat *
    dickson1921_param_matrix_entry c2 c3 x 3%nat i *
    dickson1921_param_matrix_entry c2 c3 x 3%nat j.

Definition dickson1921_imag_basis (k : nat) : CDQuat :=
  match k with
  | 1%nat => qi
  | 2%nat => qj
  | 3%nat => qk
  | _ => quat_zero
  end.

Definition dickson1921_near_identity (k : nat) (eps : R) : CDQuat :=
  quat_add quat_one (quat_scale eps (dickson1921_imag_basis k)).

Definition dickson1921_param_eq6_delta_coord
    (c2 c3 : R) (k : nat) (x : CDQuat) (i : nat) : R :=
  match k, i with
  | 1%nat, 0%nat => c2 * quat_coord x 1%nat
  | 1%nat, 1%nat => quat_coord x 0%nat
  | 1%nat, 2%nat => - c2 * quat_coord x 3%nat
  | 1%nat, 3%nat => - quat_coord x 2%nat
  | 2%nat, 0%nat => c3 * quat_coord x 2%nat
  | 2%nat, 1%nat => c3 * quat_coord x 3%nat
  | 2%nat, 2%nat => quat_coord x 0%nat
  | 2%nat, 3%nat => quat_coord x 1%nat
  | 3%nat, 0%nat => - c2 * c3 * quat_coord x 3%nat
  | 3%nat, 1%nat => - c3 * quat_coord x 2%nat
  | 3%nat, 2%nat => c2 * quat_coord x 1%nat
  | 3%nat, 3%nat => quat_coord x 0%nat
  | _, _ => 0
  end.

(** Dickson's equation (6), specialized to the dim-4 parameterized family and
    written as an explicit first-order coordinate increment law for the
    near-identity factor 1 + eps * e_k on the right. *)
Theorem dickson1921_param_eq6_dim4 : forall c2 c3 x k eps i,
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  quat_coord (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) i =
  quat_coord x i + eps * dickson1921_param_eq6_delta_coord c2 c3 k x i.
Proof.
  intros c2 c3 [a b c d] k eps i Hk Hi.
  destruct k as [|[|[|[|k]]]];
  destruct i as [|[|[|[|i]]]];
  simpl in Hk, Hi; try lia;
  unfold dickson1921_param_mul, dickson1921_near_identity,
         dickson1921_imag_basis, quat_add, quat_one, quat_scale,
         dickson1921_param_eq6_delta_coord, quat_coord, qi, qj, qk, quat_zero;
  simpl; ring.
Qed.

Definition dickson1921_param_gamma (c2 c3 : R) (i j k : nat) : R :=
  match i, j, k with
  | 1%nat, 2%nat, 3%nat => 1
  | 2%nat, 1%nat, 3%nat => -1
  | 1%nat, 3%nat, 2%nat => c2
  | 3%nat, 1%nat, 2%nat => - c2
  | 2%nat, 3%nat, 1%nat => - c3
  | 3%nat, 2%nat, 1%nat => c3
  | _, _, _ => 0
  end.

(** Canonical equation (7) vanishing constraints in the dim-4 parameterized
    lane: any structure constant with two equal lower indices vanishes. *)
Theorem dickson1921_param_eq7_repeated_index_vanish : forall c2 c3 i j,
  (1 <= i <= 3)%nat ->
  (1 <= j <= 3)%nat ->
  i <> j ->
  dickson1921_param_gamma c2 c3 j i i = 0 /\
  dickson1921_param_gamma c2 c3 i j i = 0 /\
  dickson1921_param_gamma c2 c3 i i j = 0.
Proof.
  intros c2 c3 i j Hi Hj Hneq.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj, Hneq; try lia; simpl in *; try contradiction; repeat split; reflexivity.
Qed.

Definition dickson1921_param_c (c2 c3 : R) (i : nat) : R :=
  match i with
  | 1%nat => c2
  | 2%nat => c3
  | 3%nat => - c2 * c3
  | _ => 0
  end.

Definition dickson1921_param_metric_weight (c2 c3 : R) (i : nat) : R :=
  match i with
  | 0%nat => 1
  | 1%nat => - c2
  | 2%nat => - c3
  | 3%nat => c2 * c3
  | _ => 0
  end.

Definition dickson1921_param_eq7_linear_form (c2 c3 : R) (k : nat) (x : CDQuat) : R :=
  dickson1921_param_metric_weight c2 c3 0%nat *
    quat_coord x 0%nat * dickson1921_param_eq6_delta_coord c2 c3 k x 0%nat +
  dickson1921_param_metric_weight c2 c3 1%nat *
    quat_coord x 1%nat * dickson1921_param_eq6_delta_coord c2 c3 k x 1%nat +
  dickson1921_param_metric_weight c2 c3 2%nat *
    quat_coord x 2%nat * dickson1921_param_eq6_delta_coord c2 c3 k x 2%nat +
  dickson1921_param_metric_weight c2 c3 3%nat *
    quat_coord x 3%nat * dickson1921_param_eq6_delta_coord c2 c3 k x 3%nat.

(** Paper equation (8) in the dim-4 parameterized lane. *)
Theorem dickson1921_param_eq8_distinct : forall c2 c3 i j k,
  (1 <= i <= 3)%nat ->
  (1 <= j <= 3)%nat ->
  (1 <= k <= 3)%nat ->
  i <> j ->
  i <> k ->
  j <> k ->
  dickson1921_param_c c2 c3 k * dickson1921_param_gamma c2 c3 i j k +
  dickson1921_param_c c2 c3 j * dickson1921_param_gamma c2 c3 i k j = 0.
Proof.
  intros c2 c3 i j k Hi Hj Hk Hij Hik Hjk.
  unfold dickson1921_param_c, dickson1921_param_gamma.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  destruct k as [|[|[|[|k]]]];
  simpl in Hi, Hj, Hk, Hij, Hik, Hjk; try lia; simpl in *; try contradiction; ring.
Qed.

Definition dickson1921_param_near_identity_quadratic_factor
    (c2 c3 : R) (k : nat) : R :=
  match k with
  | 1%nat => - c2
  | 2%nat => - c3
  | 3%nat => c2 * c3
  | _ => 0
  end.

Theorem dickson1921_param_near_identity_norm : forall c2 c3 k eps,
  (1 <= k <= 3)%nat ->
  dickson1921_param_norm c2 c3 (dickson1921_near_identity k eps) =
  1 + dickson1921_param_near_identity_quadratic_factor c2 c3 k * eps^2.
Proof.
  intros c2 c3 k eps Hk.
  destruct k as [|[|[|[|k]]]];
  simpl in Hk; try lia;
  unfold dickson1921_param_norm, dickson1921_near_identity,
         dickson1921_imag_basis, quat_add, quat_one, quat_scale,
         dickson1921_param_near_identity_quadratic_factor,
         qi, qj, qk, quat_zero;
  simpl; ring.
Qed.

Theorem dickson1921_param_infinitesimal_no_linear_term : forall c2 c3 k eps x,
  (1 <= k <= 3)%nat ->
  dickson1921_param_norm c2 c3
    (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) -
  dickson1921_param_norm c2 c3 x =
  eps^2 * dickson1921_param_near_identity_quadratic_factor c2 c3 k *
    dickson1921_param_norm c2 c3 x.
Proof.
  intros c2 c3 k eps x Hk.
  rewrite dickson1921_param_norm_mul.
  rewrite dickson1921_param_near_identity_norm by exact Hk.
  ring.
Qed.

(** Paper equation (7), derived from the infinitesimal no-linear-term law:
    the first-order metric pairing between x and its infinitesimal increment
    vanishes identically. *)
Theorem dickson1921_param_eq7_linear_form_vanish : forall c2 c3 k x,
  (1 <= k <= 3)%nat ->
  dickson1921_param_eq7_linear_form c2 c3 k x = 0.
Proof.
  intros c2 c3 k x Hk.
  pose proof (dickson1921_param_infinitesimal_no_linear_term c2 c3 k 1 x Hk) as H1.
  pose proof (dickson1921_param_infinitesimal_no_linear_term c2 c3 k (-1) x Hk) as Hm1.
  destruct x as [a b c d].
  destruct k as [|[|[|[|k]]]]; simpl in Hk; try lia.
  - unfold dickson1921_param_eq7_linear_form, dickson1921_param_metric_weight,
           dickson1921_param_eq6_delta_coord.
    simpl.
    assert (Hdiff :
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 1%nat 1)) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) -
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 1%nat (-1))) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) = 0).
    { rewrite H1, Hm1. ring. }
    unfold dickson1921_param_norm, dickson1921_param_mul, dickson1921_near_identity,
           dickson1921_imag_basis, quat_add, quat_one, quat_scale,
           qi, qj, qk, quat_zero in Hdiff.
    simpl in Hdiff.
    nra.
  - unfold dickson1921_param_eq7_linear_form, dickson1921_param_metric_weight,
           dickson1921_param_eq6_delta_coord.
    simpl.
    assert (Hdiff :
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 2%nat 1)) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) -
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 2%nat (-1))) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) = 0).
    { rewrite H1, Hm1. ring. }
    unfold dickson1921_param_norm, dickson1921_param_mul, dickson1921_near_identity,
           dickson1921_imag_basis, quat_add, quat_one, quat_scale,
           qi, qj, qk, quat_zero in Hdiff.
    simpl in Hdiff.
    nra.
  - unfold dickson1921_param_eq7_linear_form, dickson1921_param_metric_weight,
           dickson1921_param_eq6_delta_coord.
    simpl.
    assert (Hdiff :
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 3%nat 1)) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) -
      (dickson1921_param_norm c2 c3
         (dickson1921_param_mul c2 c3
            {| qa := a; qb := b; qc := c; qd := d |}
            (dickson1921_near_identity 3%nat (-1))) -
       dickson1921_param_norm c2 c3 {| qa := a; qb := b; qc := c; qd := d |}) = 0).
    { rewrite H1, Hm1. ring. }
    unfold dickson1921_param_norm, dickson1921_param_mul, dickson1921_near_identity,
           dickson1921_imag_basis, quat_add, quat_one, quat_scale,
           qi, qj, qk, quat_zero in Hdiff.
    simpl in Hdiff.
    nra.
Qed.

Definition quat_eps_entry (k : nat) (eps : R) (i j : nat) : R :=
  quat_identity_entry i j + eps * quat_hamilton_basis_entry k i j.

Definition quat_eps_gram_entry (k : nat) (eps : R) (i j : nat) : R :=
  quat_matrix_mul_entry (quat_matrix_transpose (quat_eps_entry k eps))
                        (quat_eps_entry k eps) i j.

Theorem quat_hamilton_apply_eq_mul : forall x xi : CDQuat,
  quat_hamilton_apply x xi = quat_mul x xi.
Proof.
  intros [a b c d] [e f g h].
  unfold quat_hamilton_apply, quat_hamilton_entry, quat_coord, quat_mul.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_matrix_apply_eq_mul : forall c2 c3 x xi,
  dickson1921_param_matrix_apply c2 c3 x xi =
  dickson1921_param_mul c2 c3 x xi.
Proof.
  intros c2 c3 [a b c d] [e f g h].
  unfold dickson1921_param_matrix_apply, dickson1921_param_matrix_entry,
         dickson1921_param_mul, quat_coord.
  simpl.
  f_equal; ring.
Qed.

Theorem dickson1921_param_eq13_dim4 : forall c2 c3 x xi,
  dickson1921_param_norm c2 c3 (dickson1921_param_matrix_apply c2 c3 x xi) =
  dickson1921_param_norm c2 c3 x * dickson1921_param_norm c2 c3 xi.
Proof.
  intros c2 c3 x xi.
  rewrite dickson1921_param_matrix_apply_eq_mul.
  exact (dickson1921_param_norm_mul c2 c3 x xi).
Qed.

Theorem dickson1921_param_eq15_dim4 : forall c2 c3 x (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  dickson1921_param_gram_entry c2 c3 x i j =
  dickson1921_param_norm c2 c3 x * dickson1921_param_form_entry c2 c3 i j.
Proof.
  intros c2 c3 [a b c d] i j Hi Hj.
  unfold dickson1921_param_gram_entry, dickson1921_param_form_entry,
         dickson1921_param_matrix_entry, dickson1921_param_norm.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Definition dickson1921_basis_quat (k : nat) : CDQuat :=
  match k with
  | 0%nat => quat_one
  | 1%nat => qi
  | 2%nat => qj
  | 3%nat => qk
  | _ => quat_zero
  end.

Definition dickson1921_param_basis_entry (c2 c3 : R) (k i j : nat) : R :=
  dickson1921_param_matrix_entry c2 c3 (dickson1921_basis_quat k) i j.

Definition dickson1921_param_basis_square_scalar (c2 c3 : R) (k : nat) : R :=
  match k with
  | 1%nat => c2
  | 2%nat => c3
  | 3%nat => - c2 * c3
  | _ => 0
  end.

Theorem dickson1921_param_matrix_basis_expansion : forall c2 c3 x i j,
  dickson1921_param_matrix_entry c2 c3 x i j =
  qa x * dickson1921_param_basis_entry c2 c3 0%nat i j +
  qb x * dickson1921_param_basis_entry c2 c3 1%nat i j +
  qc x * dickson1921_param_basis_entry c2 c3 2%nat i j +
  qd x * dickson1921_param_basis_entry c2 c3 3%nat i j.
Proof.
  intros c2 c3 [a b c d] i j.
  unfold dickson1921_param_matrix_entry, dickson1921_param_basis_entry,
         dickson1921_basis_quat, quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl; ring.
Qed.

Theorem dickson1921_param_basis_metric_skew : forall c2 c3 (k i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  dickson1921_param_form_entry c2 c3 j j *
    dickson1921_param_basis_entry c2 c3 k j i =
  - (dickson1921_param_form_entry c2 c3 i i *
     dickson1921_param_basis_entry c2 c3 k i j).
Proof.
  intros c2 c3 k i j Hk Hi Hj.
  unfold dickson1921_param_form_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct k as [|[|[|[|k]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_square : forall c2 c3 (k i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 k)
                        (dickson1921_param_basis_entry c2 c3 k) i j =
  dickson1921_param_basis_square_scalar c2 c3 k * quat_identity_entry i j.
Proof.
  intros c2 c3 k i j Hk Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_basis_square_scalar, dickson1921_param_matrix_entry,
         dickson1921_basis_quat, quat_one, qi, qj, qk, quat_zero,
         quat_identity_entry.
  destruct k as [|[|[|[|k]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_anticommute : forall c2 c3 (k l i j : nat),
  (1 <= k <= 3)%nat ->
  (1 <= l <= 3)%nat ->
  k <> l ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 k)
                        (dickson1921_param_basis_entry c2 c3 l) i j +
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 l)
                        (dickson1921_param_basis_entry c2 c3 k) i j = 0.
Proof.
  intros c2 c3 k l i j Hk Hl Hneq Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct k as [|[|[|[|k]]]];
  destruct l as [|[|[|[|l]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hl, Hneq, Hi, Hj; try lia; simpl in *; try contradiction; ring.
Qed.

Theorem dickson1921_param_basis_12 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 1%nat)
                        (dickson1921_param_basis_entry c2 c3 2%nat) i j =
  dickson1921_param_basis_entry c2 c3 3%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_21 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 2%nat)
                        (dickson1921_param_basis_entry c2 c3 1%nat) i j =
  - dickson1921_param_basis_entry c2 c3 3%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_13 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 1%nat)
                        (dickson1921_param_basis_entry c2 c3 3%nat) i j =
  c2 * dickson1921_param_basis_entry c2 c3 2%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_31 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 3%nat)
                        (dickson1921_param_basis_entry c2 c3 1%nat) i j =
  - c2 * dickson1921_param_basis_entry c2 c3 2%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_23 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 2%nat)
                        (dickson1921_param_basis_entry c2 c3 3%nat) i j =
  - c3 * dickson1921_param_basis_entry c2 c3 1%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_param_basis_32 : forall c2 c3 (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 3%nat)
                        (dickson1921_param_basis_entry c2 c3 2%nat) i j =
  c3 * dickson1921_param_basis_entry c2 c3 1%nat i j.
Proof.
  intros c2 c3 i j Hi Hj.
  unfold quat_matrix_mul_entry, dickson1921_param_basis_entry,
         dickson1921_param_matrix_entry, dickson1921_basis_quat,
         quat_one, qi, qj, qk, quat_zero.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

(** Sanity-check that the gamma notation agrees with the already-landed
    generalized basis-generator multiplication table. *)
Theorem dickson1921_param_gamma_basis_product : forall c2 c3 i j k m n,
  (1 <= i <= 3)%nat ->
  (1 <= j <= 3)%nat ->
  (1 <= k <= 3)%nat ->
  i <> j ->
  i <> k ->
  j <> k ->
  (m < 4)%nat ->
  (n < 4)%nat ->
  quat_matrix_mul_entry (dickson1921_param_basis_entry c2 c3 i)
                        (dickson1921_param_basis_entry c2 c3 j) m n =
  dickson1921_param_gamma c2 c3 i j k *
    dickson1921_param_basis_entry c2 c3 k m n.
Proof.
  intros c2 c3 i j k m n Hi Hj Hk Hij Hik Hjk Hm Hn.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  destruct k as [|[|[|[|k]]]];
  simpl in Hi, Hj, Hk, Hij, Hik, Hjk; try lia; simpl in *; try contradiction.
  - rewrite dickson1921_param_basis_12 by assumption.
    ring.
  - rewrite dickson1921_param_basis_13 by assumption.
    ring.
  - rewrite dickson1921_param_basis_21 by assumption.
    ring.
  - rewrite dickson1921_param_basis_23 by assumption.
    ring.
  - rewrite dickson1921_param_basis_31 by assumption.
    ring.
  - rewrite dickson1921_param_basis_32 by assumption.
    ring.
Qed.

(** Printed equation (10) on p.112: in the dim-4 lane the relevant sign
    relation is the skew swap on the first two lower indices when the third
    index is repeated.  Here it is a formal corollary of the equation (7)
    vanishing theorem. *)
Theorem dickson1921_param_eq10_sign_relation : forall c2 c3 i j,
  (1 <= i <= 3)%nat ->
  (1 <= j <= 3)%nat ->
  i <> j ->
  dickson1921_param_gamma c2 c3 j i i =
  - dickson1921_param_gamma c2 c3 i j i.
Proof.
  intros c2 c3 i j Hi Hj Hneq.
  destruct (dickson1921_param_eq7_repeated_index_vanish c2 c3 i j Hi Hj Hneq)
    as [_ [Hij0 _]].
  destruct (dickson1921_param_eq7_repeated_index_vanish c2 c3 i j Hi Hj Hneq)
    as [Hji0 _].
  rewrite Hji0, Hij0.
  ring.
Qed.

(** Printed equation (11), specialized to the repo's dim-4 parameterized
    family.  This is the same concrete transformation law already proved as
    equation (5), now re-surfaced in the paper's n = 4 specialization slot. *)
Theorem dickson1921_param_eq11_dim4 : forall c2 c3 x xi,
  quat_coord (dickson1921_param_mul c2 c3 x xi) 0%nat =
    quat_coord x 0%nat * quat_coord xi 0%nat +
    c2 * quat_coord x 1%nat * quat_coord xi 1%nat +
    c3 * quat_coord x 2%nat * quat_coord xi 2%nat -
    c2 * c3 * quat_coord x 3%nat * quat_coord xi 3%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 1%nat =
    quat_coord x 0%nat * quat_coord xi 1%nat +
    quat_coord x 1%nat * quat_coord xi 0%nat -
    c3 * quat_coord x 2%nat * quat_coord xi 3%nat +
    c3 * quat_coord x 3%nat * quat_coord xi 2%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 2%nat =
    quat_coord x 0%nat * quat_coord xi 2%nat +
    quat_coord x 2%nat * quat_coord xi 0%nat +
    c2 * quat_coord x 1%nat * quat_coord xi 3%nat -
    c2 * quat_coord x 3%nat * quat_coord xi 1%nat /\
  quat_coord (dickson1921_param_mul c2 c3 x xi) 3%nat =
    quat_coord x 0%nat * quat_coord xi 3%nat +
    quat_coord x 3%nat * quat_coord xi 0%nat +
    quat_coord x 1%nat * quat_coord xi 2%nat -
    quat_coord x 2%nat * quat_coord xi 1%nat.
Proof.
  exact dickson1921_param_eq5_dim4.
Qed.

(** Bridge from the paper's n = 4 specialization back to the explicit unit
    multiplication facts already landed for the parameterized family. *)
Theorem dickson1921_param_eq11_unit_table : forall c2 c3,
  dickson1921_param_mul c2 c3 qi qi = quat_scale c2 quat_one /\
  dickson1921_param_mul c2 c3 qj qj = quat_scale c3 quat_one /\
  dickson1921_param_mul c2 c3 qk qk = quat_scale (- c2 * c3) quat_one /\
  dickson1921_param_mul c2 c3 qi qj = qk /\
  dickson1921_param_mul c2 c3 qj qi = quat_neg qk /\
  dickson1921_param_mul c2 c3 qi qk = quat_scale c2 qj /\
  dickson1921_param_mul c2 c3 qk qi = quat_scale (- c2) qj /\
  dickson1921_param_mul c2 c3 qj qk = quat_scale (- c3) qi /\
  dickson1921_param_mul c2 c3 qk qj = quat_scale c3 qi.
Proof.
  intros c2 c3.
  repeat split;
    [ exact (dickson1921_param_i_sq c2 c3)
    | exact (dickson1921_param_j_sq c2 c3)
    | exact (dickson1921_param_k_sq c2 c3)
    | exact (dickson1921_param_ij c2 c3)
    | exact (dickson1921_param_ji c2 c3)
    | exact (dickson1921_param_ik c2 c3)
    | exact (dickson1921_param_ki c2 c3)
    | exact (dickson1921_param_jk c2 c3)
    | exact (dickson1921_param_kj c2 c3) ].
Qed.

Definition dickson1921_matrix_det4 (m : nat -> nat -> R) : R :=
  m 0%nat 0%nat *
    (m 1%nat 1%nat * (m 2%nat 2%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 2%nat) -
     m 1%nat 2%nat * (m 2%nat 1%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 1%nat) +
     m 1%nat 3%nat * (m 2%nat 1%nat * m 3%nat 2%nat - m 2%nat 2%nat * m 3%nat 1%nat)) -
  m 0%nat 1%nat *
    (m 1%nat 0%nat * (m 2%nat 2%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 2%nat) -
     m 1%nat 2%nat * (m 2%nat 0%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 0%nat) +
     m 1%nat 3%nat * (m 2%nat 0%nat * m 3%nat 2%nat - m 2%nat 2%nat * m 3%nat 0%nat)) +
  m 0%nat 2%nat *
    (m 1%nat 0%nat * (m 2%nat 1%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 1%nat) -
     m 1%nat 1%nat * (m 2%nat 0%nat * m 3%nat 3%nat - m 2%nat 3%nat * m 3%nat 0%nat) +
     m 1%nat 3%nat * (m 2%nat 0%nat * m 3%nat 1%nat - m 2%nat 1%nat * m 3%nat 0%nat)) -
  m 0%nat 3%nat *
    (m 1%nat 0%nat * (m 2%nat 1%nat * m 3%nat 2%nat - m 2%nat 2%nat * m 3%nat 1%nat) -
     m 1%nat 1%nat * (m 2%nat 0%nat * m 3%nat 2%nat - m 2%nat 2%nat * m 3%nat 0%nat) +
     m 1%nat 2%nat * (m 2%nat 0%nat * m 3%nat 1%nat - m 2%nat 1%nat * m 3%nat 0%nat)).

(** Dickson's section-6 determinant note: the four-rowed determinant of the
    general number equals o(x)^2. *)
Theorem dickson1921_param_det_equals_norm_sq : forall c2 c3 x,
  dickson1921_matrix_det4 (dickson1921_param_matrix_entry c2 c3 x) =
  (dickson1921_param_norm c2 c3 x)^2.
Proof.
  intros c2 c3 [a b c d].
  unfold dickson1921_matrix_det4, dickson1921_param_matrix_entry,
         dickson1921_param_norm.
  simpl.
  ring.
Qed.

(** Printed equation (12), surfaced in paper order. *)
Theorem dickson1921_param_eq12_table : forall c2 c3,
  dickson1921_param_mul c2 c3 qi qi = quat_scale c2 quat_one /\
  dickson1921_param_mul c2 c3 qj qj = quat_scale c3 quat_one /\
  dickson1921_param_mul c2 c3 qk qk = quat_scale (- c2 * c3) quat_one /\
  dickson1921_param_mul c2 c3 qi qj = qk /\
  dickson1921_param_mul c2 c3 qj qi = quat_neg qk /\
  dickson1921_param_mul c2 c3 qi qk = quat_scale c2 qj /\
  dickson1921_param_mul c2 c3 qk qi = quat_scale (- c2) qj /\
  dickson1921_param_mul c2 c3 qj qk = quat_scale (- c3) qi /\
  dickson1921_param_mul c2 c3 qk qj = quat_scale c3 qi.
Proof.
  exact dickson1921_param_eq11_unit_table.
Qed.

(** Dickson's section-6 conclusion packages the associative generalized
    quaternion family obtained from the printed table (12). *)
Theorem dickson1921_param_direct_generalization_summary : forall c2 c3,
  (exists s : Dickson1921Surface CDQuat, True) /\
  (forall x y z,
      dickson1921_param_mul c2 c3 (dickson1921_param_mul c2 c3 x y) z =
      dickson1921_param_mul c2 c3 x (dickson1921_param_mul c2 c3 y z)) /\
  (forall x,
      dickson1921_matrix_det4 (dickson1921_param_matrix_entry c2 c3 x) =
      (dickson1921_param_norm c2 c3 x)^2) /\
  (forall x k eps i,
      (1 <= k <= 3)%nat ->
      (i < 4)%nat ->
      quat_coord (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) i =
      quat_coord x i + eps * dickson1921_param_eq6_delta_coord c2 c3 k x i) /\
  (forall k eps,
      (1 <= k <= 3)%nat ->
      dickson1921_param_norm c2 c3 (dickson1921_near_identity k eps) =
      1 + dickson1921_param_near_identity_quadratic_factor c2 c3 k * eps^2) /\
  (forall x k eps,
      (1 <= k <= 3)%nat ->
      dickson1921_param_norm c2 c3
        (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) -
      dickson1921_param_norm c2 c3 x =
      eps^2 * dickson1921_param_near_identity_quadratic_factor c2 c3 k *
        dickson1921_param_norm c2 c3 x) /\
  (forall x k,
      (1 <= k <= 3)%nat ->
      dickson1921_param_eq7_linear_form c2 c3 k x = 0) /\
  dickson1921_param_mul c2 c3 qi qi = quat_scale c2 quat_one /\
  dickson1921_param_mul c2 c3 qj qj = quat_scale c3 quat_one /\
  dickson1921_param_mul c2 c3 qk qk = quat_scale (- c2 * c3) quat_one /\
  dickson1921_param_mul c2 c3 qi qj = qk /\
  dickson1921_param_mul c2 c3 qj qi = quat_neg qk /\
  dickson1921_param_mul c2 c3 qi qk = quat_scale c2 qj /\
  dickson1921_param_mul c2 c3 qk qi = quat_scale (- c2) qj /\
  dickson1921_param_mul c2 c3 qj qk = quat_scale (- c3) qi /\
  dickson1921_param_mul c2 c3 qk qj = quat_scale c3 qi.
Proof.
  intros c2 c3.
  split.
  - exists (dickson1921_param_surface c2 c3).
    exact I.
  - repeat split.
    + exact (dickson1921_param_mul_assoc c2 c3).
    + exact (dickson1921_param_det_equals_norm_sq c2 c3).
    + exact (dickson1921_param_eq6_dim4 c2 c3).
    + exact (dickson1921_param_near_identity_norm c2 c3).
    + intros x k eps Hk.
      exact (dickson1921_param_infinitesimal_no_linear_term c2 c3 k eps x Hk).
    + intros x k Hk.
      exact (dickson1921_param_eq7_linear_form_vanish c2 c3 k x Hk).
    + exact (dickson1921_param_i_sq c2 c3).
    + exact (dickson1921_param_j_sq c2 c3).
    + exact (dickson1921_param_k_sq c2 c3).
    + exact (dickson1921_param_ij c2 c3).
    + exact (dickson1921_param_ji c2 c3).
    + exact (dickson1921_param_ik c2 c3).
    + exact (dickson1921_param_ki c2 c3).
    + exact (dickson1921_param_jk c2 c3).
    + exact (dickson1921_param_kj c2 c3).
Qed.

(** The customary quaternion algebra is recovered when c2 = c3 = -1. *)
Theorem dickson1921_param_quaternion_case : forall x,
  dickson1921_matrix_det4 (dickson1921_param_matrix_entry (-1) (-1) x) =
  (quat_norm_sq x)^2 /\
  dickson1921_param_mul (-1) (-1) qi qj = qk /\
  dickson1921_param_mul (-1) (-1) qj qk = qi /\
  dickson1921_param_mul (-1) (-1) qk qi = qj.
Proof.
  intro x.
  repeat split.
  - rewrite dickson1921_param_det_equals_norm_sq.
    rewrite dickson1921_param_specializes_to_standard_norm.
    reflexivity.
  - exact (dickson1921_param_ij (-1) (-1)).
  - rewrite dickson1921_param_jk.
    unfold quat_scale, qi.
    simpl.
    f_equal; ring.
  - rewrite dickson1921_param_ki.
    unfold quat_scale, qj.
    simpl.
    f_equal; ring.
Qed.

Theorem dickson1921_param_matrix_specializes_to_standard : forall x i j,
  dickson1921_param_matrix_entry (-1) (-1) x i j = quat_hamilton_entry x i j.
Proof.
  intros [a b c d] i j.
  unfold dickson1921_param_matrix_entry, quat_hamilton_entry, quat_hamilton_basis_entry.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl; ring.
Qed.

Theorem dickson1921_param_form_specializes_to_identity : forall (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  dickson1921_param_form_entry (-1) (-1) i j = quat_identity_entry i j.
Proof.
  intros i j Hi Hj.
  unfold dickson1921_param_form_entry, quat_identity_entry.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; try reflexivity; ring.
Qed.

(** Dickson's eq.(13) in the canonical n = 4 lane. *)
Theorem dickson1921_eq13_dim4 : forall x xi : CDQuat,
  quat_norm_sq (quat_hamilton_apply x xi) =
  quat_norm_sq x * quat_norm_sq xi.
Proof.
  intros x xi.
  rewrite quat_hamilton_apply_eq_mul.
  exact (quat_norm_mul x xi).
Qed.

(** Dickson's eq.(15): M(x)^T * M(x) = o(x) * I_4. *)
Theorem dickson1921_eq15_dim4 : forall x (i j : nat),
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_hamilton_gram_entry x i j =
  quat_norm_sq x * quat_identity_entry i j.
Proof.
  intros [a b c d] i j Hi Hj.
  unfold quat_hamilton_gram_entry, quat_matrix_mul_entry, quat_matrix_transpose,
         quat_hamilton_entry, quat_hamilton_basis_entry, quat_identity_entry,
         quat_norm_sq.
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hi, Hj; try lia; simpl; ring.
Qed.

(** The coefficient matrices in eq.(16) are skew and satisfy the Hamilton
    square / anticommutation relations in the canonical lane. *)
Theorem dickson1921_eq16_basis_skew : forall (k i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_transpose (quat_hamilton_basis_entry k) i j =
  - quat_hamilton_basis_entry k i j.
Proof.
  intros k i j Hk Hi Hj.
  unfold quat_matrix_transpose, quat_hamilton_basis_entry.
  destruct k as [|[|[|[|k]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_eq16_basis_square : forall (k i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (quat_hamilton_basis_entry k)
                        (quat_hamilton_basis_entry k) i j =
  - quat_identity_entry i j.
Proof.
  intros k i j Hk Hi Hj.
  unfold quat_matrix_mul_entry, quat_hamilton_basis_entry, quat_identity_entry.
  destruct k as [|[|[|[|k]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hi, Hj; try lia; simpl; ring.
Qed.

Theorem dickson1921_eq16_basis_anticommute : forall (k l i j : nat),
  (1 <= k <= 3)%nat ->
  (1 <= l <= 3)%nat ->
  k <> l ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_matrix_mul_entry (quat_hamilton_basis_entry k)
                        (quat_hamilton_basis_entry l) i j +
  quat_matrix_mul_entry (quat_hamilton_basis_entry l)
                        (quat_hamilton_basis_entry k) i j = 0.
Proof.
  intros k l i j Hk Hl Hneq Hi Hj.
  unfold quat_matrix_mul_entry, quat_hamilton_basis_entry.
  destruct k as [|[|[|[|k]]]];
  destruct l as [|[|[|[|l]]]];
  destruct i as [|[|[|[|i]]]];
  destruct j as [|[|[|[|j]]]];
  simpl in Hk, Hl, Hi, Hj, Hneq; try lia; simpl in *; try contradiction; ring.
Qed.

(** ================================================================== *)
(** * Canonical infinitesimal-transform lane (paper sections 5-6).     *)
(** ================================================================== *)

(** Dickson's infinitesimal step starts from a near-identity multiplier xi
    whose norm is 1 up to second order.  In the canonical quaternion lane this
    is exactly the family 1 + eps * e_k for k in {1,2,3}. *)

Theorem dickson1921_imag_basis_norm_one : forall k,
  (1 <= k <= 3)%nat ->
  quat_norm_sq (dickson1921_imag_basis k) = 1.
Proof.
  intros k Hk.
  destruct k as [| [| [| [| k]]]]; simpl in Hk; try lia;
  unfold dickson1921_imag_basis, qi, qj, qk, quat_norm_sq; simpl; ring.
Qed.

Theorem dickson1921_near_identity_norm : forall k eps,
  (1 <= k <= 3)%nat ->
  quat_norm_sq (dickson1921_near_identity k eps) = 1 + eps^2.
Proof.
  intros k eps Hk.
  destruct k as [| [| [| [| k]]]]; simpl in Hk; try lia;
  unfold dickson1921_near_identity, dickson1921_imag_basis,
         qi, qj, qk, quat_add, quat_one, quat_scale, quat_norm_sq;
  simpl; ring.
Qed.

Theorem dickson1921_near_identity_matrix : forall k eps (i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_hamilton_entry (dickson1921_near_identity k eps) i j =
  quat_eps_entry k eps i j.
Proof.
  intros k eps i j Hk Hi Hj.
  destruct k as [| [| [| [| k]]]]; simpl in Hk; try lia;
  destruct i as [| [| [| [| i]]]];
  destruct j as [| [| [| [| j]]]];
  simpl in Hi, Hj; try lia;
  unfold dickson1921_near_identity, dickson1921_imag_basis,
         qi, qj, qk, quat_zero, quat_add, quat_one, quat_scale,
         quat_hamilton_entry, quat_eps_entry, quat_identity_entry,
         quat_hamilton_basis_entry;
  simpl; ring.
Qed.

(** Exact quaternion form of the paper's infinitesimal orthogonality claim:
    (I + eps * B_k)^T (I + eps * B_k) = (1 + eps^2) * I. *)
Theorem dickson1921_infinitesimal_gram : forall k eps (i j : nat),
  (1 <= k <= 3)%nat ->
  (i < 4)%nat ->
  (j < 4)%nat ->
  quat_eps_gram_entry k eps i j =
  (1 + eps^2) * quat_identity_entry i j.
Proof.
  intros k eps i j Hk Hi Hj.
  destruct k as [| [| [| [| k]]]];
  destruct i as [| [| [| [| i]]]];
  destruct j as [| [| [| [| j]]]];
  simpl in Hk, Hi, Hj; try lia;
  unfold quat_eps_gram_entry, quat_matrix_mul_entry, quat_matrix_transpose,
         quat_eps_entry, quat_identity_entry, quat_hamilton_basis_entry;
  simpl; ring.
Qed.

(** Hence left multiplication by 1 + eps * e_k scales the norm by exactly
    1 + eps^2, so the linear term vanishes. *)
Theorem dickson1921_infinitesimal_norm_exact : forall k eps x,
  (1 <= k <= 3)%nat ->
  quat_norm_sq (quat_hamilton_apply (dickson1921_near_identity k eps) x) =
  (1 + eps^2) * quat_norm_sq x.
Proof.
  intros k eps x Hk.
  rewrite dickson1921_eq13_dim4.
  rewrite dickson1921_near_identity_norm by exact Hk.
  ring.
Qed.

Theorem dickson1921_infinitesimal_no_linear_term : forall k eps x,
  (1 <= k <= 3)%nat ->
  quat_norm_sq (quat_hamilton_apply (dickson1921_near_identity k eps) x) -
  quat_norm_sq x =
  eps^2 * quat_norm_sq x.
Proof.
  intros k eps x Hk.
  rewrite dickson1921_infinitesimal_norm_exact by exact Hk.
  ring.
Qed.

(** Dickson remarks that the cases n = 5 and n = 6 are excluded before
    handing off to Hurwitz's general theorem.  We surface those two internal
    obstruction anchors explicitly in the current real-number formalization. *)

Theorem dickson1921_n5_det_obstruction :
  ~(exists r : R, r * r = (-1 : R)^5).
Proof.
  apply hurwitz_odd_excluded_formal.
  exact odd_5.
Qed.

Theorem dickson1921_n6_skew_obstruction :
  (skew_product_count 6 > skew_sym_dim 6)%nat.
Proof.
  exact n6_eliminated.
Qed.

Theorem dickson1921_small_internal_exclusions :
  ~(exists r : R, r * r = (-1 : R)^5) /\
  (skew_product_count 6 > skew_sym_dim 6)%nat.
Proof.
  split.
  - exact dickson1921_n5_det_obstruction.
  - exact dickson1921_n6_skew_obstruction.
Qed.

(** After Dickson's internal n = 5 / n = 6 exclusions, the remaining tracked
    higher Cayley-Dickson tower dimensions are ruled out by the Hurwitz lane. *)
Theorem dickson1921_hurwitz_handoff : forall n : nat,
  tracked_cd_tower_dimension n ->
  ~ hurwitz_square_dimension n ->
  ~(hurwitz_radon n = n).
Proof.
  intros n Htracked Hnonsquare Heq.
  pose proof (hurwitz_cd_tower_classification n Htracked) as Hclass.
  apply Hnonsquare.
  apply Hclass.
  exact Heq.
Qed.

(** Paper-order exclusion summary for the current repo formalization:
    Dickson excludes n = 5 and n = 6 internally, and the tracked higher
    dimensions then fall under Hurwitz's classification. *)
Theorem dickson1921_section7_exclusion_summary :
  ~(exists r : R, r * r = (-1 : R)^5) /\
  (skew_product_count 6 > skew_sym_dim 6)%nat /\
  (forall n : nat,
      tracked_cd_tower_dimension n ->
      ~ hurwitz_square_dimension n ->
      ~(hurwitz_radon n = n)).
Proof.
  repeat split.
  - exact dickson1921_n5_det_obstruction.
  - exact dickson1921_n6_skew_obstruction.
  - exact dickson1921_hurwitz_handoff.
Qed.

Definition dickson1921_section7_dimension (n : nat) : Prop :=
  n = 5%nat \/ n = 6%nat \/ tracked_cd_tower_dimension n.

Theorem dickson1921_section7_square_case_classification : forall n : nat,
  dickson1921_section7_dimension n ->
  (hurwitz_radon n = n <-> hurwitz_square_dimension n).
Proof.
  intros n Hn.
  destruct Hn as [-> | [-> | Htracked]].
  - simpl. split.
    + intro H. lia.
    + intro H.
      destruct H as [H1 | [H2 | [H4 | H8]]]; lia.
  - simpl. split.
    + intro H. lia.
    + intro H.
      destruct H as [H1 | [H2 | [H4 | H8]]]; lia.
  - exact (hurwitz_cd_tower_classification n Htracked).
Qed.

Theorem dickson1921_section7_noncomposition_summary : forall n : nat,
  dickson1921_section7_dimension n ->
  ~ hurwitz_square_dimension n ->
  hurwitz_radon n <> n.
Proof.
  intros n Hn Hnonsquare Heq.
  apply Hnonsquare.
  apply (proj1 (dickson1921_section7_square_case_classification n Hn)).
  exact Heq.
Qed.

(** Paper-order package: the generalized-family lane supplies the section-5/6
    infinitesimal foundation, and section 7 then reduces the surviving square
    cases to the same dimension classification already tracked via Hurwitz. *)
Theorem dickson1921_param_foundation_section7_summary :
  (forall c2 c3,
      (exists s : Dickson1921Surface CDQuat, True) /\
      (forall x y z,
          dickson1921_param_mul c2 c3 (dickson1921_param_mul c2 c3 x y) z =
          dickson1921_param_mul c2 c3 x (dickson1921_param_mul c2 c3 y z)) /\
      (forall x,
          dickson1921_matrix_det4 (dickson1921_param_matrix_entry c2 c3 x) =
          (dickson1921_param_norm c2 c3 x)^2) /\
      (forall x k eps i,
          (1 <= k <= 3)%nat ->
          (i < 4)%nat ->
          quat_coord (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) i =
          quat_coord x i + eps * dickson1921_param_eq6_delta_coord c2 c3 k x i) /\
      (forall k eps,
          (1 <= k <= 3)%nat ->
          dickson1921_param_norm c2 c3 (dickson1921_near_identity k eps) =
          1 + dickson1921_param_near_identity_quadratic_factor c2 c3 k * eps^2) /\
      (forall x k eps,
          (1 <= k <= 3)%nat ->
          dickson1921_param_norm c2 c3
            (dickson1921_param_mul c2 c3 x (dickson1921_near_identity k eps)) -
          dickson1921_param_norm c2 c3 x =
          eps^2 * dickson1921_param_near_identity_quadratic_factor c2 c3 k *
            dickson1921_param_norm c2 c3 x) /\
      (forall x k,
          (1 <= k <= 3)%nat ->
          dickson1921_param_eq7_linear_form c2 c3 k x = 0)) /\
  (forall n : nat,
      dickson1921_section7_dimension n ->
      (hurwitz_radon n = n <-> hurwitz_square_dimension n)) /\
  (forall n : nat,
      dickson1921_section7_dimension n ->
      ~ hurwitz_square_dimension n ->
      hurwitz_radon n <> n).
Proof.
  split.
  - intros a2 a3.
    destruct (dickson1921_param_direct_generalization_summary a2 a3)
      as [Hsurf [Hassoc [Hdet [Heq6 [Hnear [Hinf [Heq7 _]]]]]]].
    split.
    + exact Hsurf.
    + split.
      * exact Hassoc.
      * split.
        -- exact Hdet.
        -- split.
           ++ exact Heq6.
           ++ split.
              ** exact Hnear.
              ** split.
                 --- exact Hinf.
                 --- exact Heq7.
  - split.
    + exact dickson1921_section7_square_case_classification.
    + exact dickson1921_section7_noncomposition_summary.
Qed.

Module Dickson1921FloatSection7Lane (F : FLOAT_OPS).
  Import F.
  Module P := Dickson1921FloatParamLane F.

  Definition d1921_float_param_section5_6_foundation (c2 c3 : t) : Prop :=
    (forall x y,
        P.Q.quat_param_norm c2 c3 (P.Q.quat_param_mul c2 c3 x y) =
        P.Q.quat_param_norm c2 c3 x * P.Q.quat_param_norm c2 c3 y) /\
    (forall x xi,
        P.Q.quat_param_matrix_apply c2 c3 x xi =
        P.Q.quat_param_mul c2 c3 x xi) /\
    (forall x i j,
        (i < 4)%nat ->
        (j < 4)%nat ->
        P.Q.quat_param_gram_entry c2 c3 x i j =
        P.Q.quat_param_norm c2 c3 x * P.Q.quat_param_form_entry c2 c3 i j) /\
    (forall x k eps i,
        (1 <= k <= 3)%nat ->
        (i < 4)%nat ->
        P.Q.quat_coord (P.Q.quat_param_mul c2 c3 x (P.Q.quat_near_identity k eps)) i =
        P.Q.quat_coord x i + eps * P.Q.quat_param_eq6_delta_coord c2 c3 k x i) /\
    (forall k eps,
        (1 <= k <= 3)%nat ->
        P.Q.quat_param_norm c2 c3 (P.Q.quat_near_identity k eps) =
        one + P.Q.quat_param_near_identity_quadratic_factor c2 c3 k * (eps * eps)) /\
    (forall x k eps,
        (1 <= k <= 3)%nat ->
        P.Q.quat_param_norm c2 c3
          (P.Q.quat_param_mul c2 c3 x (P.Q.quat_near_identity k eps)) -
        P.Q.quat_param_norm c2 c3 x =
        (eps * eps) * P.Q.quat_param_near_identity_quadratic_factor c2 c3 k *
          P.Q.quat_param_norm c2 c3 x) /\
    (forall x k,
        (1 <= k <= 3)%nat ->
        P.Q.quat_param_eq7_linear_form c2 c3 k x = zero).

  Theorem d1921_float_param_foundation_section7_summary :
    (forall c2 c3 : t, d1921_float_param_section5_6_foundation c2 c3) /\
    (forall n : nat,
        dickson1921_section7_dimension n ->
        (hurwitz_radon n = n <-> hurwitz_square_dimension n)) /\
    (forall n : nat,
        dickson1921_section7_dimension n ->
        ~ hurwitz_square_dimension n ->
        hurwitz_radon n <> n).
  Proof.
    split.
    - intros c2 c3.
      unfold d1921_float_param_section5_6_foundation.
      destruct (P.d1921_float_param_norm_summary c2 c3)
        as [_ [_ [_ [_ [_ Hnorm]]]]].
      destruct (P.d1921_float_param_matrix_summary c2 c3)
        as [Hmat [_ Hgram]].
      destruct (P.d1921_float_param_infinitesimal_summary c2 c3)
        as [Heq6 [Hnear [Hinf Heq7]]].
      repeat split.
      + exact Hnorm.
      + exact Hmat.
      + exact Hgram.
      + exact Heq6.
      + exact Hnear.
      + exact Hinf.
      + exact Heq7.
    - split.
      + exact dickson1921_section7_square_case_classification.
      + exact dickson1921_section7_noncomposition_summary.
  Qed.
End Dickson1921FloatSection7Lane.

Theorem Dickson1921_lane_compiles : True.
Proof. exact I. Qed.
