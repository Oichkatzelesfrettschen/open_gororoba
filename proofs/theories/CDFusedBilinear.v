From Stdlib Require Import Reals Bool Arith ZArith Lia.
From OpenGororoba Require Import
  Prelude
  CayleyDicksonAlgebra
  Sedenion
  OctonionNorm
  CDLinearLemmas
  CDSignBridge
  SedenionSignBridge.
Open Scope R_scope.

(** * CDFusedBilinear: flattened multiplication specs for Cayley-Dickson levels.

    This file does not try to certify machine-level scheduling wins. Instead it
    provides compact bilinear operator forms that proofs can rewrite to, so
    later Brown lanes can avoid repeatedly unfolding the recursive tower
    presentation. *)

Definition quat_mul_fused (p q : CDQuat) : CDQuat :=
  match p, q with
  | mkQuat a b c d, mkQuat e f g h =>
      mkQuat
        (a * e - b * f - c * g - d * h)
        (a * f + b * e + c * h - d * g)
        (a * g - b * h + c * e + d * f)
        (a * h + b * g - c * f + d * e)
  end.

Definition oct_mul_fused (x y : CDOct) : CDOct :=
  match x, y with
  | mkOct a b, mkOct c d =>
      mkOct
        (quat_add (quat_mul a c)
                  (quat_neg (quat_mul (quat_conj d) b)))
        (quat_add (quat_mul d a)
                  (quat_mul b (quat_conj c)))
  end.

Definition sed_mul_fused (x y : CDSed) : CDSed :=
  match x, y with
  | mkSed a b, mkSed c d =>
      mkSed
        (oct_add (oct_mul a c)
                 (oct_neg (oct_mul (oct_conj d) b)))
        (oct_add (oct_mul d a)
                 (oct_mul b (oct_conj c)))
  end.

Theorem quat_mul_fused_eq : forall p q : CDQuat,
  quat_mul_fused p q = quat_mul p q.
Proof.
  intros [a b c d] [e f g h].
  unfold quat_mul_fused, quat_mul.
  simpl.
  apply (f_equal4 mkQuat); ring.
Qed.

Theorem oct_mul_fused_eq : forall x y : CDOct,
  oct_mul_fused x y = oct_mul x y.
Proof.
  intros [a b] [c d].
  unfold oct_mul_fused, oct_mul.
  simpl.
  reflexivity.
Qed.

Theorem sed_mul_fused_eq : forall x y : CDSed,
  sed_mul_fused x y = sed_mul x y.
Proof.
  intros [a b] [c d].
  unfold sed_mul_fused, sed_mul.
  simpl.
  unfold oct_add, oct_neg.
  reflexivity.
Qed.

Record CDFusedBilinearSurface
    (A : Type)
    (add mul fused_mul : A -> A -> A)
    (scale : R -> A -> A) := {
  cd_fused_mul_eq :
    forall x y : A, fused_mul x y = mul x y;
  cd_fused_mul_add_left :
    forall x x' y : A,
      fused_mul (add x x') y =
      add (fused_mul x y) (fused_mul x' y);
  cd_fused_mul_add_right :
    forall x y y' : A,
      fused_mul x (add y y') =
      add (fused_mul x y) (fused_mul x y');
  cd_fused_mul_scale_left :
    forall (r : R) (x y : A),
      fused_mul (scale r x) y = scale r (fused_mul x y);
  cd_fused_mul_scale_right :
    forall (r : R) (x y : A),
      fused_mul x (scale r y) = scale r (fused_mul x y)
}.

Definition quat_fused_bilinear_surface :
  CDFusedBilinearSurface CDQuat quat_add quat_mul quat_mul_fused quat_scale.
Proof.
  refine
    {| cd_fused_mul_eq := quat_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_add_left.
  - intros x y y'.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_add_right.
  - intros r x y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_scale_left.
  - intros r x y.
    repeat rewrite quat_mul_fused_eq.
    apply quat_mul_scale_right.
Defined.

Definition oct_fused_bilinear_surface :
  CDFusedBilinearSurface CDOct oct_add oct_mul oct_mul_fused oct_scale.
Proof.
  refine
    {| cd_fused_mul_eq := oct_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_add_left.
  - intros x y y'.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_add_right.
  - intros r x y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_scale_left.
  - intros r x y.
    repeat rewrite oct_mul_fused_eq.
    apply oct_mul_scale_right.
Defined.

Definition sed_fused_bilinear_surface :
  CDFusedBilinearSurface CDSed sed_add sed_mul sed_mul_fused sed_scale.
Proof.
  refine
    {| cd_fused_mul_eq := sed_mul_fused_eq |}.
  - intros x x' y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_add_left.
  - intros x y y'.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_add_right.
  - intros r x y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_scale_left.
  - intros r x y.
    repeat rewrite sed_mul_fused_eq.
    apply sed_mul_scale_right.
Defined.

Theorem oct_mul_fused_basis_xor : forall i j : nat,
  (i < 8)%nat -> (j < 8)%nat ->
  oct_mul_fused (oct_e i) (oct_e j) =
    oct_scale (sign_to_R (oct_sign i j)) (oct_e (Nat.lxor i j)).
Proof.
  intros i j Hi Hj.
  rewrite oct_mul_fused_eq.
  apply oct_basis_mul_xor; assumption.
Qed.

Record CDOctBasisFusedSurface := {
  cd_oct_fused_basis_xor :
    forall i j : nat,
      (i < 8)%nat -> (j < 8)%nat ->
      oct_mul_fused (oct_e i) (oct_e j) =
        oct_scale (sign_to_R (oct_sign i j)) (oct_e (Nat.lxor i j));
  cd_oct_fused_self_neg :
    forall i : nat,
      (1 <= i)%nat -> (i < 8)%nat ->
      oct_mul_fused (oct_e i) (oct_e i) = oct_neg (oct_e 0)
}.

Definition oct_basis_fused_surface : CDOctBasisFusedSurface.
Proof.
  refine {| cd_oct_fused_basis_xor := oct_mul_fused_basis_xor |}.
  intros i Hlo Hhi.
  rewrite oct_mul_fused_eq.
  apply oct_mul_self_neg; assumption.
Defined.

Definition sed_sign (i j : nat) : Z := cd_sign_fuel 5 16 i j.

Ltac close_sed_fused_case :=
  vm_compute;
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  ring.

Ltac destruct_nat_disjunction H :=
  repeat match type of H with
         | _ \/ _ =>
             let H' := fresh H in
             destruct H as [->|H'];
             [| rename H' into H]
         end;
  subst.

Theorem sed_mul_fused_basis_xor : forall i j : nat,
  (i < 16)%nat -> (j < 16)%nat ->
  sed_mul_fused (sed_e i) (sed_e j) =
    sed_scale (sign_to_R (sed_sign i j)) (sed_e (Nat.lxor i j)).
Proof.
  intros i j Hi Hj.
  assert (Hcasesi :
    i = 0%nat \/ i = 1%nat \/ i = 2%nat \/ i = 3%nat \/
    i = 4%nat \/ i = 5%nat \/ i = 6%nat \/ i = 7%nat \/
    i = 8%nat \/ i = 9%nat \/ i = 10%nat \/ i = 11%nat \/
    i = 12%nat \/ i = 13%nat \/ i = 14%nat \/ i = 15%nat)
    by lia.
  assert (Hcasesj :
    j = 0%nat \/ j = 1%nat \/ j = 2%nat \/ j = 3%nat \/
    j = 4%nat \/ j = 5%nat \/ j = 6%nat \/ j = 7%nat \/
    j = 8%nat \/ j = 9%nat \/ j = 10%nat \/ j = 11%nat \/
    j = 12%nat \/ j = 13%nat \/ j = 14%nat \/ j = 15%nat)
    by lia.
  destruct_nat_disjunction Hcasesi.
  all: destruct_nat_disjunction Hcasesj.
  all: close_sed_fused_case.
Qed.

Theorem sed_mul_fused_self_neg : forall i : nat,
  (1 <= i)%nat -> (i < 16)%nat ->
  sed_mul_fused (sed_e i) (sed_e i) = sed_neg (sed_e 0).
Proof.
  intros i Hlo Hhi.
  assert (Hcases :
    i = 1%nat \/ i = 2%nat \/ i = 3%nat \/ i = 4%nat \/ i = 5%nat \/
    i = 6%nat \/ i = 7%nat \/ i = 8%nat \/ i = 9%nat \/ i = 10%nat \/
    i = 11%nat \/ i = 12%nat \/ i = 13%nat \/ i = 14%nat \/ i = 15%nat)
    by lia.
  destruct_nat_disjunction Hcases.
  all: close_sed_fused_case.
Qed.

Record CDSedBasisFusedSurface := {
  cd_sed_fused_basis_xor :
    forall i j : nat,
      (i < 16)%nat -> (j < 16)%nat ->
      sed_mul_fused (sed_e i) (sed_e j) =
        sed_scale (sign_to_R (sed_sign i j)) (sed_e (Nat.lxor i j));
  cd_sed_fused_self_neg :
    forall i : nat,
      (1 <= i)%nat -> (i < 16)%nat ->
      sed_mul_fused (sed_e i) (sed_e i) = sed_neg (sed_e 0)
}.

Definition sed_basis_fused_surface : CDSedBasisFusedSurface.
Proof.
  refine {| cd_sed_fused_basis_xor := sed_mul_fused_basis_xor;
            cd_sed_fused_self_neg := sed_mul_fused_self_neg |}.
Defined.

Lemma sed_mul_fused_e1_e9 :
  sed_mul_fused (sed_e 1) (sed_e 9) = sed_scale (-1) (sed_e 8).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e9. Qed.
Lemma sed_mul_fused_e9_e2 :
  sed_mul_fused (sed_e 9) (sed_e 2) = sed_scale (-1) (sed_e 11).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e9_e2. Qed.
Lemma sed_mul_fused_e8_e2 :
  sed_mul_fused (sed_e 8) (sed_e 2) = sed_scale (-1) (sed_e 10).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e8_e2. Qed.
Lemma sed_mul_fused_e1_e11 :
  sed_mul_fused (sed_e 1) (sed_e 11) = sed_e 10.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e11. Qed.
Lemma sed_mul_fused_e1_e2 :
  sed_mul_fused (sed_e 1) (sed_e 2) = sed_e 3.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e2. Qed.
Lemma sed_mul_fused_e2_e4 :
  sed_mul_fused (sed_e 2) (sed_e 4) = sed_e 6.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e2_e4. Qed.
Lemma sed_mul_fused_e3_e4 :
  sed_mul_fused (sed_e 3) (sed_e 4) = sed_e 7.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e3_e4. Qed.
Lemma sed_mul_fused_e1_e6 :
  sed_mul_fused (sed_e 1) (sed_e 6) = sed_scale (-1) (sed_e 7).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e6. Qed.
Lemma sed_mul_fused_e2_e9 :
  sed_mul_fused (sed_e 2) (sed_e 9) = sed_e 11.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e2_e9. Qed.
Lemma sed_mul_fused_e3_e9 :
  sed_mul_fused (sed_e 3) (sed_e 9) = sed_scale (-1) (sed_e 10).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e3_e9. Qed.
Lemma sed_mul_fused_e2_e1 :
  sed_mul_fused (sed_e 2) (sed_e 1) = sed_scale (-1) (sed_e 3).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e2_e1. Qed.
Lemma sed_mul_fused_e1_e10 :
  sed_mul_fused (sed_e 1) (sed_e 10) = sed_scale (-1) (sed_e 11).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e10. Qed.
Lemma sed_mul_fused_e3_e10 :
  sed_mul_fused (sed_e 3) (sed_e 10) = sed_e 9.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e3_e10. Qed.
Lemma sed_mul_fused_e2_e11 :
  sed_mul_fused (sed_e 2) (sed_e 11) = sed_scale (-1) (sed_e 9).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e2_e11. Qed.
Lemma sed_mul_fused_e3_e1 :
  sed_mul_fused (sed_e 3) (sed_e 1) = sed_e 2.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e3_e1. Qed.
Lemma sed_mul_fused_e4_e1 :
  sed_mul_fused (sed_e 4) (sed_e 1) = sed_scale (-1) (sed_e 5).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e4_e1. Qed.
Lemma sed_mul_fused_e1_e12 :
  sed_mul_fused (sed_e 1) (sed_e 12) = sed_scale (-1) (sed_e 13).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e12. Qed.
Lemma sed_mul_fused_e5_e12 :
  sed_mul_fused (sed_e 5) (sed_e 12) = sed_e 9.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e5_e12. Qed.
Lemma sed_mul_fused_e4_e13 :
  sed_mul_fused (sed_e 4) (sed_e 13) = sed_scale (-1) (sed_e 9).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e4_e13. Qed.
Lemma sed_mul_fused_e5_e1 :
  sed_mul_fused (sed_e 5) (sed_e 1) = sed_e 4.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e5_e1. Qed.
Lemma sed_mul_fused_e1_e13 :
  sed_mul_fused (sed_e 1) (sed_e 13) = sed_e 12.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e13. Qed.
Lemma sed_mul_fused_e5_e13 :
  sed_mul_fused (sed_e 5) (sed_e 13) = sed_scale (-1) (sed_e 8).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e5_e13. Qed.
Lemma sed_mul_fused_e6_e1 :
  sed_mul_fused (sed_e 6) (sed_e 1) = sed_e 7.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e6_e1. Qed.
Lemma sed_mul_fused_e1_e14 :
  sed_mul_fused (sed_e 1) (sed_e 14) = sed_e 15.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e14. Qed.
Lemma sed_mul_fused_e7_e14 :
  sed_mul_fused (sed_e 7) (sed_e 14) = sed_scale (-1) (sed_e 9).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e7_e14. Qed.
Lemma sed_mul_fused_e6_e15 :
  sed_mul_fused (sed_e 6) (sed_e 15) = sed_e 9.
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e6_e15. Qed.
Lemma sed_mul_fused_e7_e1 :
  sed_mul_fused (sed_e 7) (sed_e 1) = sed_scale (-1) (sed_e 6).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e7_e1. Qed.
Lemma sed_mul_fused_e1_e15 :
  sed_mul_fused (sed_e 1) (sed_e 15) = sed_scale (-1) (sed_e 14).
Proof. rewrite sed_mul_fused_eq. exact sed_mul_e1_e15. Qed.

Record CDSedFocusedBasisFusedSurface := {
  cd_sed_fused_full : CDSedBasisFusedSurface;
  cd_sed_fused_e1_e9 : sed_mul_fused (sed_e 1) (sed_e 9) = sed_scale (-1) (sed_e 8);
  cd_sed_fused_e9_e2 : sed_mul_fused (sed_e 9) (sed_e 2) = sed_scale (-1) (sed_e 11);
  cd_sed_fused_e8_e2 : sed_mul_fused (sed_e 8) (sed_e 2) = sed_scale (-1) (sed_e 10);
  cd_sed_fused_e1_e11 : sed_mul_fused (sed_e 1) (sed_e 11) = sed_e 10;
  cd_sed_fused_e1_e2 : sed_mul_fused (sed_e 1) (sed_e 2) = sed_e 3;
  cd_sed_fused_e2_e4 : sed_mul_fused (sed_e 2) (sed_e 4) = sed_e 6;
  cd_sed_fused_e3_e4 : sed_mul_fused (sed_e 3) (sed_e 4) = sed_e 7;
  cd_sed_fused_e1_e6 : sed_mul_fused (sed_e 1) (sed_e 6) = sed_scale (-1) (sed_e 7);
  cd_sed_fused_e2_e9 : sed_mul_fused (sed_e 2) (sed_e 9) = sed_e 11;
  cd_sed_fused_e3_e9 : sed_mul_fused (sed_e 3) (sed_e 9) = sed_scale (-1) (sed_e 10);
  cd_sed_fused_e2_e1 : sed_mul_fused (sed_e 2) (sed_e 1) = sed_scale (-1) (sed_e 3);
  cd_sed_fused_e1_e10 : sed_mul_fused (sed_e 1) (sed_e 10) = sed_scale (-1) (sed_e 11);
  cd_sed_fused_e3_e10 : sed_mul_fused (sed_e 3) (sed_e 10) = sed_e 9;
  cd_sed_fused_e2_e11 : sed_mul_fused (sed_e 2) (sed_e 11) = sed_scale (-1) (sed_e 9);
  cd_sed_fused_e3_e1 : sed_mul_fused (sed_e 3) (sed_e 1) = sed_e 2;
  cd_sed_fused_e4_e1 : sed_mul_fused (sed_e 4) (sed_e 1) = sed_scale (-1) (sed_e 5);
  cd_sed_fused_e1_e12 : sed_mul_fused (sed_e 1) (sed_e 12) = sed_scale (-1) (sed_e 13);
  cd_sed_fused_e5_e12 : sed_mul_fused (sed_e 5) (sed_e 12) = sed_e 9;
  cd_sed_fused_e4_e13 : sed_mul_fused (sed_e 4) (sed_e 13) = sed_scale (-1) (sed_e 9);
  cd_sed_fused_e5_e1 : sed_mul_fused (sed_e 5) (sed_e 1) = sed_e 4;
  cd_sed_fused_e1_e13 : sed_mul_fused (sed_e 1) (sed_e 13) = sed_e 12;
  cd_sed_fused_e5_e13 : sed_mul_fused (sed_e 5) (sed_e 13) = sed_scale (-1) (sed_e 8);
  cd_sed_fused_e6_e1 : sed_mul_fused (sed_e 6) (sed_e 1) = sed_e 7;
  cd_sed_fused_e1_e14 : sed_mul_fused (sed_e 1) (sed_e 14) = sed_e 15;
  cd_sed_fused_e7_e14 : sed_mul_fused (sed_e 7) (sed_e 14) = sed_scale (-1) (sed_e 9);
  cd_sed_fused_e6_e15 : sed_mul_fused (sed_e 6) (sed_e 15) = sed_e 9;
  cd_sed_fused_e7_e1 : sed_mul_fused (sed_e 7) (sed_e 1) = sed_scale (-1) (sed_e 6);
  cd_sed_fused_e1_e15 : sed_mul_fused (sed_e 1) (sed_e 15) = sed_scale (-1) (sed_e 14)
}.

Definition sed_focused_basis_fused_surface : CDSedFocusedBasisFusedSurface.
Proof.
  refine
    {| cd_sed_fused_full := sed_basis_fused_surface;
       cd_sed_fused_e1_e9 := sed_mul_fused_e1_e9;
       cd_sed_fused_e9_e2 := sed_mul_fused_e9_e2;
       cd_sed_fused_e8_e2 := sed_mul_fused_e8_e2;
       cd_sed_fused_e1_e11 := sed_mul_fused_e1_e11;
       cd_sed_fused_e1_e2 := sed_mul_fused_e1_e2;
       cd_sed_fused_e2_e4 := sed_mul_fused_e2_e4;
       cd_sed_fused_e3_e4 := sed_mul_fused_e3_e4;
       cd_sed_fused_e1_e6 := sed_mul_fused_e1_e6;
       cd_sed_fused_e2_e9 := sed_mul_fused_e2_e9;
       cd_sed_fused_e3_e9 := sed_mul_fused_e3_e9;
       cd_sed_fused_e2_e1 := sed_mul_fused_e2_e1;
       cd_sed_fused_e1_e10 := sed_mul_fused_e1_e10;
       cd_sed_fused_e3_e10 := sed_mul_fused_e3_e10;
       cd_sed_fused_e2_e11 := sed_mul_fused_e2_e11;
       cd_sed_fused_e3_e1 := sed_mul_fused_e3_e1;
       cd_sed_fused_e4_e1 := sed_mul_fused_e4_e1;
       cd_sed_fused_e1_e12 := sed_mul_fused_e1_e12;
       cd_sed_fused_e5_e12 := sed_mul_fused_e5_e12;
       cd_sed_fused_e4_e13 := sed_mul_fused_e4_e13;
       cd_sed_fused_e5_e1 := sed_mul_fused_e5_e1;
       cd_sed_fused_e1_e13 := sed_mul_fused_e1_e13;
       cd_sed_fused_e5_e13 := sed_mul_fused_e5_e13;
       cd_sed_fused_e6_e1 := sed_mul_fused_e6_e1;
       cd_sed_fused_e1_e14 := sed_mul_fused_e1_e14;
       cd_sed_fused_e7_e14 := sed_mul_fused_e7_e14;
       cd_sed_fused_e6_e15 := sed_mul_fused_e6_e15;
       cd_sed_fused_e7_e1 := sed_mul_fused_e7_e1;
       cd_sed_fused_e1_e15 := sed_mul_fused_e1_e15 |}.
Defined.
