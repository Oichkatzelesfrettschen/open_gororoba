(** * CDSignHalfStep: Structural one-step lemmas for cd_sign_fuel.

    Mirrors the 4-case branch structure of the recursive sign computation.
    These lemmas peel exactly one recursive layer of cd_sign_fuel.

    The octonion-level corollaries at the end keep the concrete finite facts
    in the fast path by discharging them with vm_compute over {0..7}. *)

From Stdlib Require Import ZArith Arith Bool Lia.
From OpenGororoba Require Import M3IsAssociator.

Open Scope Z_scope.

(** ================================================================== *)
(** * One-step structural lemmas.                                      *)
(** ================================================================== *)

Lemma cd_sign_fuel_lo_lo : forall (fuel dim p q : nat),
  (0 < Nat.div dim 2)%nat ->
  (p < Nat.div dim 2)%nat ->
  (q < Nat.div dim 2)%nat ->
  cd_sign_fuel (S fuel) dim p q = cd_sign_fuel fuel (Nat.div dim 2) p q.
Proof.
  intros fuel dim p q Hhalf Hp Hq.
  remember (Nat.div dim 2) as half eqn:Hhalfdef.
  destruct half as [|half']; [lia|].
  cbn [cd_sign_fuel].
  rewrite <- Hhalfdef.
  destruct (Nat.ltb p (S half')) eqn:Hp'; [| apply Nat.ltb_ge in Hp'; lia].
  destruct (Nat.ltb q (S half')) eqn:Hq'; [reflexivity|].
  apply Nat.ltb_ge in Hq'. lia.
Qed.

Lemma cd_sign_fuel_lo_hi : forall (fuel dim p q : nat),
  (0 < Nat.div dim 2)%nat ->
  (p < Nat.div dim 2)%nat ->
  (Nat.div dim 2 <= q)%nat ->
  cd_sign_fuel (S fuel) dim p q =
  cd_sign_fuel fuel (Nat.div dim 2) (q - Nat.div dim 2)%nat p.
Proof.
  intros fuel dim p q Hhalf Hp Hq.
  remember (Nat.div dim 2) as half eqn:Hhalfdef.
  destruct half as [|half']; [lia|].
  cbn [cd_sign_fuel].
  rewrite <- Hhalfdef.
  destruct (Nat.ltb p (S half')) eqn:Hp'; [| apply Nat.ltb_ge in Hp'; lia].
  destruct (Nat.ltb q (S half')) eqn:Hq'; [| reflexivity].
  apply Nat.ltb_lt in Hq'. lia.
Qed.

Lemma cd_sign_fuel_hi_lo : forall (fuel dim p q : nat),
  (0 < Nat.div dim 2)%nat ->
  (Nat.div dim 2 <= p)%nat ->
  (q < Nat.div dim 2)%nat ->
  cd_sign_fuel (S fuel) dim p q =
    let s := cd_sign_fuel fuel (Nat.div dim 2) (p - Nat.div dim 2)%nat q in
    if Nat.eqb q 0 then s else Z.opp s.
Proof.
  intros fuel dim p q Hhalf Hp Hq.
  remember (Nat.div dim 2) as half eqn:Hhalfdef.
  destruct half as [|half']; [lia|].
  cbn [cd_sign_fuel].
  rewrite <- Hhalfdef.
  destruct (Nat.ltb p (S half')) eqn:Hp'; [|].
  - apply Nat.ltb_lt in Hp'. lia.
  - destruct (Nat.ltb q (S half')) eqn:Hq'; [reflexivity|].
    apply Nat.ltb_ge in Hq'. lia.
Qed.

Lemma cd_sign_fuel_hi_hi : forall (fuel dim p q : nat),
  (0 < Nat.div dim 2)%nat ->
  (Nat.div dim 2 <= p)%nat ->
  (Nat.div dim 2 <= q)%nat ->
  cd_sign_fuel (S fuel) dim p q =
    if Nat.eqb (q - Nat.div dim 2)%nat 0
    then (-1)%Z
    else cd_sign_fuel fuel (Nat.div dim 2)
           (q - Nat.div dim 2)%nat (p - Nat.div dim 2)%nat.
Proof.
  intros fuel dim p q Hhalf Hp Hq.
  remember (Nat.div dim 2) as half eqn:Hhalfdef.
  destruct half as [|half']; [lia|].
  cbn [cd_sign_fuel].
  rewrite <- Hhalfdef.
  destruct (Nat.ltb p (S half')) eqn:Hp'; [apply Nat.ltb_lt in Hp'; lia|].
  destruct (Nat.ltb q (S half')) eqn:Hq'; [apply Nat.ltb_lt in Hq'; lia|].
  reflexivity.
Qed.

(** ================================================================== *)
(** * Concrete octonion corollaries.                                   *)
(** ================================================================== *)

Ltac destruct_oct_index n :=
  destruct n as [|[|[|[|[|[|[|[|]]]]]]]]; try lia.

Lemma cd_sign_fuel_0_left : forall q : nat,
  (q < 8)%nat ->
  cd_sign_fuel 4 8 0 q = 1%Z.
Proof.
  intros q Hq.
  destruct_oct_index q.
  all: vm_compute; reflexivity.
Qed.

Lemma cd_sign_fuel_0_right : forall p : nat,
  (p < 8)%nat ->
  cd_sign_fuel 4 8 p 0 = 1%Z.
Proof.
  intros p Hp.
  destruct_oct_index p.
  all: vm_compute; reflexivity.
Qed.

Lemma cd_sign_fuel_self_neg : forall p : nat,
  (0 < p)%nat ->
  (p < 8)%nat ->
  cd_sign_fuel 4 8 p p = (-1)%Z.
Proof.
  intros p Hp Hq.
  destruct_oct_index p.
  all: try lia.
  all: vm_compute; reflexivity.
Qed.

Lemma cd_sign_fuel_anticomm : forall p q : nat,
  (0 < p)%nat ->
  (p < 8)%nat ->
  (0 < q)%nat ->
  (q < 8)%nat ->
  p <> q ->
  cd_sign_fuel 4 8 p q = Z.opp (cd_sign_fuel 4 8 q p).
Proof.
  intros p q Hp0 Hp Hq0 Hq Hneq.
  destruct_oct_index p;
  destruct_oct_index q;
  try contradiction;
  try lia;
  vm_compute; reflexivity.
Qed.
