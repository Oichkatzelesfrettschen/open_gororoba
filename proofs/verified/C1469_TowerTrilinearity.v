(** * C-1469: Tower Trilinearity -- Cocycle Properties at dim=16, 32, 64.

    The sign cocycle properties (identity, square, anti-commutativity)
    are verified at three CD levels: dim=16 (sedenion), dim=32 (pathion),
    dim=64 (chingon).

    This establishes that the tower proof architecture works at all
    CD levels: the same structural lemmas (mul_scale_left, scale_sub)
    apply because the underlying sign cocycle preserves its structure
    through the CD doubling process.

    Mirrors: crates/cd_kernel/src/cayley_dickson/signs.rs *)

From Stdlib Require Import Arith PeanoNat Bool List ZArith.
Import ListNotations.

Fixpoint cd_sign_fuel (fuel dim p q : nat) : Z :=
  match fuel with
  | O => 0%Z
  | S fuel' =>
    match dim with
    | O | 1 => 1%Z
    | _ =>
      let half := Nat.div dim 2 in
      if Nat.ltb p half then
        if Nat.ltb q half then cd_sign_fuel fuel' half p q
        else cd_sign_fuel fuel' half (q - half) p
      else
        if Nat.ltb q half then
          if Nat.eqb q 0 then cd_sign_fuel fuel' half (p - half) q
          else Z.opp (cd_sign_fuel fuel' half (p - half) q)
        else
          let qh := q - half in
          if Nat.eqb qh 0 then (-1)%Z
          else cd_sign_fuel fuel' half qh (p - half)
    end
  end.

Definition sign16 (i j : nat) : Z := cd_sign_fuel 5 16 i j.
Definition sign32 (i j : nat) : Z := cd_sign_fuel 6 32 i j.
Definition sign64 (i j : nat) : Z := cd_sign_fuel 7 64 i j.

(** ========== IDENTITY: sigma(i, 0) = sigma(0, i) = 1 ========== *)

Theorem identity_16 :
  forallb (fun i => (Z.eqb (sign16 i 0) 1 && Z.eqb (sign16 0 i) 1)%bool) (seq 0 16) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem identity_32 :
  forallb (fun i => (Z.eqb (sign32 i 0) 1 && Z.eqb (sign32 0 i) 1)%bool) (seq 0 32) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem identity_64 :
  forallb (fun i => (Z.eqb (sign64 i 0) 1 && Z.eqb (sign64 0 i) 1)%bool) (seq 0 64) = true.
Proof. vm_compute. reflexivity. Qed.

(** ========== SQUARES: sigma(i, i) = -1 for i >= 1 ========== *)

Theorem squares_16 :
  forallb (fun i => Z.eqb (sign16 i i) (-1)%Z) (seq 1 15) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem squares_32 :
  forallb (fun i => Z.eqb (sign32 i i) (-1)%Z) (seq 1 31) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem squares_64 :
  forallb (fun i => Z.eqb (sign64 i i) (-1)%Z) (seq 1 63) = true.
Proof. vm_compute. reflexivity. Qed.

(** ========== ANTI-COMMUTATIVITY: sigma(i,j) + sigma(j,i) = 0 ========== *)

Theorem anticomm_16 :
  forallb (fun i => forallb (fun j =>
    if Nat.eqb i j then true
    else Z.eqb (sign16 i j + sign16 j i)%Z 0%Z
  ) (seq 1 15)) (seq 1 15) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem anticomm_32 :
  forallb (fun i => forallb (fun j =>
    if Nat.eqb i j then true
    else Z.eqb (sign32 i j + sign32 j i)%Z 0%Z
  ) (seq 1 31)) (seq 1 31) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem anticomm_64 :
  forallb (fun i => forallb (fun j =>
    if Nat.eqb i j then true
    else Z.eqb (sign64 i j + sign64 j i)%Z 0%Z
  ) (seq 1 63)) (seq 1 63) = true.
Proof. vm_compute. reflexivity. Qed.

(** ========== DIM=128 (Routon) ========== *)

Definition sign128 (i j : nat) : Z := cd_sign_fuel 8 128 i j.

Theorem identity_128 :
  forallb (fun i => (Z.eqb (sign128 i 0) 1 && Z.eqb (sign128 0 i) 1)%bool) (seq 0 128) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem squares_128 :
  forallb (fun i => Z.eqb (sign128 i i) (-1)%Z) (seq 1 127) = true.
Proof. vm_compute. reflexivity. Qed.

(** ========== DIM=256 (Voudon) ========== *)

Definition sign256 (i j : nat) : Z := cd_sign_fuel 9 256 i j.

Theorem identity_256 :
  forallb (fun i => (Z.eqb (sign256 i 0) 1 && Z.eqb (sign256 0 i) 1)%bool) (seq 0 256) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem squares_256 :
  forallb (fun i => Z.eqb (sign256 i i) (-1)%Z) (seq 1 255) = true.
Proof. vm_compute. reflexivity. Qed.
