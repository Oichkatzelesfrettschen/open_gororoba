(** * S_3: The symmetric group on 3 elements.

    Concrete enumeration of all 6 permutations of {0, 1, 2} as an
    inductive type, with group action defined by exhaustive matching.

    Used to formalize S3-invariance of bilinear forms (C-885):
    any S3-invariant bilinear form on a 3-element set is necessarily
    proportional to the all-ones matrix (democratic mixing).

    Pattern follows FiniteBijection.v: inductive enum + exhaustive destruct. *)

From Stdlib Require Import Arith PeanoNat Lia List.
Import ListNotations.

(** Index type for a 3-element set. *)
Inductive Idx3 : Set :=
  | I0 | I1 | I2.

(** The 6 permutations of S_3.
    Named by their cycle structure:
    - S3id:   identity
    - S3t01:  transposition (0 1)
    - S3t02:  transposition (0 2)
    - S3t12:  transposition (1 2)
    - S3r012: 3-cycle (0 1 2) -- sends 0->1, 1->2, 2->0
    - S3r021: 3-cycle (0 2 1) -- sends 0->2, 2->1, 1->0 *)
Inductive S3Perm : Set :=
  | S3id | S3t01 | S3t02 | S3t12 | S3r012 | S3r021.

(** Group action: apply a permutation to an index. *)
Definition apply_s3 (p : S3Perm) (i : Idx3) : Idx3 :=
  match p, i with
  | S3id,   x   => x
  | S3t01,  I0  => I1
  | S3t01,  I1  => I0
  | S3t01,  I2  => I2
  | S3t02,  I0  => I2
  | S3t02,  I1  => I1
  | S3t02,  I2  => I0
  | S3t12,  I0  => I0
  | S3t12,  I1  => I2
  | S3t12,  I2  => I1
  | S3r012, I0  => I1
  | S3r012, I1  => I2
  | S3r012, I2  => I0
  | S3r021, I0  => I2
  | S3r021, I1  => I0
  | S3r021, I2  => I1
  end.

(** Identity acts trivially. *)
Lemma apply_s3_id : forall i : Idx3, apply_s3 S3id i = i.
Proof. destruct i; reflexivity. Qed.

(** All S3 elements as a list (for enumeration). *)
Definition all_s3 : list S3Perm := [S3id; S3t01; S3t02; S3t12; S3r012; S3r021].

(** All Idx3 elements as a list. *)
Definition all_idx3 : list Idx3 := [I0; I1; I2].

(** Transpositions are involutions (self-inverse). *)
Lemma s3t01_involution : forall i, apply_s3 S3t01 (apply_s3 S3t01 i) = i.
Proof. destruct i; reflexivity. Qed.

Lemma s3t02_involution : forall i, apply_s3 S3t02 (apply_s3 S3t02 i) = i.
Proof. destruct i; reflexivity. Qed.

Lemma s3t12_involution : forall i, apply_s3 S3t12 (apply_s3 S3t12 i) = i.
Proof. destruct i; reflexivity. Qed.

(** 3-cycles compose to identity after 3 applications. *)
Lemma s3r012_order3 : forall i,
  apply_s3 S3r012 (apply_s3 S3r012 (apply_s3 S3r012 i)) = i.
Proof. destruct i; reflexivity. Qed.

(** S3r021 is the inverse of S3r012. *)
Lemma s3r021_inverse_r012 : forall i,
  apply_s3 S3r021 (apply_s3 S3r012 i) = i.
Proof. destruct i; reflexivity. Qed.

(** Transitivity: for any two indices, some permutation maps one to the other. *)
Lemma s3_transitive : forall i j : Idx3,
  exists sigma : S3Perm, apply_s3 sigma i = j.
Proof.
  destruct i; destruct j.
  - exists S3id; reflexivity.
  - exists S3t01; reflexivity.
  - exists S3t02; reflexivity.
  - exists S3t01; reflexivity.
  - exists S3id; reflexivity.
  - exists S3t12; reflexivity.
  - exists S3t02; reflexivity.
  - exists S3t12; reflexivity.
  - exists S3id; reflexivity.
Qed.
