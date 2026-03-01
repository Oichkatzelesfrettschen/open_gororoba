(** * Cayley-Dickson dimension tower: finite bijection.

    Models the map cd_dimension_nacelle_map from
    adm_algebra_bridge.rs:189-198.

    The hypothesis: algebraic dimension determines natural segmentation
    symmetry of the warp bubble geometry. The map is the identity on
    the set {1, 2, 4, 8, 16} (Cayley-Dickson tower dimensions). *)

From Stdlib Require Import Arith PeanoNat Lia.

(** Inductive type for Cayley-Dickson algebra dimensions. *)
Inductive CDDim : Set :=
  | CDReal       (* dim = 1, reals *)
  | CDComplex    (* dim = 2, complex numbers *)
  | CDQuaternion (* dim = 4, quaternions *)
  | CDOctonion   (* dim = 8, octonions *)
  | CDSedenion.  (* dim = 16, sedenions *)

(** Map CD algebra to its dimension as a natural number. *)
Definition cd_dim_nat (d : CDDim) : nat :=
  match d with
  | CDReal       => 1
  | CDComplex    => 2
  | CDQuaternion => 4
  | CDOctonion   => 8
  | CDSedenion   => 16
  end.

(** Map CD dimension to nacelle count (the proposed identity map). *)
Definition nacelle_count (d : CDDim) : nat :=
  match d with
  | CDReal       => 1
  | CDComplex    => 2
  | CDQuaternion => 4
  | CDOctonion   => 8
  | CDSedenion   => 16
  end.

(** C-879: The dimension-to-nacelle map is the identity.
    Mirrors: cd_dimension_nacelle_map(n) == Some(n) for valid dims. *)
Theorem nacelle_is_identity :
  forall d : CDDim, nacelle_count d = cd_dim_nat d.
Proof.
  destruct d; reflexivity.
Qed.

(** The dimension map is injective on CDDim. *)
Theorem cd_dim_injective :
  forall d1 d2 : CDDim, cd_dim_nat d1 = cd_dim_nat d2 -> d1 = d2.
Proof.
  destruct d1; destruct d2; simpl; intro H;
    try reflexivity; try discriminate.
Qed.

(** The nacelle map is injective. *)
Theorem nacelle_injective :
  forall d1 d2 : CDDim, nacelle_count d1 = nacelle_count d2 -> d1 = d2.
Proof.
  destruct d1; destruct d2; simpl; intro H;
    try reflexivity; try discriminate.
Qed.

(** All CD dimensions are positive. *)
Theorem cd_dim_positive :
  forall d : CDDim, 0 < cd_dim_nat d.
Proof.
  destruct d; simpl; lia.
Qed.

(** Each CD dimension is a power of 2. *)
Theorem cd_dim_power_of_two :
  forall d : CDDim, exists k : nat, cd_dim_nat d = 2 ^ k.
Proof.
  destruct d.
  - exists 0. reflexivity.
  - exists 1. reflexivity.
  - exists 2. reflexivity.
  - exists 3. reflexivity.
  - exists 4. reflexivity.
Qed.
