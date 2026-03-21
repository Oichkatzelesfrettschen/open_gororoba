(** * Pathion: Cayley-Dickson construction at dim=32.

    Defines the pathion type via CD doubling of CDSed:
    - CDPathion = (CDSed, CDSed)   -- pathions, dim=32

    CD doubling formula: (a,b)(c,d) = (ac - conj(d)*b, da + b*conj(c)).

    Pathions lose all remaining regularity beyond sedenions.
    They have dim/2-1 = 15 missing ZD graph edges (involution pairs).

    Mirrors: crates/algebra_experimental/src/higher_cd.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** * Pathion = pair of sedenions via CD doubling. *)

Record CDPathion := mkPathion { pathion_lo : CDSed; pathion_hi : CDSed }.

Definition pathion_zero : CDPathion := mkPathion sed_zero sed_zero.

(** Pathion conjugate: (a, b)* = (a*, -b). *)
Definition pathion_conj (x : CDPathion) : CDPathion :=
  mkPathion (sed_conj (pathion_lo x)) (sed_neg (pathion_hi x)).

(** Pathion multiplication via CD doubling:
    (a,b)(c,d) = (ac - conj(d)*b, da + b*conj(c)). *)
Definition pathion_mul (x y : CDPathion) : CDPathion :=
  let a := pathion_lo x in
  let b := pathion_hi x in
  let c := pathion_lo y in
  let d := pathion_hi y in
  mkPathion
    (sed_sub (sed_mul a c) (sed_mul (sed_conj d) b))
    (sed_add (sed_mul d a) (sed_mul b (sed_conj c))).

(** * Pathion arithmetic. *)

Definition pathion_add (x y : CDPathion) : CDPathion :=
  mkPathion (sed_add (pathion_lo x) (pathion_lo y))
            (sed_add (pathion_hi x) (pathion_hi y)).

Definition pathion_neg (x : CDPathion) : CDPathion :=
  mkPathion (sed_neg (pathion_lo x)) (sed_neg (pathion_hi x)).

Definition pathion_sub (x y : CDPathion) : CDPathion :=
  pathion_add x (pathion_neg y).

(** Pathion scalar multiplication. *)
Definition pathion_scale (r : R) (x : CDPathion) : CDPathion :=
  mkPathion (sed_scale r (pathion_lo x)) (sed_scale r (pathion_hi x)).

(** * Pathion norm squared. *)

Definition pathion_norm_sq (x : CDPathion) : R :=
  sed_norm_sq (pathion_lo x) + sed_norm_sq (pathion_hi x).

(** * Basis element constructors for pathions (0..31). *)

Definition pathion_e (i : nat) : CDPathion :=
  match i with
  | 0  => mkPathion (sed_e 0)  sed_zero
  | 1  => mkPathion (sed_e 1)  sed_zero
  | 2  => mkPathion (sed_e 2)  sed_zero
  | 3  => mkPathion (sed_e 3)  sed_zero
  | 4  => mkPathion (sed_e 4)  sed_zero
  | 5  => mkPathion (sed_e 5)  sed_zero
  | 6  => mkPathion (sed_e 6)  sed_zero
  | 7  => mkPathion (sed_e 7)  sed_zero
  | 8  => mkPathion (sed_e 8)  sed_zero
  | 9  => mkPathion (sed_e 9)  sed_zero
  | 10 => mkPathion (sed_e 10) sed_zero
  | 11 => mkPathion (sed_e 11) sed_zero
  | 12 => mkPathion (sed_e 12) sed_zero
  | 13 => mkPathion (sed_e 13) sed_zero
  | 14 => mkPathion (sed_e 14) sed_zero
  | 15 => mkPathion (sed_e 15) sed_zero
  | 16 => mkPathion sed_zero (sed_e 0)
  | 17 => mkPathion sed_zero (sed_e 1)
  | 18 => mkPathion sed_zero (sed_e 2)
  | 19 => mkPathion sed_zero (sed_e 3)
  | 20 => mkPathion sed_zero (sed_e 4)
  | 21 => mkPathion sed_zero (sed_e 5)
  | 22 => mkPathion sed_zero (sed_e 6)
  | 23 => mkPathion sed_zero (sed_e 7)
  | 24 => mkPathion sed_zero (sed_e 8)
  | 25 => mkPathion sed_zero (sed_e 9)
  | 26 => mkPathion sed_zero (sed_e 10)
  | 27 => mkPathion sed_zero (sed_e 11)
  | 28 => mkPathion sed_zero (sed_e 12)
  | 29 => mkPathion sed_zero (sed_e 13)
  | 30 => mkPathion sed_zero (sed_e 14)
  | 31 => mkPathion sed_zero (sed_e 15)
  | _  => pathion_zero
  end.
