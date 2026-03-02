(** * Sedenion: Cayley-Dickson construction at dims 8 and 16.

    Defines octonion and sedenion types via CD doubling of CDQuat:
    - CDOct = (CDQuat, CDQuat)   -- octonions, dim=8
    - CDSed = (CDOct, CDOct)     -- sedenions, dim=16

    CD doubling formula: (a,b)(c,d) = (ac - conj(d)*b, d*a + b*conj(c)).

    Octonions lose associativity but retain alternativity.
    Sedenions lose alternativity and gain zero divisors.

    Mirrors: algebra_core/src/construction/cayley_dickson.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra.

(** * Octonion = pair of quaternions via CD doubling. *)

Record CDOct := mkOct { oct_lo : CDQuat; oct_hi : CDQuat }.

Definition oct_zero : CDOct := mkOct quat_zero quat_zero.

(** Octonion conjugate: (a, b)* = (a*, -b). *)
Definition oct_conj (x : CDOct) : CDOct :=
  mkOct (quat_conj (oct_lo x)) (quat_neg (oct_hi x)).

(** Octonion multiplication via CD doubling:
    (a,b)(c,d) = (ac - conj(d)*b, da + b*conj(c)). *)
Definition oct_mul (x y : CDOct) : CDOct :=
  let a := oct_lo x in
  let b := oct_hi x in
  let c := oct_lo y in
  let d := oct_hi y in
  mkOct
    (quat_add (quat_mul a c) (quat_neg (quat_mul (quat_conj d) b)))
    (quat_add (quat_mul d a) (quat_mul b (quat_conj c))).

(** * Sedenion = pair of octonions via CD doubling. *)

Record CDSed := mkSed { sed_lo : CDOct; sed_hi : CDOct }.

Definition sed_zero : CDSed := mkSed oct_zero oct_zero.

(** Sedenion conjugate: (a, b)* = (a*, -b). *)
Definition sed_conj (x : CDSed) : CDSed :=
  mkSed (oct_conj (sed_lo x)) (mkOct (quat_neg (oct_lo (sed_hi x)))
                                       (quat_neg (oct_hi (sed_hi x)))).

(** Sedenion multiplication via CD doubling:
    (a,b)(c,d) = (ac - conj(d)*b, da + b*conj(c)). *)
Definition sed_mul (x y : CDSed) : CDSed :=
  let a := sed_lo x in
  let b := sed_hi x in
  let c := sed_lo y in
  let d := sed_hi y in
  mkSed
    (mkOct
      (quat_add (oct_lo (oct_mul a c))
                (quat_neg (oct_lo (oct_mul (oct_conj d) b))))
      (quat_add (oct_hi (oct_mul a c))
                (quat_neg (oct_hi (oct_mul (oct_conj d) b)))))
    (mkOct
      (quat_add (oct_lo (oct_mul d a)) (oct_lo (oct_mul b (oct_conj c))))
      (quat_add (oct_hi (oct_mul d a)) (oct_hi (oct_mul b (oct_conj c))))).

(** * Basis element constructors for octonions. *)

(** Embed quaternion basis into octonion lo-half. *)
Definition oct_e (i : nat) : CDOct :=
  match i with
  | 0 => mkOct quat_one quat_zero
  | 1 => mkOct (mkQuat 0 1 0 0) quat_zero
  | 2 => mkOct (mkQuat 0 0 1 0) quat_zero
  | 3 => mkOct (mkQuat 0 0 0 1) quat_zero
  | 4 => mkOct quat_zero quat_one
  | 5 => mkOct quat_zero (mkQuat 0 1 0 0)
  | 6 => mkOct quat_zero (mkQuat 0 0 1 0)
  | 7 => mkOct quat_zero (mkQuat 0 0 0 1)
  | _ => oct_zero
  end.

(*<*sedzd>*)
(** Zero-divisor pair in the sedenions (Moreno-Froloff).
    a = e3 + e10, b = e6 - e15 (1-indexed basis). *)
Definition sed_zd_a : CDSed :=
  mkSed
    (mkOct (mkQuat 0 0 0 1) quat_zero)       (* e3 in lo *)
    (mkOct (mkQuat 0 0 1 0) quat_zero).       (* e10 in hi *)

Definition sed_zd_b : CDSed :=
  mkSed
    (mkOct quat_zero (mkQuat 0 0 1 0))        (* e6 in lo *)
    (mkOct quat_zero (mkQuat 0 0 0 (-1))).    (* -e15 in hi *)
(*</sedzd>*)
