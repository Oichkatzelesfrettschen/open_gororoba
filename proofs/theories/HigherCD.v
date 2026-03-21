(** * HigherCD: Cayley-Dickson types at dim=64, 128, 256.

    CDChingon = (CDPathion, CDPathion)   dim=64
    CDRouton  = (CDChingon, CDChingon)   dim=128
    CDVoudon  = (CDRouton, CDRouton)     dim=256

    Mirrors: crates/gororoba_algebra/src/construction/cd_tower.rs *)

From Stdlib Require Import Reals.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
                                 OctonionNorm Pathion.
Open Scope R_scope.

(** ========== CHINGON (dim=64) ========== *)

Record CDChingon := mkChingon { chingon_lo : CDPathion; chingon_hi : CDPathion }.
Definition chingon_conj (x : CDChingon) : CDChingon :=
  mkChingon (pathion_conj (chingon_lo x)) (pathion_neg (chingon_hi x)).
Definition chingon_add (x y : CDChingon) : CDChingon :=
  mkChingon (pathion_add (chingon_lo x) (chingon_lo y))
            (pathion_add (chingon_hi x) (chingon_hi y)).
Definition chingon_neg (x : CDChingon) : CDChingon :=
  mkChingon (pathion_neg (chingon_lo x)) (pathion_neg (chingon_hi x)).
Definition chingon_sub (x y : CDChingon) : CDChingon :=
  chingon_add x (chingon_neg y).
Definition chingon_scale (r : R) (x : CDChingon) : CDChingon :=
  mkChingon (pathion_scale r (chingon_lo x)) (pathion_scale r (chingon_hi x)).
Definition chingon_zero : CDChingon := mkChingon pathion_zero pathion_zero.
Definition chingon_mul (x y : CDChingon) : CDChingon :=
  mkChingon
    (pathion_sub (pathion_mul (chingon_lo x) (chingon_lo y))
                 (pathion_mul (pathion_conj (chingon_hi y)) (chingon_hi x)))
    (pathion_add (pathion_mul (chingon_hi y) (chingon_lo x))
                 (pathion_mul (chingon_hi x) (pathion_conj (chingon_lo y)))).
Definition chingon_assoc (a b c : CDChingon) : CDChingon :=
  chingon_sub (chingon_mul (chingon_mul a b) c)
              (chingon_mul a (chingon_mul b c)).

(** ========== ROUTON (dim=128) ========== *)

Record CDRouton := mkRouton { routon_lo : CDChingon; routon_hi : CDChingon }.
Definition routon_conj (x : CDRouton) : CDRouton :=
  mkRouton (chingon_conj (routon_lo x)) (chingon_neg (routon_hi x)).
Definition routon_add (x y : CDRouton) : CDRouton :=
  mkRouton (chingon_add (routon_lo x) (routon_lo y))
           (chingon_add (routon_hi x) (routon_hi y)).
Definition routon_neg (x : CDRouton) : CDRouton :=
  mkRouton (chingon_neg (routon_lo x)) (chingon_neg (routon_hi x)).
Definition routon_sub (x y : CDRouton) : CDRouton :=
  routon_add x (routon_neg y).
Definition routon_scale (r : R) (x : CDRouton) : CDRouton :=
  mkRouton (chingon_scale r (routon_lo x)) (chingon_scale r (routon_hi x)).
Definition routon_zero : CDRouton := mkRouton chingon_zero chingon_zero.
Definition routon_mul (x y : CDRouton) : CDRouton :=
  mkRouton
    (chingon_sub (chingon_mul (routon_lo x) (routon_lo y))
                 (chingon_mul (chingon_conj (routon_hi y)) (routon_hi x)))
    (chingon_add (chingon_mul (routon_hi y) (routon_lo x))
                 (chingon_mul (routon_hi x) (chingon_conj (routon_lo y)))).
Definition routon_assoc (a b c : CDRouton) : CDRouton :=
  routon_sub (routon_mul (routon_mul a b) c)
             (routon_mul a (routon_mul b c)).

(** ========== VOUDON (dim=256) ========== *)

Record CDVoudon := mkVoudon { voudon_lo : CDRouton; voudon_hi : CDRouton }.
Definition voudon_conj (x : CDVoudon) : CDVoudon :=
  mkVoudon (routon_conj (voudon_lo x)) (routon_neg (voudon_hi x)).
Definition voudon_add (x y : CDVoudon) : CDVoudon :=
  mkVoudon (routon_add (voudon_lo x) (voudon_lo y))
           (routon_add (voudon_hi x) (voudon_hi y)).
Definition voudon_neg (x : CDVoudon) : CDVoudon :=
  mkVoudon (routon_neg (voudon_lo x)) (routon_neg (voudon_hi x)).
Definition voudon_sub (x y : CDVoudon) : CDVoudon :=
  voudon_add x (voudon_neg y).
Definition voudon_scale (r : R) (x : CDVoudon) : CDVoudon :=
  mkVoudon (routon_scale r (voudon_lo x)) (routon_scale r (voudon_hi x)).
Definition voudon_zero : CDVoudon := mkVoudon routon_zero routon_zero.
Definition voudon_mul (x y : CDVoudon) : CDVoudon :=
  mkVoudon
    (routon_sub (routon_mul (voudon_lo x) (voudon_lo y))
                (routon_mul (routon_conj (voudon_hi y)) (voudon_hi x)))
    (routon_add (routon_mul (voudon_hi y) (voudon_lo x))
                (routon_mul (voudon_hi x) (routon_conj (voudon_lo y)))).
Definition voudon_assoc (a b c : CDVoudon) : CDVoudon :=
  voudon_sub (voudon_mul (voudon_mul a b) c)
             (voudon_mul a (voudon_mul b c)).

(** ========== ERISTON (dim=512) ========== *)

Record CDEriston := mkEriston { eriston_lo : CDVoudon; eriston_hi : CDVoudon }.
Definition eriston_conj (x : CDEriston) : CDEriston :=
  mkEriston (voudon_conj (eriston_lo x)) (voudon_neg (eriston_hi x)).
Definition eriston_add (x y : CDEriston) : CDEriston :=
  mkEriston (voudon_add (eriston_lo x) (eriston_lo y))
            (voudon_add (eriston_hi x) (eriston_hi y)).
Definition eriston_neg (x : CDEriston) : CDEriston :=
  mkEriston (voudon_neg (eriston_lo x)) (voudon_neg (eriston_hi x)).
Definition eriston_sub (x y : CDEriston) : CDEriston :=
  eriston_add x (eriston_neg y).
Definition eriston_scale (r : R) (x : CDEriston) : CDEriston :=
  mkEriston (voudon_scale r (eriston_lo x)) (voudon_scale r (eriston_hi x)).
Definition eriston_zero : CDEriston := mkEriston voudon_zero voudon_zero.
Definition eriston_mul (x y : CDEriston) : CDEriston :=
  mkEriston
    (voudon_sub (voudon_mul (eriston_lo x) (eriston_lo y))
                (voudon_mul (voudon_conj (eriston_hi y)) (eriston_hi x)))
    (voudon_add (voudon_mul (eriston_hi y) (eriston_lo x))
                (voudon_mul (eriston_hi x) (voudon_conj (eriston_lo y)))).
Definition eriston_assoc (a b c : CDEriston) : CDEriston :=
  eriston_sub (eriston_mul (eriston_mul a b) c)
              (eriston_mul a (eriston_mul b c)).

(** ========== DEKAVOUDON (dim=1024) ========== *)

Record CDDekaVoudon := mkDekaVoudon { dekavoudon_lo : CDEriston; dekavoudon_hi : CDEriston }.
Definition dekavoudon_conj (x : CDDekaVoudon) : CDDekaVoudon :=
  mkDekaVoudon (eriston_conj (dekavoudon_lo x)) (eriston_neg (dekavoudon_hi x)).
Definition dekavoudon_add (x y : CDDekaVoudon) : CDDekaVoudon :=
  mkDekaVoudon (eriston_add (dekavoudon_lo x) (dekavoudon_lo y))
               (eriston_add (dekavoudon_hi x) (dekavoudon_hi y)).
Definition dekavoudon_neg (x : CDDekaVoudon) : CDDekaVoudon :=
  mkDekaVoudon (eriston_neg (dekavoudon_lo x)) (eriston_neg (dekavoudon_hi x)).
Definition dekavoudon_sub (x y : CDDekaVoudon) : CDDekaVoudon :=
  dekavoudon_add x (dekavoudon_neg y).
Definition dekavoudon_scale (r : R) (x : CDDekaVoudon) : CDDekaVoudon :=
  mkDekaVoudon (eriston_scale r (dekavoudon_lo x)) (eriston_scale r (dekavoudon_hi x)).
Definition dekavoudon_zero : CDDekaVoudon := mkDekaVoudon eriston_zero eriston_zero.
Definition dekavoudon_mul (x y : CDDekaVoudon) : CDDekaVoudon :=
  mkDekaVoudon
    (eriston_sub (eriston_mul (dekavoudon_lo x) (dekavoudon_lo y))
                 (eriston_mul (eriston_conj (dekavoudon_hi y)) (dekavoudon_hi x)))
    (eriston_add (eriston_mul (dekavoudon_hi y) (dekavoudon_lo x))
                 (eriston_mul (dekavoudon_hi x) (eriston_conj (dekavoudon_lo y)))).
Definition dekavoudon_assoc (a b c : CDDekaVoudon) : CDDekaVoudon :=
  dekavoudon_sub (dekavoudon_mul (dekavoudon_mul a b) c)
                 (dekavoudon_mul a (dekavoudon_mul b c)).

(** ========== CD-2048 (dim=2048) ========== *)
Record CD2048 := mk2048 { cd2048_lo : CDDekaVoudon; cd2048_hi : CDDekaVoudon }.
Definition cd2048_conj x := mk2048 (dekavoudon_conj (cd2048_lo x)) (dekavoudon_neg (cd2048_hi x)).
Definition cd2048_add x y := mk2048 (dekavoudon_add (cd2048_lo x) (cd2048_lo y)) (dekavoudon_add (cd2048_hi x) (cd2048_hi y)).
Definition cd2048_neg x := mk2048 (dekavoudon_neg (cd2048_lo x)) (dekavoudon_neg (cd2048_hi x)).
Definition cd2048_sub x y := cd2048_add x (cd2048_neg y).
Definition cd2048_scale r x := mk2048 (dekavoudon_scale r (cd2048_lo x)) (dekavoudon_scale r (cd2048_hi x)).
Definition cd2048_mul x y := mk2048
  (dekavoudon_sub (dekavoudon_mul (cd2048_lo x) (cd2048_lo y)) (dekavoudon_mul (dekavoudon_conj (cd2048_hi y)) (cd2048_hi x)))
  (dekavoudon_add (dekavoudon_mul (cd2048_hi y) (cd2048_lo x)) (dekavoudon_mul (cd2048_hi x) (dekavoudon_conj (cd2048_lo y)))).
Definition cd2048_assoc a b c := cd2048_sub (cd2048_mul (cd2048_mul a b) c) (cd2048_mul a (cd2048_mul b c)).

(** ========== CD-4096 (dim=4096) ========== *)
Record CD4096 := mk4096 { cd4096_lo : CD2048; cd4096_hi : CD2048 }.
Definition cd4096_conj x := mk4096 (cd2048_conj (cd4096_lo x)) (cd2048_neg (cd4096_hi x)).
Definition cd4096_add x y := mk4096 (cd2048_add (cd4096_lo x) (cd4096_lo y)) (cd2048_add (cd4096_hi x) (cd4096_hi y)).
Definition cd4096_neg x := mk4096 (cd2048_neg (cd4096_lo x)) (cd2048_neg (cd4096_hi x)).
Definition cd4096_sub x y := cd4096_add x (cd4096_neg y).
Definition cd4096_scale r x := mk4096 (cd2048_scale r (cd4096_lo x)) (cd2048_scale r (cd4096_hi x)).
Definition cd4096_mul x y := mk4096
  (cd2048_sub (cd2048_mul (cd4096_lo x) (cd4096_lo y)) (cd2048_mul (cd2048_conj (cd4096_hi y)) (cd4096_hi x)))
  (cd2048_add (cd2048_mul (cd4096_hi y) (cd4096_lo x)) (cd2048_mul (cd4096_hi x) (cd2048_conj (cd4096_lo y)))).
Definition cd4096_assoc a b c := cd4096_sub (cd4096_mul (cd4096_mul a b) c) (cd4096_mul a (cd4096_mul b c)).

(** ========== CD-8192 (dim=8192) ========== *)
Record CD8192 := mk8192 { cd8192_lo : CD4096; cd8192_hi : CD4096 }.
Definition cd8192_conj x := mk8192 (cd4096_conj (cd8192_lo x)) (cd4096_neg (cd8192_hi x)).
Definition cd8192_add x y := mk8192 (cd4096_add (cd8192_lo x) (cd8192_lo y)) (cd4096_add (cd8192_hi x) (cd8192_hi y)).
Definition cd8192_neg x := mk8192 (cd4096_neg (cd8192_lo x)) (cd4096_neg (cd8192_hi x)).
Definition cd8192_sub x y := cd8192_add x (cd8192_neg y).
Definition cd8192_scale r x := mk8192 (cd4096_scale r (cd8192_lo x)) (cd4096_scale r (cd8192_hi x)).
Definition cd8192_mul x y := mk8192
  (cd4096_sub (cd4096_mul (cd8192_lo x) (cd8192_lo y)) (cd4096_mul (cd4096_conj (cd8192_hi y)) (cd8192_hi x)))
  (cd4096_add (cd4096_mul (cd8192_hi y) (cd8192_lo x)) (cd4096_mul (cd8192_hi x) (cd4096_conj (cd8192_lo y)))).
Definition cd8192_assoc a b c := cd8192_sub (cd8192_mul (cd8192_mul a b) c) (cd8192_mul a (cd8192_mul b c)).

(** ========== CD-16384 (dim=16384, Tessareskaidekavoudon) ========== *)
Record CD16384 := mk16384 { cd16384_lo : CD8192; cd16384_hi : CD8192 }.
Definition cd16384_conj x := mk16384 (cd8192_conj (cd16384_lo x)) (cd8192_neg (cd16384_hi x)).
Definition cd16384_add x y := mk16384 (cd8192_add (cd16384_lo x) (cd16384_lo y)) (cd8192_add (cd16384_hi x) (cd16384_hi y)).
Definition cd16384_neg x := mk16384 (cd8192_neg (cd16384_lo x)) (cd8192_neg (cd16384_hi x)).
Definition cd16384_sub x y := cd16384_add x (cd16384_neg y).
Definition cd16384_scale r x := mk16384 (cd8192_scale r (cd16384_lo x)) (cd8192_scale r (cd16384_hi x)).
Definition cd16384_mul x y := mk16384
  (cd8192_sub (cd8192_mul (cd16384_lo x) (cd16384_lo y)) (cd8192_mul (cd8192_conj (cd16384_hi y)) (cd16384_hi x)))
  (cd8192_add (cd8192_mul (cd16384_hi y) (cd16384_lo x)) (cd8192_mul (cd16384_hi x) (cd8192_conj (cd16384_lo y)))).
Definition cd16384_assoc a b c := cd16384_sub (cd16384_mul (cd16384_mul a b) c) (cd16384_mul a (cd16384_mul b c)).
