(** * SU3JacobiFull: Complete SU(3) Jacobi identity in Z[sqrt(3)].

    All 8 Gell-Mann structure constants including f_{458} = f_{678} = sqrt(3)/2
    are encoded as pairs (a, b) representing a + b*sqrt(3) in Z[sqrt(3)].

    Multiplication: (a1 + b1*s)(a2 + b2*s) = (a1*a2 + 3*b1*b2) + (a1*b2 + a2*b1)*s
    where s = sqrt(3).

    The Jacobi identity is verified for ALL C(8,3) = 56 ordered triples. *)

From Stdlib Require Import List Arith Bool ZArith.
Import ListNotations.

(** Z[sqrt(3)] element: (rational_part, sqrt3_part). *)
Definition Zs3 := (Z * Z)%type.

Definition zs3_zero : Zs3 := (0%Z, 0%Z).

Definition zs3_add (x y : Zs3) : Zs3 :=
  ((fst x + fst y)%Z, (snd x + snd y)%Z).

Definition zs3_neg (x : Zs3) : Zs3 :=
  (Z.opp (fst x), Z.opp (snd x)).

Definition zs3_mul (x y : Zs3) : Zs3 :=
  ((fst x * fst y + 3 * snd x * snd y)%Z,
   (fst x * snd y + snd x * fst y)%Z).

Definition zs3_eq (x y : Zs3) : bool :=
  (Z.eqb (fst x) (fst y) && Z.eqb (snd x) (snd y))%bool.

(** The 9 independent SU(3) structure constants as 2*f_{abc} in Z[sqrt(3)].
    Table: (a, b, c, 2*f) with a < b, stored as Zs3 pair. *)
Definition su3_full_2f : list (nat * nat * nat * Zs3) :=
  [ (1, 2, 3, (2%Z, 0%Z))      (* f_{123} = 1 *)
  ; (1, 4, 7, (1%Z, 0%Z))      (* f_{147} = 1/2 *)
  ; (1, 5, 6, ((-1)%Z, 0%Z))   (* f_{156} = -1/2 *)
  ; (2, 4, 6, (1%Z, 0%Z))      (* f_{246} = 1/2 *)
  ; (2, 5, 7, (1%Z, 0%Z))      (* f_{257} = 1/2 *)
  ; (3, 4, 5, (1%Z, 0%Z))      (* f_{345} = 1/2 *)
  ; (3, 6, 7, ((-1)%Z, 0%Z))   (* f_{367} = -1/2 *)
  ; (4, 5, 8, (0%Z, 1%Z))      (* f_{458} = sqrt(3)/2 *)
  ; (6, 7, 8, (0%Z, 1%Z))      (* f_{678} = sqrt(3)/2 *)
  ].

Theorem su3_full_count : length su3_full_2f = 9.
Proof. reflexivity. Qed.

(** Antisymmetric lookup in Z[sqrt(3)]. *)
Fixpoint lookup_2f_full (a b c : nat) (table : list (nat * nat * nat * Zs3)) : Zs3 :=
  match table with
  | [] => zs3_zero
  | (x, y, z, v) :: rest =>
    if (Nat.eqb a x && Nat.eqb b y && Nat.eqb c z)%bool then v
    else if (Nat.eqb b x && Nat.eqb a y && Nat.eqb c z)%bool then zs3_neg v
    else if (Nat.eqb a x && Nat.eqb c y && Nat.eqb b z)%bool then zs3_neg v
    else if (Nat.eqb b x && Nat.eqb c y && Nat.eqb a z)%bool then v
    else if (Nat.eqb c x && Nat.eqb a y && Nat.eqb b z)%bool then v
    else if (Nat.eqb c x && Nat.eqb b y && Nat.eqb a z)%bool then zs3_neg v
    else lookup_2f_full a b c rest
  end.

Definition f2_full (a b c : nat) : Zs3 :=
  lookup_2f_full a b c su3_full_2f.

(** Verify f_{123} = (2, 0) and f_{458} = (0, 1). *)
Theorem f123_full : zs3_eq (f2_full 1 2 3) (2%Z, 0%Z) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem f458_full : zs3_eq (f2_full 4 5 8) (0%Z, 1%Z) = true.
Proof. vm_compute. reflexivity. Qed.

(** Jacobi sum in Z[sqrt(3)]:
    sum_d (2f_{abd} * 2f_{cde} + 2f_{bcd} * 2f_{ade} + 2f_{cad} * 2f_{bde}) *)
Definition jacobi_sum_full (a b c e : nat) : Zs3 :=
  List.fold_left
    (fun acc d_minus_1 =>
      let d := S d_minus_1 in
      zs3_add acc
        (zs3_add (zs3_mul (f2_full a b d) (f2_full c d e))
          (zs3_add (zs3_mul (f2_full b c d) (f2_full a d e))
                   (zs3_mul (f2_full c a d) (f2_full b d e)))))
    (List.seq 0 8)
    zs3_zero.

(** Verify Jacobi for ALL triples (a,b,c) with a < b < c in {1..8}. *)
Definition all_jacobi_full : bool :=
  List.forallb (fun abc =>
    match abc with
    | (a, (b, c)) =>
      List.forallb (fun e =>
        zs3_eq (jacobi_sum_full a b c e) zs3_zero)
        [1;2;3;4;5;6;7;8]
    end)
    (List.flat_map (fun a =>
      List.flat_map (fun b =>
        List.map (fun c => (a, (b, c)))
          (List.seq (S b) (9 - S b)))
        (List.seq (S a) (9 - S a)))
      (List.seq 1 8)).

Theorem jacobi_identity_complete :
  all_jacobi_full = true.
Proof. vm_compute. reflexivity. Qed.
