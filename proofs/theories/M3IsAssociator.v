(** * M3IsAssociator: m3 = Assoc on all 168 non-Fano triples.

    Uses the recursive cd_sign_fuel approach (matching the Rust
    cd_basis_mul_sign implementation) to compute the octonion sign
    table algorithmically, then verifies the associator classification
    on all 210 ordered triples via vm_compute boolean reflection.

    Claims: C-1505 (this formal proof). *)

From Stdlib Require Import ZArith Bool List Arith.
Import ListNotations.
Open Scope Z_scope.

(** * Recursive sign computation (matches cd_kernel::cd_basis_mul_sign).

    cd_sign(fuel, dim, i, j):
      if i = 0 or j = 0: +1
      if i = j: -1
      if i < dim/2 and j < dim/2: recurse on half
      if i < dim/2 and j >= dim/2: negate if j-dim/2 < i, then recurse
      if i >= dim/2 and j < dim/2: negate if j <= i-dim/2, then recurse
      if i >= dim/2 and j >= dim/2: negate, swap, recurse on conjugate *)

(** Recursive sign computation matching cd_kernel::cd_basis_mul_sign exactly.

    Rust logic (signs.rs lines 2-28):
      dim=1: return 1
      p<half, q<half: recurse(half, p, q)
      p<half, q>=half: recurse(half, q-half, p)  -- NOTE: swaps p and q
      p>=half, q<half: s = recurse(half, p-half, q); if q=0 then s else -s
      p>=half, q>=half: if q-half=0 then -1 else recurse(half, q-half, p-half) *)

Fixpoint cd_sign_fuel (fuel : nat) (dim p q : nat) : Z :=
  match fuel with
  | O => 1%Z  (* base: dim=1 *)
  | S fuel' =>
    let half := Nat.div dim 2 in
    if Nat.eqb half 0 then 1%Z
    else if andb (Nat.ltb p half) (Nat.ltb q half) then
      cd_sign_fuel fuel' half p q
    else if andb (Nat.ltb p half) (negb (Nat.ltb q half)) then
      (* p<half, q>=half: recurse(half, q-half, p) -- swap! *)
      cd_sign_fuel fuel' half (q - half)%nat p
    else if andb (negb (Nat.ltb p half)) (Nat.ltb q half) then
      (* p>=half, q<half: s = recurse(half, p-half, q); negate if q!=0 *)
      let s := cd_sign_fuel fuel' half (p - half)%nat q in
      if Nat.eqb q 0 then s else Z.opp s
    else
      (* both >= half *)
      let qh := (q - half)%nat in
      let ph := (p - half)%nat in
      if Nat.eqb qh 0 then (-1)%Z
      else cd_sign_fuel fuel' half qh ph
  end.

(** Octonion sign: use fuel = 4 (log2(8) + 1), dim = 8. *)
Definition oct_sign (i j : nat) : Z := cd_sign_fuel 4 8 i j.

(** Quick sanity checks matching SedenionAlternativityFails.v *)
Example sign_1_2 : oct_sign 1 2 = 1. Proof. vm_compute. reflexivity. Qed.
Example sign_1_3 : oct_sign 1 3 = (-1). Proof. vm_compute. reflexivity. Qed.
Example sign_2_3 : oct_sign 2 3 = 1. Proof. vm_compute. reflexivity. Qed.
Example sign_self : oct_sign 3 3 = (-1). Proof. vm_compute. reflexivity. Qed.

(** * Associator coefficient via sign table.

    [e_i, e_j, e_k] = (e_i * e_j) * e_k - e_i * (e_j * e_k)
    = sign(i XOR j, k) * e_{i XOR j XOR k} - sign(i, j XOR k) * e_{i XOR j XOR k}
    = (sign(i XOR j, k) - sign(i, j XOR k)) * e_{target}

    where target = i XOR j XOR k. *)

Definition assoc_coeff (i j k : nat) : Z :=
  let ij := Nat.lxor i j in
  let jk := Nat.lxor j k in
  oct_sign ij k * oct_sign i j - oct_sign i jk * oct_sign j k.

(** Non-Fano detection via XOR. *)
Definition is_nonfano (i j k : nat) : bool :=
  negb (Nat.eqb (Nat.lxor (Nat.lxor i j) k) 0).

(** Build all 210 ordered triples of distinct elements from {1..7}. *)
Definition all_triples : list (nat * nat * nat) :=
  List.flat_map (fun i =>
    List.flat_map (fun j =>
      if Nat.eqb i j then []
      else List.flat_map (fun k =>
        if orb (Nat.eqb k i) (Nat.eqb k j) then []
        else [(i, j, k)])
        (List.seq 1 7))
      (List.seq 1 7))
    (List.seq 1 7).

(** Associator is nonzero on all 168 non-Fano triples. *)
Definition check_nonfano_nonzero : bool :=
  List.forallb (fun ijk =>
    match ijk with (i, j, k) =>
      if is_nonfano i j k then negb (Z.eqb (assoc_coeff i j k) 0)
      else true
    end) all_triples.

Theorem nonfano_assoc_nonzero : check_nonfano_nonzero = true.
Proof. vm_compute. reflexivity. Qed.

(** Associator is zero on all 42 Fano triples. *)
Definition check_fano_zero : bool :=
  List.forallb (fun ijk =>
    match ijk with (i, j, k) =>
      if negb (is_nonfano i j k) then Z.eqb (assoc_coeff i j k) 0
      else true
    end) all_triples.

Theorem fano_assoc_zero : check_fano_zero = true.
Proof. vm_compute. reflexivity. Qed.

(** Associator magnitude is exactly 2 on all non-Fano triples.
    This means m3 = +/-2 * e_{i XOR j XOR k}. *)
Definition check_nonfano_abs2 : bool :=
  List.forallb (fun ijk =>
    match ijk with (i, j, k) =>
      if is_nonfano i j k then
        orb (Z.eqb (assoc_coeff i j k) 2)
            (Z.eqb (assoc_coeff i j k) (-2))
      else true
    end) all_triples.

Theorem nonfano_assoc_is_pm2 : check_nonfano_abs2 = true.
Proof. vm_compute. reflexivity. Qed.

(** * Summary:
    1. Associator = 0 on all 42 Fano-line triples.
    2. Associator != 0 on all 168 non-Fano triples.
    3. |Associator| = 2 on all non-Fano triples.
    All verified by boolean reflection via vm_compute. *)
