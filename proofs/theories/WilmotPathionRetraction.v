(** * WilmotPathionRetraction: retraction m3 classification on {1..15}.

    Computes the homotopy transfer m3 from the sedenion retraction
    p: S -> O (project out upper octonion half) on all C(15,3) = 455
    unordered triples of distinct imaginary sedenion units.

    The retraction m3 on basis triples:
      m3(e_i, e_j, e_k) = p(e_i * e_j * e_k) - p(e_i) * p(e_j * e_k)
                         - p(e_i * e_j) * p(e_k) + p(e_i) * p(e_j) * p(e_k)

    Since p(e_i) = e_i if i < 8, else 0, and each product is
    sign * e_{XOR}, the entire computation is Z-valued.

    Expected Wilmot classification (Table 2):
      U_1 = 35 (scalar/associative triads)
      U_2 = 252 (mixed type)
      U_3 = 168 (fully non-associative, m3 = Assoc)
      Total: 35 + 252 + 168 = 455

    Claim: C-1508. *)

From Stdlib Require Import ZArith Bool List Arith.
Import ListNotations.
Open Scope Z_scope.

Require Import CDProofs.M3IsAssociator.

(** Sedenion sign: fuel=5, dim=16. *)
Definition sed_sign (i j : nat) : Z := cd_sign_fuel 5 16 i j.

(** Project: keep index if < 8, else 0. Returns (index, coefficient).
    A sedenion basis element e_i projects to:
      p(e_i) = e_i  if i < 8
      p(e_i) = 0    if i >= 8 *)
Definition proj_index (i : nat) : bool := Nat.ltb i 8.

(** Sedenion basis product: e_i * e_j = sign(i,j) * e_{i XOR j}.
    Returns (result_index, sign_coefficient). *)
Definition sed_mul_basis (i j : nat) : (nat * Z) :=
  if Nat.eqb i 0 then (j, 1)
  else if Nat.eqb j 0 then (i, 1)
  else (Nat.lxor i j, sed_sign i j).

(** Triple product e_i * e_j * e_k = (e_i * e_j) * e_k.
    Returns (result_index, sign_coefficient). *)
Definition sed_mul3_basis (i j k : nat) : (nat * Z) :=
  let '(ij_idx, ij_sign) := sed_mul_basis i j in
  let '(ijk_idx, ijk_sign) := sed_mul_basis ij_idx k in
  (ijk_idx, ij_sign * ijk_sign).

(** Retraction m3 on basis triples.

    m3(e_i, e_j, e_k) has 4 terms. Each term is p(product) where
    p projects to the lower octonion (index < 8).

    Term 1: p((e_i * e_j) * e_k) = sign * e_target if target < 8
    Term 2: -p(e_i) * p(e_j * e_k)
    Term 3: -p(e_i * e_j) * p(e_k)
    Term 4: +p(e_i) * p(e_j) * p(e_k)

    Since each projected element is either e_idx (with sign) or 0,
    the result is a linear combination of basis elements.

    We compute the coefficient on e_target where target = i XOR j XOR k
    (the XOR of all three). If the result is nonzero, m3 is nontrivial.

    Actually, m3 can land on ANY basis element, not just e_{i^j^k}.
    So we compute the full 8-component vector and check for nonzero. *)

(** Coefficient of e_t in p(product), where product = (idx, sign).
    p(idx, sign) contributes sign * delta(t, idx) if idx < 8. *)
Definition proj_coeff (t : nat) (idx : nat) (s : Z) : Z :=
  if andb (Nat.ltb idx 8) (Nat.eqb t idx) then s else 0.

(** m3 coefficient on basis element e_t for triple (i,j,k).

    m3(e_i, e_j, e_k)[t] =
      p((e_i*e_j)*e_k)[t]
      - (p(e_i) * p(e_j*e_k))[t]
      - (p(e_i*e_j) * p(e_k))[t]
      + (p(e_i) * p(e_j) * p(e_k))[t]

    For projected products:
      p(e_a) * p(e_b) is nonzero only if both a,b < 8,
      and equals sign(a,b) * e_{a XOR b}.
    Similarly for triple products. *)
Definition m3_coeff (i j k t : nat) : Z :=
  (* Term 1: p((e_i * e_j) * e_k) *)
  let '(ijk_idx, ijk_s) := sed_mul3_basis i j k in
  let t1 := proj_coeff t ijk_idx ijk_s in

  (* Term 2: -p(e_i) * p(e_j * e_k) *)
  let '(jk_idx, jk_s) := sed_mul_basis j k in
  let t2 := if andb (proj_index i) (proj_index jk_idx) then
    let '(prod_idx, prod_s) := sed_mul_basis i jk_idx in
    proj_coeff t prod_idx (jk_s * prod_s)
  else 0 in

  (* Term 3: -p(e_i * e_j) * p(e_k) *)
  let '(ij_idx, ij_s) := sed_mul_basis i j in
  let t3 := if andb (proj_index ij_idx) (proj_index k) then
    let '(prod_idx, prod_s) := sed_mul_basis ij_idx k in
    proj_coeff t prod_idx (ij_s * prod_s)
  else 0 in

  (* Term 4: +p(e_i) * p(e_j) * p(e_k) *)
  let t4 := if andb (proj_index i) (andb (proj_index j) (proj_index k)) then
    let '(ij_idx2, ij_s2) := sed_mul_basis i j in
    let '(ijk_idx2, ijk_s2) := sed_mul_basis ij_idx2 k in
    proj_coeff t ijk_idx2 (ij_s2 * ijk_s2)
  else 0 in

  t1 - t2 - t3 + t4.

(** Check if m3 is zero on a triple (i,j,k): all 8 components vanish. *)
Definition m3_is_zero (i j k : nat) : bool :=
  List.forallb (fun t => Z.eqb (m3_coeff i j k t) 0)
    (List.seq 0 8).

(** Build all UNORDERED triples from {1..15}: C(15,3) = 455. *)
Definition unordered_sed_triples : list (nat * nat * nat) :=
  List.flat_map (fun i =>
    List.flat_map (fun j =>
      List.flat_map (fun k =>
        [(i, j, k)])
        (List.seq (j + 1) (15 - j)))
      (List.seq (i + 1) (15 - i)))
    (List.seq 1 15).

(** Total count should be 455. *)
Definition unordered_count : nat := List.length unordered_sed_triples.
Theorem unordered_is_455 : unordered_count = 455%nat.
Proof. vm_compute. reflexivity. Qed.

(** Wilmot U_1: m3 = 0 (scalar/associative triads). *)
Definition wilmot_u1_count : nat :=
  List.length (List.filter (fun ijk =>
    match ijk with (i, j, k) => m3_is_zero i j k end)
    unordered_sed_triples).

(** Wilmot U_3: m3 != 0 (non-trivial homotopy transfer). *)
Definition wilmot_u3_count : nat :=
  List.length (List.filter (fun ijk =>
    match ijk with (i, j, k) => negb (m3_is_zero i j k) end)
    unordered_sed_triples).

(** NOTE: The m3 formula used above is the NAIVE associator-based version:
    m3(e_i,e_j,e_k) = (e_i*e_j)*e_k - e_i*(e_j*e_k) projected to lower 8.

    The CORRECT homotopy transfer m3 (from Rust test_homotopy_transfer_m3)
    uses a DIFFERENT formula with the fold-average retraction:
      i(x) = (x, x)           -- diagonal section
      p(u,v) = (u+v)/2        -- fold-average projection
      h = id - i*p             -- homotopy
      m3(x,y,z) = p(h(i(x)*i(y))*i(z)) - p(i(x)*h(i(y)*i(z)))

    This involves rational (1/2) coefficients from the averaging, making
    pure Z-arithmetic insufficient.  The correct implementation would
    require either:
    (a) Q-arithmetic in Rocq (feasible but heavier infrastructure), or
    (b) Scaling by 4 to clear denominators (m3 * 4 is Z-valued).

    The Wilmot U_1/U_2/U_3 classification is verified computationally
    in Rust (C-1465, test_pathion_retraction_m3) but not yet formalized
    in Rocq due to the rational coefficient issue.

    What IS verified here:
    - Total unordered triples: 455 = C(15,3)
    - The naive associator-based classification (sign-table) gives
      a DIFFERENT decomposition than Wilmot's retraction m3
    - The discrepancy is documented in C8b (35+112+308 vs 35+252+168) *)

(** We CAN verify the naive m3 counts. *)
Theorem naive_m3_zero_count :
  wilmot_u1_count = 343%nat.
Proof. vm_compute. reflexivity. Qed.

Theorem naive_m3_nonzero_count :
  wilmot_u3_count = 112%nat.
Proof. vm_compute. reflexivity. Qed.

Theorem naive_partition :
  (wilmot_u1_count + wilmot_u3_count)%nat = 455%nat.
Proof. vm_compute. reflexivity. Qed.

(** So the naive (associator-projected) classification gives:
    343 zero + 112 nonzero = 455.
    This differs from BOTH:
    - Wilmot retraction m3: 35 + 420 = 455 (from Rust)
    - Sign-table associator (ordered, C8b): 35 + 112 + 308 = 455

    The 343 = 7^3 is suggestive: it may count triples where all
    three pairwise products land in the lower octonion (index < 8),
    making the projection trivial. Investigation deferred. *)
