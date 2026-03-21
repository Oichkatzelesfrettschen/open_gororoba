(** * C-1463: Arithmetic Inventory of Sedenion Structure Constants.

    Kernel-checked arithmetic identities that anchor the combinatorial
    structure of the sedenion algebra.  All proofs use vm_compute
    (concrete evaluation, no symbolic ring), so they compile in
    milliseconds and use negligible memory.

    These facts are the formal backbone for:
    - Wilmot Table 2 (U_1 triad counts)
    - ZD pair / assessor counting
    - S3 orbit decomposition
    - Quaternion subalgebra counting (H_n formula)

    Mirrors: crates/algebra_experimental/src/sedenion_subalgebras.rs *)

From Stdlib Require Import Arith PeanoNat.

(** * Wilmot Table 2 for U_1 (sedenions): 35 + 60 + 360 = 455. *)

Theorem wilmot_table2_u1_total : 35 + 60 + 360 = 455.
Proof. vm_compute. reflexivity. Qed.

(** C(15,3) = 455 total triads of 15 imaginary sedenion basis elements. *)
Theorem c15_3 : 15 * 14 * 13 / 6 = 455.
Proof. vm_compute. reflexivity. Qed.

(** * Quaternion subalgebra count H_n = N_n*(N_n-1)/6 at level 4. *)

Theorem h_15_equals_35 : 15 * 14 / 6 = 35.
Proof. vm_compute. reflexivity. Qed.

(** At level 5 (trigintaduonions): H_31 = 31*30/6 = 155. *)
Theorem h_31_equals_155 : 31 * 30 / 6 = 155.
Proof. vm_compute. reflexivity. Qed.

(** Wilmot Table 2 for U_2: 155 + 620 + 3720 = 4495. *)
Theorem wilmot_table2_u2_total : 155 + 620 + 3720 = 4495.
Proof. vm_compute. reflexivity. Qed.

(** C(31,3) = 4495 for U_2. *)
Theorem c31_3 : 31 * 30 * 29 / 6 = 4495.
Proof. vm_compute. reflexivity. Qed.

(** * Zero-divisor pair counts. *)

(** 42 primitive assessors: 7 low indices x (7 high indices - 1 excluded). *)
Theorem assessor_count_42 : 7 * 6 = 42.
Proof. vm_compute. reflexivity. Qed.

(** 84 standard ZDs = 42 assessors x 2 signs. *)
Theorem standard_zd_count : 42 * 2 = 84.
Proof. vm_compute. reflexivity. Qed.

(** 168 directed ZD pairs = 84 ZDs x 2 directions (or 42 x 4 partners). *)
Theorem directed_zd_pairs : 84 * 2 = 168.
Proof. vm_compute. reflexivity. Qed.

(** GL(3,2) order = 168 (the automorphism group of the Fano plane). *)
Theorem gl32_order : (8 - 1) * (8 - 2) * (8 - 4) = 168.
Proof. vm_compute. reflexivity. Qed.

(** * S3 orbit decomposition of signed friction. *)

(** 105 = C(15,2) braid pairs of imaginary sedenion basis elements. *)
Theorem braid_pair_count : 15 * 14 / 2 = 105.
Proof. vm_compute. reflexivity. Qed.

(** S3 orbit decomposition: 21 + 36 + 48 = 105 (interleaved stride). *)
Theorem s3_orbit_decomposition : 21 + 36 + 48 = 105.
Proof. vm_compute. reflexivity. Qed.

(** * Non-associativity type decomposition: 35 + 84 + 84 + 252 = 455. *)

Theorem triad_type_decomposition : 35 + 84 + 84 + 252 = 455.
Proof. vm_compute. reflexivity. Qed.

(** Type X = 3 * Type B = 3 * Type C. *)
Theorem type_x_is_3_times_bc : 252 = 3 * 84.
Proof. vm_compute. reflexivity. Qed.

(** * Incidence matrix rank structure: 21 + 6 = 27. *)

Theorem incidence_rank_containment : 21 + 6 = 27.
Proof. vm_compute. reflexivity. Qed.

(** 21 = C(7,2): pair-space of 7-dimensional octonionic scaffold. *)
Theorem rank_bc_is_c72 : 7 * 6 / 2 = 21.
Proof. vm_compute. reflexivity. Qed.

(** * Gresnigt S3 orbit sum factor. *)

(** The orbit sum factor 3/2 arises from 1 + 1/4 + 1/4 = 3/2.
    In natural number arithmetic: 4 + 1 + 1 = 6 and 6/4 = 3/2. *)
Theorem orbit_sum_numerator : 4 + 1 + 1 = 6.
Proof. vm_compute. reflexivity. Qed.

(** The partner coefficient: 0 + 3 + 3 = 6 (same factor). *)
Theorem orbit_partner_numerator : 0 + 3 + 3 = 6.
Proof. vm_compute. reflexivity. Qed.

(** * CD tower dimensions. *)

Theorem sedenion_dim : 2^4 = 16.
Proof. vm_compute. reflexivity. Qed.

Theorem cl8_dim : 2^8 = 256.
Proof. vm_compute. reflexivity. Qed.

Theorem cl8_is_mat16 : 256 = 16 * 16.
Proof. vm_compute. reflexivity. Qed.

(** Minimal left ideal dimension = sedenion dimension (Bott periodicity). *)
Theorem minimal_ideal_dim : 256 / 16 = 16.
Proof. vm_compute. reflexivity. Qed.
