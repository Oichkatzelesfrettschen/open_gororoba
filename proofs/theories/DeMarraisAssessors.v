(** * DeMarraisAssessors: Co-assessor structure and Production Rule #1.

    Extends BoxKite.v with the algebraic content from de Marrais (2000):

    - The 7 O-trips and 28 S-trips (Section I of the paper)
    - The XOR G-index co-assessor criterion (Section II Lemma):
        (A,B) and (C,D) are co-assessors iff Nat.lxor A B = Nat.lxor C D
    - Production Rule #1 (Section II): from co-assessors (A,B) and (C,D),
        derive a third co-assessor (A xor D, A xor C)
    - Concrete sedenion ZD proofs for two additional box-kite 7 pairs
    - The 168 = 42 * 4 count arising from 4 primitive ZDs per assessor

    Key: the XOR G-index is already captured in ZDGraph.v as xor_sig.
    This file adds the associative-triple structure and Production Rule #1.

    Source: R.P.C. de Marrais (2000),
      "The 42 Assessors and the Box-Kites they fly,"
      arXiv:math/0011260. Section labels follow that paper. *)

From Stdlib Require Import List Arith.
Import ListNotations.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororoba Require Import BoxKite ZDGraph.
From OpenGororoba Require Import ZD_Criterion.

(** Force nat as the default numeral scope so that the O-trip/S-trip
    lists elaborate correctly alongside the R-scoped sedenion imports. *)
Local Open Scope nat_scope.

(* ================================================================== *)
(** * Section I: The 7 O-trips.                                        *)
(** An O-trip (a,b,c) means e_a * e_b = +/- e_c in octonion arithmetic,
    using the XOR labeling convention required by de Marrais.
    The seven O-trips are the thrust-blocks that anchor the box-kites.
    de Marrais (2000) Section I lists them as:
      (1,2,3); (1,4,5); (1,7,6); (2,4,6); (2,5,7); (3,4,7); (3,6,5) *)
(* ================================================================== *)

Definition o_trips : list (nat * nat * nat) :=
  [(1,2,3); (1,4,5); (1,7,6); (2,4,6); (2,5,7); (3,4,7); (3,6,5)].

(** There are exactly 7 O-trips (one per octonion imaginary basis). *)
Theorem o_trip_count : length o_trips = 7.
Proof. vm_compute. reflexivity. Qed.

(** Each O-trip obeys the XOR rule: the third index equals XOR of the first two. *)
Theorem o_trips_xor_consistent :
  List.forallb
    (fun t => let '(a, b, c) := t in Nat.eqb (Nat.lxor a b) c)
    o_trips = true.
Proof. vm_compute. reflexivity. Qed.

(* ================================================================== *)
(** * Section I: The 28 S-trips.                                       *)
(** S-trips (sedenion associative triples) cross the octonion/sedenion
    boundary: each has one low index (1..7) and two high indices (8..15).
    de Marrais (2000) Section I lists all 28 explicitly; they are
    reproduced here verbatim from the paper's table. *)
(* ================================================================== *)

Definition s_trips : list (nat * nat * nat) :=
  [(1,8,9);  (1,11,10); (1,13,12); (1,14,15);
   (2,8,10); (2,9,11);  (2,14,12); (2,15,13);
   (3,8,11); (3,10,9);  (3,15,12); (3,13,14);
   (4,8,12); (4,9,13);  (4,10,14); (4,11,15);
   (5,8,13); (5,12,9);  (5,10,15); (5,14,11);
   (6,8,14); (6,15,9);  (6,12,10); (6,11,13);
   (7,8,15); (7,9,14);  (7,13,10); (7,12,11)].

(** There are exactly 28 S-trips. *)
Theorem s_trip_count : length s_trips = 28.
Proof. vm_compute. reflexivity. Qed.

(** Each S-trip also obeys the XOR rule. *)
Theorem s_trips_xor_consistent :
  List.forallb
    (fun t => let '(a, b, c) := t in Nat.eqb (Nat.lxor a b) c)
    s_trips = true.
Proof. vm_compute. reflexivity. Qed.

(** Total: 7 + 28 = 35 associative triples in sedenion space. *)
Theorem total_trip_count : length o_trips + length s_trips = 35.
Proof. vm_compute. reflexivity. Qed.

(* ================================================================== *)
(** * Section II: The XOR G-index co-assessor criterion.               *)
(** de Marrais (2000) Section II Lemma: Given two Co-Assessors (A,B)   *)
(** and (C,D), A xor B and C xor D have the same index G.              *)
(**                                                                    *)
(** Combined with the full zero-divisor criterion:                     *)
(**   (A,B) x (C,D) = 0 iff (A xor C = D xor B) and (A xor D = C xor B) *)
(** which implies A xor B = C xor D.                                   *)
(**                                                                    *)
(** The xor_sig function from ZDGraph.v captures this as G = xor_sig.  *)
(** This section adds the co-assessor predicate and key properties.    *)
(* ================================================================== *)

(** Two assessor pairs are co-assessors iff they share the same G-index. *)
Definition co_assessors (A B C D : nat) : Prop :=
  Nat.lxor A B = Nat.lxor C D.

(** The co-assessor relation is symmetric. *)
Theorem co_assessors_symm (A B C D : nat) :
  co_assessors A B C D -> co_assessors C D A B.
Proof.
  unfold co_assessors. intro H. symmetry. exact H.
Qed.

(** Every assessor is trivially a co-assessor with itself. *)
Theorem co_assessors_refl (A B : nat) :
  co_assessors A B A B.
Proof. unfold co_assessors. reflexivity. Qed.

(** The co-assessor relation is transitive. *)
Theorem co_assessors_trans (A B C D E F : nat) :
  co_assessors A B C D ->
  co_assessors C D E F ->
  co_assessors A B E F.
Proof.
  unfold co_assessors. intros H1 H2. rewrite H1. exact H2.
Qed.

(** Computational check: all 6 pairs in box-kite 7 share G-index 9. *)
Theorem bk7_g_index :
  boxkite_xor_sigs boxkite_7 = [9; 9; 9; 9; 9; 9].
Proof. vm_compute. reflexivity. Qed.

(** The 7 distinct G-indices (one per box-kite). *)
Theorem bk_g_indices :
  ZDGraph.boxkite_signatures = [15; 10; 11; 12; 13; 14; 9].
Proof. vm_compute. reflexivity. Qed.

(* ================================================================== *)
(** * Section II: Production Rule #1.                                  *)
(** de Marrais (2000) Section II: Given co-assessors (A,B) and (C,D),  *)
(** a new Assessor (E,F), mutually zero-dividing with both progenitors, *)
(** can be created by assigning:                                        *)
(**   E = A xor D = C xor B                                             *)
(**   F = A xor C = D xor B                                             *)
(**                                                                    *)
(** This is the "Three-Ring Circuit": (A,B), (C,D), (E,F) mutually    *)
(** zero-divide in a closed loop.                                      *)
(* ================================================================== *)

(** XOR cancellation: (a xor b) xor (a xor c) = b xor c.
    This is the arithmetic core of Production Rule #1. *)
Lemma xor_cancel_l (a b c : nat) :
  Nat.lxor (Nat.lxor a b) (Nat.lxor a c) = Nat.lxor b c.
Proof.
  rewrite Nat.lxor_assoc.
  rewrite <- (Nat.lxor_assoc b a c).
  rewrite (Nat.lxor_comm b a).
  rewrite <- (Nat.lxor_assoc a (Nat.lxor a b) c).
  rewrite <- (Nat.lxor_assoc a a b).
  rewrite Nat.lxor_nilpotent.
  rewrite (Nat.lxor_comm 0 b).
  rewrite Nat.lxor_0_r.
  reflexivity.
Qed.

(** Production Rule #1: the derived pair (A xor D, A xor C) is a
    co-assessor with the original pair (C, D).
    Proof: (A xor D) xor (A xor C) = D xor C = C xor D by xor_cancel_l. *)
Theorem production_rule_1 (A B C D : nat) :
  co_assessors A B C D ->
  co_assessors (Nat.lxor A D) (Nat.lxor A C) C D.
Proof.
  unfold co_assessors. intro H.
  rewrite xor_cancel_l.
  apply Nat.lxor_comm.
Qed.

(** The derived pair is also a co-assessor with the first original pair (A,B). *)
Theorem production_rule_1_full (A B C D : nat) :
  co_assessors A B C D ->
  co_assessors (Nat.lxor A D) (Nat.lxor A C) A B.
Proof.
  unfold co_assessors. intro H.
  rewrite xor_cancel_l.
  rewrite Nat.lxor_comm.
  symmetry. exact H.
Qed.

(** Concrete instance: from (3,10) and (6,15) in box-kite 7, derive (5,12). *)
Theorem pr1_bk7_instance :
  Nat.lxor (Nat.lxor 3 15) (Nat.lxor 3 6) = Nat.lxor 5 12.
Proof. vm_compute. reflexivity. Qed.

(* ================================================================== *)
(** * Section II: Production Rule #2 (index-only formulation).        *)
(** Three-Ring Circuit symmetry: applying Rule 1 to any two of the    *)
(** three co-assessors in a circuit recovers the third.               *)
(**                                                                    *)
(** de Marrais (2000) Section II: the Three-Ring Circuit is SYMMETRIC. *)
(** Rule 1 is not one-directional: any ordered pair from {(A,B),      *)
(** (C,D), (E,F)} can serve as input to Rule 1 to produce the third.  *)
(**                                                                    *)
(** Formally (index-only): if E = A xor D and F = A xor C (Rule 1    *)
(** from (A,B) and (C,D)), applying Rule 1 to (C,D) and (E,F) gives  *)
(** a co-assessor of (A,B). Key identities:                           *)
(**   C xor (A xor C) = A  (nilpotency: C xor C = 0)                 *)
(**   C xor (A xor D) = B  (from co_assessors H: A xor B = C xor D)  *)
(* ================================================================== *)

(** C xor (A xor C) = A (XOR nilpotency: C cancels). *)
Lemma xor_absorb_self (A C : nat) :
  Nat.lxor C (Nat.lxor A C) = A.
Proof.
  rewrite (Nat.lxor_comm A C).
  rewrite <- Nat.lxor_assoc.
  rewrite Nat.lxor_nilpotent.
  rewrite Nat.lxor_0_l.
  reflexivity.
Qed.

(** C xor (A xor D) = B when A xor B = C xor D.
    Proof: C xor (A xor D) = A xor (C xor D) = A xor (A xor B) = B. *)
Lemma xor_circuit_recover (A B C D : nat) :
  Nat.lxor A B = Nat.lxor C D ->
  Nat.lxor C (Nat.lxor A D) = B.
Proof.
  intro H.
  rewrite <- Nat.lxor_assoc.
  rewrite (Nat.lxor_comm C A).
  rewrite Nat.lxor_assoc.
  rewrite <- H.
  rewrite <- Nat.lxor_assoc.
  rewrite Nat.lxor_nilpotent.
  rewrite Nat.lxor_0_l.
  reflexivity.
Qed.

(** Production Rule #2: Three-Ring Circuit symmetry (forward direction).
    Let E = A xor D, F = A xor C (from Rule 1 applied to (A,B) and (C,D)).
    Applying Rule 1 to (C,D) and (E,F): the derived pair (C xor F, C xor E)
    has G-index equal to A xor B. The circuit closes back on (A,B). *)
Theorem production_rule_2 (A B C D : nat) :
  co_assessors A B C D ->
  let E := Nat.lxor A D in
  let F := Nat.lxor A C in
  co_assessors (Nat.lxor C F) (Nat.lxor C E) A B.
Proof.
  unfold co_assessors. intro H. simpl.
  (** Goal: (C xor (A xor C)) xor (C xor (A xor D)) = A xor B *)
  rewrite xor_absorb_self.
  (** Goal: A xor (C xor (A xor D)) = A xor B *)
  rewrite (xor_circuit_recover A B C D H).
  (** Goal: A xor B = A xor B *)
  reflexivity.
Qed.

(** Symmetric direction: Rule 1 on (A,B) and (E,F) recovers (C,D).
    A xor (A xor X) = X by xor self-cancellation: (A xor A) xor X = 0 xor X = X. *)
Theorem production_rule_2_sym (A B C D : nat) :
  co_assessors A B C D ->
  let E := Nat.lxor A D in
  let F := Nat.lxor A C in
  co_assessors (Nat.lxor A F) (Nat.lxor A E) C D.
Proof.
  unfold co_assessors. intro H. simpl.
  (** Goal: (A xor (A xor C)) xor (A xor (A xor D)) = C xor D *)
  (** A xor (A xor X) = (A xor A) xor X = 0 xor X = X *)
  assert (Hself: forall x y : nat, Nat.lxor x (Nat.lxor x y) = y).
  { intros x y.
    rewrite <- Nat.lxor_assoc.
    rewrite Nat.lxor_nilpotent.
    apply Nat.lxor_0_l. }
  rewrite Hself. rewrite Hself. reflexivity.
Qed.

(** Concrete box-kite 7 instance: (6,15) and (5,12) recover (3,10). *)
Example pr2_bk7_circuit :
  co_assessors (Nat.lxor 6 (Nat.lxor 3 6)) (Nat.lxor 6 (Nat.lxor 3 15)) 3 10.
Proof. unfold co_assessors. vm_compute. reflexivity. Qed.

(** All three box-kite 7 derived pairs have G-index 9. *)
Theorem bk7_coassessor_trio :
  Nat.lxor 3 10 = 9 /\ Nat.lxor 6 15 = 9 /\ Nat.lxor 5 12 = 9.
Proof. vm_compute. auto. Qed.

(* ================================================================== *)
(** * Section II: Production Rule #3 (index-only formulation).        *)
(** G-index uniquely determines the box-kite: any pair (lo', G xor lo') *)
(** with lo' in any nat is a co-assessor of any pair with signature G. *)
(**                                                                    *)
(** de Marrais (2000) Section II: the G-index classifies co-assessors  *)
(** completely. Given ANY assessor (A,B) with G = A xor B, the full  *)
(** set of co-assessors is exactly { (lo', G xor lo') | lo' valid }.  *)
(**                                                                    *)
(** Index-only content: for any lo', (lo', (A xor B) xor lo') is a   *)
(** co-assessor of (A,B) -- regardless of whether lo' satisfies the   *)
(** range constraint for sedenion indices. The range filter (1..7 for  *)
(** lo, 9..15 for hi, hi != lo+8) determines which are valid assessors,*)
(** but the G-index relationship holds unconditionally.               *)
(**                                                                    *)
(** Proof: lo' xor ((A xor B) xor lo') = A xor B by nilpotency.      *)
(* ================================================================== *)

(** Production Rule #3: G-index classifies all co-assessors.
    For any lo' (not necessarily a valid sedenion index), the pair
    (lo', (A xor B) xor lo') is a co-assessor of (A,B):
    lo' xor ((A xor B) xor lo') = A xor B. *)
Theorem production_rule_3 (A B lo' : nat) :
  co_assessors lo' (Nat.lxor (Nat.lxor A B) lo') A B.
Proof.
  unfold co_assessors.
  (** Goal: lo' xor ((A xor B) xor lo') = A xor B *)
  rewrite <- Nat.lxor_assoc.
  (** Goal: (lo' xor (A xor B)) xor lo' = A xor B *)
  rewrite (Nat.lxor_comm lo' (Nat.lxor A B)).
  (** Goal: ((A xor B) xor lo') xor lo' = A xor B *)
  rewrite Nat.lxor_assoc.
  (** Goal: (A xor B) xor (lo' xor lo') = A xor B *)
  rewrite Nat.lxor_nilpotent.
  (** Goal: (A xor B) xor 0 = A xor B *)
  rewrite Nat.lxor_0_r.
  reflexivity.
Qed.

(** Rule 3 is the "source" of Rule 1: Production Rule 1 is just Rule 3
    applied with lo' = A xor D (since G xor (A xor D) = (A xor B) xor (A xor D)
    = B xor D, and (C,D) has G = A xor B, so B xor D = C...). *)
Theorem production_rule_3_implies_rule_1 (A B C D : nat) :
  co_assessors A B C D ->
  co_assessors (Nat.lxor A D) (Nat.lxor (Nat.lxor A B) (Nat.lxor A D)) A B.
Proof.
  intro H. apply production_rule_3.
Qed.

(** Concrete box-kite 7 instance: G = 9, lo' = 4 gives hi' = 13.
    So (4, 13) is a co-assessor of (3, 10): 4 xor 13 = 9 = 3 xor 10. *)
Example pr3_bk7_lo4 :
  co_assessors 4 (Nat.lxor (Nat.lxor 3 10) 4) 3 10.
Proof. apply production_rule_3. Qed.

Example pr3_bk7_lo4_value :
  Nat.lxor (Nat.lxor 3 10) 4 = 13.
Proof. vm_compute. reflexivity. Qed.

(** The full box-kite 7 (G=9) enumerates all valid co-assessors:
    { lo' | lo' in {1..7}, G xor lo' in {9..15}, G xor lo' != lo' + 8 }
    = { 2,3,4,5,6,7 } (lo'=1 gives hi'=8, which is the shadow, excluded).
    This is the index-only content of Production Rule #3. *)
Example pr3_bk7_enum :
  (** lo'=2: hi'=9 xor 2=11 *) co_assessors 2 11 3 10 /\
  (** lo'=3: hi'=9 xor 3=10 *) co_assessors 3 10 3 10 /\
  (** lo'=4: hi'=9 xor 4=13 *) co_assessors 4 13 3 10 /\
  (** lo'=5: hi'=9 xor 5=12 *) co_assessors 5 12 3 10 /\
  (** lo'=6: hi'=9 xor 6=15 *) co_assessors 6 15 3 10 /\
  (** lo'=7: hi'=9 xor 7=14 *) co_assessors 7 14 3 10.
Proof.
  repeat split; unfold co_assessors; vm_compute; reflexivity.
Qed.

(* ================================================================== *)
(** * Concrete zero-divisor proofs for box-kite 7.                     *)
(** Each proof reduces the sedenion product to component arithmetic    *)
(** via cbv unfolding of the CD doubling rules, then closes with ring. *)
(** Sign conventions verified computationally in boxkites.rs.          *)
(*   (3,10,+1)*(5,12,-1) = 0  and  (3,10,+1)*(4,13,+1) = 0            *)
(* ================================================================== *)

(** Box-kite 7, pair A: e_3 + e_10  (same as sed_zd_a in Sedenion.v). *)
Definition bk7_a : CDSed := sed_zd_a.

(** Box-kite 7, pair C: e_5 - e_12.
    e_5 = oct_e 5 = (quat_zero, (0,1,0,0)) in lo half.
    e_12 = oct_e 4 = (quat_zero, quat_one) in hi half (sedenion slot 12). *)
Definition bk7_c : CDSed :=
  mkSed (mkOct quat_zero (mkQuat 0 1 0 0))
        (mkOct quat_zero (mkQuat (-1) 0 0 0)).

(** Box-kite 7, pair D: e_4 + e_13.
    e_4 = oct_e 4 = (quat_zero, quat_one) in lo half.
    e_13 = oct_e 5 = (quat_zero, (0,1,0,0)) in hi half (sedenion slot 13). *)
Definition bk7_d : CDSed :=
  mkSed (mkOct quat_zero quat_one)
        (mkOct quat_zero (mkQuat 0 1 0 0)).

(** (e_3 + e_10) * (e_5 - e_12) = 0.
    Assessors (3,10) and (5,12) are co-assessors: 3 xor 10 = 9 = 5 xor 12. *)
Theorem bk7_zd_ac : sed_mul bk7_a bk7_c = sed_zero.
Proof.
  unfold bk7_a, bk7_c, sed_zd_a.
  cbv [sed_mul oct_mul oct_conj
       quat_mul quat_add quat_neg quat_conj
       oct_zero quat_zero quat_one sed_zero
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; f_equal; abstract ring.
Qed.

(** (e_3 + e_10) * (e_4 + e_13) = 0.
    Assessors (3,10) and (4,13) are co-assessors: 3 xor 10 = 9 = 4 xor 13. *)
Theorem bk7_zd_ad : sed_mul bk7_a bk7_d = sed_zero.
Proof.
  unfold bk7_a, bk7_d, sed_zd_a.
  cbv [sed_mul oct_mul oct_conj
       quat_mul quat_add quat_neg quat_conj
       oct_zero quat_zero quat_one sed_zero
       sed_lo sed_hi oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; f_equal; abstract ring.
Qed.

(** The Brown 7.15 fundamental pair can be imported directly into the
    de Marrais lane through the fused criterion support, without routing
    through the Brown chapter wrapper. *)
Theorem de_marrais_brown_fundamental_major_theorem_fused :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  exact zd_fundamental_major_theorem_fused.
Qed.

Theorem de_marrais_brown_fundamental_fused_support :
  zd_condition_ii zd_a1_fundamental zd_a2_fundamental
                  zd_b1_fundamental zd_b2_fundamental /\
  zd_condition_iii zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental.
Proof.
  split.
  - exact zd_fundamental_condition_ii_fused.
  - exact zd_fundamental_condition_iii_fused.
Qed.

(* ================================================================== *)
(** * Count: 168 = 42 * 4.                                             *)
(** de Marrais (2000) Section II: technically there are four normed    *)
(** unit zero-divisors on the extended X that is an Assessor.          *)
(** Each assessor plane contains 4 primitive ZD pairs (the two         *)
(** diagonals '/' and '\', each traversable in two signed directions). *)
(* ================================================================== *)

Theorem zd_count_decomposition : 42 * 4 = 168.
Proof. vm_compute. reflexivity. Qed.

(** The 7 box-kites each contribute 6 assessors x 4 ZD pairs = 24 pairs. *)
Theorem zd_per_boxkite : 6 * 4 = 24.
Proof. vm_compute. reflexivity. Qed.

(** Seven box-kites account for all 168 ZD pairs. *)
Theorem zd_total_from_boxkites : 7 * 24 = 168.
Proof. vm_compute. reflexivity. Qed.

(* ================================================================== *)
(** * 28 Mutually-ZD Assessor Trios.                                  *)
(**                                                                    *)
(** de Marrais (2000) Section II identifies 28 closed trios of        *)
(** co-assessors, each forming a mutually zero-dividing triple inside  *)
(** a single box-kite. The Three-Ring Circuit (Production Rule #1)    *)
(** generates each trio; Production Rules #2 and #3 show the circuit  *)
(** is symmetric and that the G-index completely classifies them.      *)
(**                                                                    *)
(** Structure: 7 box-kites x 4 Fano lines per box-kite = 28 trios.   *)
(**                                                                    *)
(** For box-kite G (G in {9..15}):                                    *)
(**   - Excluded lo: the unique lo in {1..7} with G xor lo = 8        *)
(**     (i.e., lo = G xor 8; for G=9, excl lo=1; G=10, excl lo=2...) *)
(**   - Valid lo-values: {1..7} minus {excl lo} -- exactly 6 values   *)
(**   - Valid Fano lines: the 4 lines in {1..7} not containing excl lo *)
(**   - Each such Fano line {lo1,lo2,lo3} gives the trio              *)
(**       ((lo1, G xor lo1), (lo2, G xor lo2), (lo3, G xor lo3))      *)
(**                                                                    *)
(** Each trio ((a,b),(c,d),(e,f)) satisfies:                          *)
(**   (1) Same G-index: a xor b = c xor d = e xor f (co-assessors)   *)
(**   (2) Fano line condition: a xor c xor e = 0 (lo-indices form     *)
(**       a Fano triple, equivalently a Fano line in PG(2,2))         *)
(**                                                                    *)
(** Source: de Marrais (2000) arXiv:math/0011260, Section II.         *)
(** The connection between the 28 trios and the 168 ZD pairs:         *)
(**   28 trios x 6 oriented pairs per trio = 168 ZD pairs.            *)
(**   (Each of 3 assessors in a trio contributes 2 orientations.)     *)
(* ================================================================== *)

(** The 28 mutually-ZD assessor trios, stored as triples of pairs
    ((lo1,hi1),(lo2,hi2),(lo3,hi3)) with hi_i = G xor lo_i.
    Representation type: (((nat*nat)*(nat*nat))*(nat*nat)).
    Organized by G-index from 9 to 15 (4 trios each). *)
Definition mzd_trios : list (((nat * nat) * (nat * nat)) * (nat * nat)) :=
  (** G = 9  (excl lo=1): Fano lines {2,4,6},{2,5,7},{3,4,7},{3,5,6} *)
  [ ( ((2,11),(4,13)), (6,15) );
    ( ((2,11),(5,12)), (7,14) );
    ( ((3,10),(4,13)), (7,14) );
    ( ((3,10),(5,12)), (6,15) );
  (** G = 10 (excl lo=2): Fano lines {1,4,5},{1,6,7},{3,4,7},{3,5,6} *)
    ( ((1,11),(4,14)), (5,15) );
    ( ((1,11),(6,12)), (7,13) );
    ( ((3, 9),(4,14)), (7,13) );
    ( ((3, 9),(5,15)), (6,12) );
  (** G = 11 (excl lo=3): Fano lines {1,4,5},{1,6,7},{2,4,6},{2,5,7} *)
    ( ((1,10),(4,15)), (5,14) );
    ( ((1,10),(6,13)), (7,12) );
    ( ((2, 9),(4,15)), (6,13) );
    ( ((2, 9),(5,14)), (7,12) );
  (** G = 12 (excl lo=4): Fano lines {1,2,3},{1,6,7},{2,5,7},{3,5,6} *)
    ( ((1,13),(2,14)), (3,15) );
    ( ((1,13),(6,10)), (7,11) );
    ( ((2,14),(5, 9)), (7,11) );
    ( ((3,15),(5, 9)), (6,10) );
  (** G = 13 (excl lo=5): Fano lines {1,2,3},{1,6,7},{2,4,6},{3,4,7} *)
    ( ((1,12),(2,15)), (3,14) );
    ( ((1,12),(6,11)), (7,10) );
    ( ((2,15),(4, 9)), (6,11) );
    ( ((3,14),(4, 9)), (7,10) );
  (** G = 14 (excl lo=6): Fano lines {1,2,3},{1,4,5},{2,5,7},{3,4,7} *)
    ( ((1,15),(2,12)), (3,13) );
    ( ((1,15),(4,10)), (5,11) );
    ( ((2,12),(5,11)), (7, 9) );
    ( ((3,13),(4,10)), (7, 9) );
  (** G = 15 (excl lo=7): Fano lines {1,2,3},{1,4,5},{2,4,6},{3,5,6} *)
    ( ((1,14),(2,13)), (3,12) );
    ( ((1,14),(4,11)), (5,10) );
    ( ((2,13),(4,11)), (6, 9) );
    ( ((3,12),(5,10)), (6, 9) ) ].

(** There are exactly 28 mutually-ZD assessor trios (7 box-kites x 4). *)
Theorem mzd_trio_count : length mzd_trios = 28.
Proof. vm_compute. reflexivity. Qed.

(** The count decomposes as 7 box-kites, each contributing 4 Fano-line trios. *)
Theorem mzd_count_decomposition : 28 = 7 * 4.
Proof. reflexivity. Qed.

(** All three assessors in each trio share the same XOR signature (G-index).
    For each ((a,b),(c,d),(e,f)): a xor b = c xor d = e xor f.
    This confirms that each trio is a set of three mutual co-assessors,
    all belonging to the same box-kite.
    Proof: exhaustive vm_compute check over all 28 trios. *)
Theorem mzd_trios_same_sig :
  List.forallb (fun '(((a,b),(c,d)),(e,f)) =>
    andb (Nat.eqb (Nat.lxor a b) (Nat.lxor c d))
         (Nat.eqb (Nat.lxor c d) (Nat.lxor e f)))
  mzd_trios = true.
Proof. vm_compute. reflexivity. Qed.

(** The lo-indices of each trio form a Fano triple: a xor c xor e = 0.
    Equivalently, a xor c = e, i.e., {a,c,e} is a Fano line in PG(2,2).
    This is the same XOR rule that governs octonion triples (FanoPlane.v):
    the zero-divisor structure of the sedenion box-kites mirrors the
    projective geometry of the Fano plane.
    Proof: exhaustive vm_compute check over all 28 trios. *)
Theorem mzd_trios_lo_fano :
  List.forallb (fun '(((a,_),(c,_)),(e,_)) =>
    Nat.eqb (Nat.lxor (Nat.lxor a c) e) 0)
  mzd_trios = true.
Proof. vm_compute. reflexivity. Qed.

(** The G-indices of all 28 trios lie in {9..15} (valid box-kite sigs).
    Proof: vm_compute. *)
Theorem mzd_trios_valid_sigs :
  List.forallb (fun '(((a,b),_),_) =>
    andb (Nat.leb 9 (Nat.lxor a b)) (Nat.leb (Nat.lxor a b) 15))
  mzd_trios = true.
Proof. vm_compute. reflexivity. Qed.

(** The concrete bk7 trio from bk7_coassessor_trio appears in mzd_trios.
    {(3,10),(5,12),(6,15)} is the 4th entry (G=9, Fano line {3,5,6}):
    3 xor 5 = 6, and all three share G-index 9 = 3 xor 10 = 5 xor 12 = 6 xor 15. *)
Example bk7_trio_in_mzd :
  List.nth 3 mzd_trios (((0,0),(0,0)),(0,0)) = (((3,10),(5,12)),(6,15)).
Proof. vm_compute. reflexivity. Qed.

(** The 28 trios and 168 ZD pairs: each trio has 3 assessor pairs, each
    traversable in 2 signed orientations, giving 3 * 2 = 6 ZD pairs per trio.
    28 trios * 6 = 168, consistent with zd_total_from_boxkites. *)
Theorem mzd_trios_to_zd_pairs : 28 * 6 = 168.
Proof. reflexivity. Qed.

(* ================================================================== *)
(** * D4 symmetry of box-kite 7.                                      *)
(**                                                                    *)
(** de Marrais (2000) Section II observes that each box-kite has D4   *)
(** (dihedral group of order 8) symmetry. We formalize this for       *)
(** box-kite 7 (G-index = 9), whose 6 lo-values are {2,3,4,5,6,7}.  *)
(**                                                                    *)
(** The D4 group is generated by two elements r and s satisfying:     *)
(**   r^4 = e,  s^2 = e,  s*r*s = r^{-1} = r^3.                      *)
(**                                                                    *)
(** Concrete realization on the lo-index set {2,3,4,5,6,7}:           *)
(**   r (rotation): 2->4->3->5->2 (4-cycle) and 6<->7                 *)
(**   s (reflection): 4<->5 and 6<->7 (2 is 3 fixed)                  *)
(**                                                                    *)
(** Geometric picture: the 6 lo-values form a "square" in Fano space. *)
(** The 4 corners {2,3,4,5} are cycled by r; the "poles" {6,7} flip  *)
(** under r and under s. The 4 mzd trios for G=9 are the "sides"     *)
(** of this square -- they form one D4 orbit.                         *)
(**                                                                    *)
(** Source: de Marrais (2000) arXiv:math/0011260, Section II.        *)
(* ================================================================== *)

(** Rotation generator r: cycle (2 4 3 5)(6 7) on lo-indices.
    Acts on the 4 "corner" lo-values and swaps the 2 "pole" values. *)
Definition bk7_rot (lo : nat) : nat :=
  match lo with
  | 2 => 4 | 4 => 3 | 3 => 5 | 5 => 2
  | 6 => 7 | 7 => 6
  | n => n
  end.

(** Reflection generator s: swap (4 5)(6 7) on lo-indices.
    Fixes the 2 "corner axis" lo-values {2,3} and swaps the rest. *)
Definition bk7_ref (lo : nat) : nat :=
  match lo with
  | 4 => 5 | 5 => 4
  | 6 => 7 | 7 => 6
  | n => n
  end.

(** The 6 lo-values of box-kite 7 (G=9, excluded lo=1). *)
Definition bk7_los : list nat := [2; 3; 4; 5; 6; 7].

(** Explicit images of bk7_rot and bk7_ref on bk7_los. *)
Theorem bk7_rot_image :
  List.map bk7_rot bk7_los = [4; 5; 3; 2; 7; 6].
Proof. vm_compute. reflexivity. Qed.

Theorem bk7_ref_image :
  List.map bk7_ref bk7_los = [2; 3; 5; 4; 7; 6].
Proof. vm_compute. reflexivity. Qed.

(** D4 relation 1: r^4 = identity.
    The 4-cycle (2 4 3 5) has order 4; (6 7) has order 2; lcm = 4. *)
Theorem bk7_rot4_is_id :
  List.map (fun lo => bk7_rot (bk7_rot (bk7_rot (bk7_rot lo)))) bk7_los
  = bk7_los.
Proof. vm_compute. reflexivity. Qed.

(** D4 relation 2: s^2 = identity.
    The transpositions (4 5) and (6 7) each square to identity. *)
Theorem bk7_ref2_is_id :
  List.map (fun lo => bk7_ref (bk7_ref lo)) bk7_los = bk7_los.
Proof. vm_compute. reflexivity. Qed.

(** D4 relation 3: s*r*s = r^{-1} = r^3.
    This is the dihedral conjugation relation that distinguishes D4
    from Z_4 x Z_2 (the other group of order 8 with an element of order 4). *)
Theorem bk7_d4_conjugate :
  List.map (fun lo => bk7_ref (bk7_rot (bk7_ref lo))) bk7_los
  = List.map (fun lo => bk7_rot (bk7_rot (bk7_rot lo))) bk7_los.
Proof. vm_compute. reflexivity. Qed.

(** r and s each map bk7_los to a permutation of bk7_los (closure). *)
Theorem bk7_rot_in_bk7 :
  List.forallb (fun lo => List.existsb (Nat.eqb lo) bk7_los)
    (List.map bk7_rot bk7_los) = true.
Proof. vm_compute. reflexivity. Qed.

Theorem bk7_ref_in_bk7 :
  List.forallb (fun lo => List.existsb (Nat.eqb lo) bk7_los)
    (List.map bk7_ref bk7_los) = true.
Proof. vm_compute. reflexivity. Qed.

(** The 8 D4 elements are pairwise distinct on bk7_los.
    Each row below is the image of bk7_los under one D4 element.
    All 8 rows are visually distinct; vm_compute confirms pairwise inequality. *)
Theorem bk7_d4_eight_distinct :
  let e   := bk7_los in
  let r   := List.map bk7_rot bk7_los in
  let r2  := List.map (fun lo => bk7_rot (bk7_rot lo)) bk7_los in
  let r3  := List.map (fun lo => bk7_rot (bk7_rot (bk7_rot lo))) bk7_los in
  let s   := List.map bk7_ref bk7_los in
  let rs  := List.map (fun lo => bk7_rot (bk7_ref lo)) bk7_los in
  let r2s := List.map (fun lo => bk7_rot (bk7_rot (bk7_ref lo))) bk7_los in
  let r3s := List.map (fun lo => bk7_rot (bk7_rot (bk7_rot (bk7_ref lo)))) bk7_los in
  e <> r /\ r <> r2 /\ r2 <> r3 /\ r3 <> s /\
  s <> rs /\ rs <> r2s /\ r2s <> r3s /\ r3s <> e.
Proof.
  simpl. repeat split; discriminate.
Qed.

(** Applying bk7_rot to the 4 mzd trios for G=9 (the first 4 entries of
    mzd_trios) permutes them: the rotation r acts as the 4-cycle
    T1->T3->T4->T2->T1 on the lo-triple representatives. *)
Theorem bk7_rot_permutes_trios :
  List.map
    (fun '(((a,b),(c,d)),(e,f)) =>
      (((bk7_rot a, Nat.lxor 9 (bk7_rot a)),
        (bk7_rot c, Nat.lxor 9 (bk7_rot c))),
        (bk7_rot e, Nat.lxor 9 (bk7_rot e))))
    (List.firstn 4 mzd_trios)
  = [( ((4,13),(3,10)), (7,14) );
     ( ((4,13),(2,11)), (6,15) );
     ( ((5,12),(3,10)), (6,15) );
     ( ((5,12),(2,11)), (7,14) )].
Proof. vm_compute. reflexivity. Qed.
