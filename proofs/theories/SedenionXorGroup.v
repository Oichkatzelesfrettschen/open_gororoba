(** * SedenionXorGroup: XOR Boolean group structure of sedenion indices.

    The 16 sedenion basis indices {0..15} form the group (Z_2)^4 under
    bitwise XOR.  This structure governs:
    - The assessor XOR signature: sig(lo, hi) = Nat.lxor lo hi
    - The 7 box-kite partition: each box-kite corresponds to one
      element of {9..15} as its XOR signature
    - The Brown-de Marrais equivalence: co-assessors share a signature

    Key structural theorems:

    1. XOR GROUP AXIOMS on nat: identity, self-inverse, associativity.

    2. ASSESSOR SIGNATURE RANGE:
       For any valid assessor (lo, hi) with lo in {1..7}, hi in {9..15},
       hi <> lo + 8 (no identity pair):
         9 <= Nat.lxor lo hi <= 15.
       Proof: complete case analysis on lo (7 cases) x hi (7 valid cases).

    3. PRODUCTION RULE #1 (XOR form):
       Given co-assessors sharing signature g, the derived pair also has g.

    4. SEVEN SIGNATURES EXHAUST {9..15}:
       The 7 concrete box-kite XOR signatures are exactly {9..15}.
       The Z_2^4 generators explain why: {9..15} = {8 XOR k : k in {1..7}}.

    Rocq companions: BoxKite.v, ZDGraph.v, DeMarraisAssessors.v.
    Rust mirror: de_marrais_2000::assessors (g_index, co_assessor_buckets). *)

From Stdlib Require Import Arith PeanoNat Lia List Bool.
Import ListNotations.

Open Scope nat_scope.

(** ================================================================== *)
(** * 1. XOR group axioms for nat.                                     *)
(** ================================================================== *)

(** Identity element: 0 is the XOR identity. *)
Theorem xor_identity_r : forall n : nat, Nat.lxor n 0 = n.
Proof. exact Nat.lxor_0_r. Qed.

Theorem xor_identity_l : forall n : nat, Nat.lxor 0 n = n.
Proof.
  intros n. rewrite Nat.lxor_comm. apply Nat.lxor_0_r.
Qed.

(** Self-inverse: every element is its own inverse. *)
Theorem xor_self_inverse : forall n : nat, Nat.lxor n n = 0.
Proof. exact Nat.lxor_nilpotent. Qed.

(** Associativity. *)
Theorem xor_associative : forall a b c : nat,
  Nat.lxor (Nat.lxor a b) c = Nat.lxor a (Nat.lxor b c).
Proof. exact Nat.lxor_assoc. Qed.

(** Commutativity. *)
Theorem xor_commutative : forall a b : nat,
  Nat.lxor a b = Nat.lxor b a.
Proof. exact Nat.lxor_comm. Qed.

(** XOR cancellation: a XOR b = a XOR c -> b = c.
    Proof: XOR both sides with a; a XOR a = 0 by self-inverse. *)
Theorem xor_cancel_l : forall a b c : nat,
  Nat.lxor a b = Nat.lxor a c -> b = c.
Proof.
  intros a b c H.
  assert (Heq : Nat.lxor a (Nat.lxor a b) = Nat.lxor a (Nat.lxor a c)).
  { rewrite H. reflexivity. }
  rewrite <- Nat.lxor_assoc, <- Nat.lxor_assoc in Heq.
  rewrite Nat.lxor_nilpotent in Heq.
  rewrite xor_identity_l, xor_identity_l in Heq.
  exact Heq.
Qed.

(** ================================================================== *)
(** * 2. Bit-3 structure of sedenion index ranges.                    *)
(** ================================================================== *)

(** Bit 3 characterization via division: n has bit 3 set iff (n / 8) mod 2 = 1.
    This is Nat.testbit_true from the stdlib.
    Key consequence: n < 8 implies bit 3 = 0, 8 <= n < 16 implies bit 3 = 1. *)

(** For lo < 8: bit 3 of lo is 0.
    Proof: case split on lo in {0..7}; vm_compute closes each. *)
Lemma testbit3_lo_lt8 : forall lo, lo < 8 -> Nat.testbit lo 3 = false.
Proof.
  intros lo Hlo.
  destruct lo as [| [| [| [| [| [| [| [| lo']]]]]]]]; try lia;
  vm_compute; reflexivity.
Qed.

(** For 8 <= hi < 16: bit 3 of hi is 1.
    Proof: case split on hi in {8..15}; vm_compute closes each. *)
Lemma testbit3_hi_ge8_lt16 : forall hi, 8 <= hi -> hi < 16 ->
  Nat.testbit hi 3 = true.
Proof.
  intros hi Hlo Hhi.
  destruct hi as [| [| [| [| [| [| [| [| [| [| [| [| [| [| [| [| hi']]]]]]]]]]]]]]]]; try lia;
  vm_compute; reflexivity.
Qed.

(** For lo < 8 and hi >= 8: bit 3 of lxor lo hi is 1. *)
Lemma xor_lo_hi_bit3_true : forall lo hi,
  lo < 8 -> 8 <= hi -> hi < 16 ->
  Nat.testbit (Nat.lxor lo hi) 3 = true.
Proof.
  intros lo hi Hlo Hhi8 Hhi16.
  rewrite Nat.lxor_spec.
  rewrite testbit3_lo_lt8 by lia.
  apply testbit3_hi_ge8_lt16; lia.
Qed.

(** If bit 3 of n is 1, then n >= 8.
    Proof: testbit n 3 = true means (n/8) mod 2 = 1, so n/8 >= 1, so n >= 8. *)
Lemma testbit3_true_ge8 : forall n,
  Nat.testbit n 3 = true -> 8 <= n.
Proof.
  intros n H.
  rewrite Nat.testbit_true in H.
  replace (2^3) with 8 in H by reflexivity.
  assert (H8 : 1 <= n / 8).
  { destruct (n / 8). simpl in H. discriminate. lia. }
  assert (Heucl : n = 8 * (n / 8) + n mod 8)
    by (apply Nat.div_mod; lia).
  nia.
Qed.

(** ================================================================== *)
(** * 3. Assessor XOR signature is in {9..15}.                        *)
(** ================================================================== *)

(** If lo in {1..7} and hi = lo+8 (the excluded self-pair):
    lxor lo hi = 8.  We prove this to use the contrapositive for the
    lo + 8 exclusion argument. *)
Lemma lxor_lo_lo_plus8 : forall lo, lo < 8 ->
  Nat.lxor lo (lo + 8) = 8.
Proof.
  intros lo Hlo.
  destruct lo as [| [| [| [| [| [| [| [| lo']]]]]]]]; try lia;
  vm_compute; reflexivity.
Qed.

(** Upper bound: lxor lo hi < 16 when lo < 8 and hi < 16.
    Proof: case split on lo in {0..7} and hi in {0..15}; vm_compute each case. *)
Lemma xor_lt_16_of_lt8_lt16 : forall lo hi,
  lo < 8 -> hi < 16 -> Nat.lxor lo hi < 16.
Proof.
  intros lo hi Hlo Hhi.
  destruct lo as [| [| [| [| [| [| [| [| lo']]]]]]]]; try lia;
  destruct hi as [| [| [| [| [| [| [| [| [| [| [| [| [| [| [| [| hi']]]]]]]]]]]]]]]]; try lia;
  vm_compute; lia.
Qed.

(** The main assessor signature range theorem.

    For any valid assessor (lo, hi) with lo in {1..7}, hi in {9..15},
    hi <> lo + 8: the XOR signature satisfies 9 <= lxor lo hi <= 15.

    Proof:
    - Lower bound: bit 3 of lxor lo hi is 1 (lo < 8, hi >= 8), so >= 8.
      And lxor lo hi = 8 only if hi = lo+8 (excluded), so >= 9.
    - Upper bound: lo < 8 and hi < 16 => lxor lo hi < 16, i.e., <= 15. *)
Theorem assessor_sig_range : forall lo hi : nat,
  1 <= lo -> lo <= 7 ->
  9 <= hi -> hi <= 15 ->
  hi <> lo + 8 ->
  9 <= Nat.lxor lo hi /\ Nat.lxor lo hi <= 15.
Proof.
  intros lo hi Hlo1 Hlo7 Hhi9 Hhi15 Hne.
  split.
  - (* Lower bound: lxor lo hi >= 9 *)
    (* Step 1: bit 3 is 1, so lxor lo hi >= 8 *)
    assert (Hbit3 : Nat.testbit (Nat.lxor lo hi) 3 = true).
    { apply xor_lo_hi_bit3_true; lia. }
    assert (Hge8 : 8 <= Nat.lxor lo hi) by (apply testbit3_true_ge8; exact Hbit3).
    (* Step 2: lxor lo hi != 8 (since hi != lo+8) *)
    assert (Hne8 : Nat.lxor lo hi <> 8).
    { intros Heq.
      (* lxor lo hi = 8 = lxor lo (lo+8) *)
      assert (Hlhs : Nat.lxor lo (lo + 8) = 8)
        by (apply lxor_lo_lo_plus8; lia).
      (* So hi = lo+8 by XOR cancellation *)
      apply Hne.
      apply xor_cancel_l with lo.
      rewrite Heq. rewrite Hlhs. reflexivity. }
    lia.
  - (* Upper bound: lxor lo hi <= 15 *)
    assert (Hlt16 : Nat.lxor lo hi < 16)
      by (apply xor_lt_16_of_lt8_lt16; lia).
    lia.
Qed.

(** ================================================================== *)
(** * 4. Production Rule #1 (XOR form).                               *)
(** ================================================================== *)

(** Given co-assessors (a, b) and (c, d) with a XOR b = c XOR d = g,
    the derived pair (a XOR d, a XOR c) also has XOR signature g.
    Proof: (a XOR d) XOR (a XOR c) = (a XOR a) XOR (d XOR c) = 0 XOR (c XOR d) = g. *)
Theorem production_rule_xor_preserves_sig : forall a b c d : nat,
  Nat.lxor a b = Nat.lxor c d ->
  Nat.lxor (Nat.lxor a d) (Nat.lxor a c) = Nat.lxor c d.
Proof.
  intros a b c d H.
  rewrite <- Nat.lxor_assoc.
  rewrite (Nat.lxor_assoc a d a).
  rewrite (Nat.lxor_comm d a).
  rewrite <- (Nat.lxor_assoc a a d).
  rewrite Nat.lxor_nilpotent.
  rewrite xor_identity_l.
  apply Nat.lxor_comm.
Qed.

(** ================================================================== *)
(** * 5. Concrete verification: 7 signatures exhaust {9..15}.         *)
(** ================================================================== *)

(** The 7 box-kite XOR signatures (matching ZDGraph.v seven_distinct_signatures). *)
Definition boxkite_sigs : list nat := [15; 10; 11; 12; 13; 14; 9].

(** Each signature is in {9..15}. *)
Theorem each_sig_in_range :
  List.forallb (fun s => (9 <=? s) && (s <=? 15)) boxkite_sigs = true.
Proof. vm_compute. reflexivity. Qed.

(** There are exactly 7 distinct signatures. *)
Theorem seven_sigs : length boxkite_sigs = 7.
Proof. reflexivity. Qed.

(** The 7 signatures cover all of {9..15}:
    for each n in {9..15}, n appears in boxkite_sigs. *)
Theorem sigs_cover_9_to_15 :
  List.forallb (fun n => List.existsb (Nat.eqb n) boxkite_sigs) [9;10;11;12;13;14;15] = true.
Proof. vm_compute. reflexivity. Qed.

(** ================================================================== *)
(** * 6. The Z_2^4 generator structure.                               *)
(** ================================================================== *)

(** The four canonical generators of Z_2^4 in the sedenion index space:
    e1, e2, e4, e8 correspond to the four bit positions. *)
Definition gen1 : nat := 1.
Definition gen2 : nat := 2.
Definition gen4 : nat := 4.
Definition gen8 : nat := 8.

(** Every non-identity element of {1..15} is a unique XOR combination
    of the generators.  The 7 octonion indices {1..7} use gen1/gen2/gen4,
    and the 7 sedenion-boundary indices {9..15} = {gen8 XOR k : k in {1..7}}
    use gen8 as the "sedenion bit". *)
Example generators_span_15 :
  Nat.lxor gen1 gen2                              = 3  /\
  Nat.lxor gen1 gen4                              = 5  /\
  Nat.lxor gen2 gen4                              = 6  /\
  Nat.lxor gen1 (Nat.lxor gen2 gen4)              = 7  /\
  gen8                                            = 8  /\
  Nat.lxor gen8 gen1                              = 9  /\
  Nat.lxor gen8 gen2                              = 10 /\
  Nat.lxor gen8 (Nat.lxor gen1 gen2)              = 11 /\
  Nat.lxor gen8 gen4                              = 12 /\
  Nat.lxor gen8 (Nat.lxor gen1 gen4)              = 13 /\
  Nat.lxor gen8 (Nat.lxor gen2 gen4)              = 14 /\
  Nat.lxor gen8 (Nat.lxor gen1 (Nat.lxor gen2 gen4)) = 15.
Proof. vm_compute. repeat split; reflexivity. Qed.

(** The 7 box-kite signatures are exactly {gen8 XOR k : k in {1..7}},
    i.e., the 7 elements obtained by flipping the sedenion bit of each
    octonion imaginary unit. *)
Theorem boxkite_sigs_eq_gen8_xor_octonions :
  List.map (Nat.lxor gen8) [1; 2; 3; 4; 5; 6; 7] = [9; 10; 11; 12; 13; 14; 15].
Proof. vm_compute. reflexivity. Qed.

(** This means: each box-kite signature g = 8 XOR (g XOR 8) where g XOR 8 in {1..7}.
    The map g |-> g XOR 8 is a bijection from {9..15} to {1..7},
    i.e., between box-kite signatures and octonion imaginary units. *)
Theorem sig_to_oct_bijection :
  List.map (fun g => Nat.lxor gen8 g) [9; 10; 11; 12; 13; 14; 15] = [1; 2; 3; 4; 5; 6; 7].
Proof. vm_compute. reflexivity. Qed.
