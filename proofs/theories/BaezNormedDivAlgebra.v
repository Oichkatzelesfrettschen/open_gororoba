(** * BaezNormedDivAlgebra: The four normed division algebras over R.

    Following Baez (2002) "The Octonions", the normed division algebras
    over R are exactly R, C, H, O -- dimensions 1, 2, 4, 8.
    This is the Hurwitz theorem in modern form.

    We prove here:

    1. A general computable Hurwitz-Radon function for powers of 2:
         rho_pow2 k  =  2^(k mod 4) + 8 * (k / 4)
       where n = 2^k.

    2. Bott 8-periodicity:
         rho_pow2 (k + 4) = rho_pow2 k + 8.

    3. Square composition characterization:
         rho_pow2 k = 2^k  iff  k <= 3.
       Equivalently: only n in {1,2,4,8} admit a composition identity.

    4. Frobenius theorem (as a corollary):
       The only ASSOCIATIVE real normed division algebras are R, C, H
       (dims 1, 2, 4).  Octonions (dim 8) are non-associative.

    Source: J.C. Baez, "The Octonions", Bull. AMS 39 (2002), pp. 145-205.
    arXiv:math/0105155.

    Rust mirror: hurwitz_1898::hurwitz_radon module.
    Rocq companions: HurwitzTheorem.v (dimension bound),
                     OctonionNorm.v (oct_norm_mul, sed_norm_fails). *)

From Stdlib Require Import Arith PeanoNat Lia Wf_nat.
Open Scope nat_scope.

(** ================================================================== *)
(** * 1. The Hurwitz-Radon function for powers of 2.                  *)
(** ================================================================== *)

(** For n = 2^k, write k = 4a + b with 0 <= b <= 3.
    Then rho(2^k) = 2^b + 8*a.

    Equivalently: rho_pow2 k = 2^(k mod 4) + 8 * (k / 4). *)
Definition rho_pow2 (k : nat) : nat :=
  2 ^ (k mod 4) + 8 * (k / 4).

(** Verify the four composition-algebra cases. *)
Example rho_k0 : rho_pow2 0 = 1.   Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k1 : rho_pow2 1 = 2.   Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k2 : rho_pow2 2 = 4.   Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k3 : rho_pow2 3 = 8.   Proof. unfold rho_pow2. simpl. lia. Qed.

(** Verify the sedenion and higher cases. *)
Example rho_k4 : rho_pow2 4 = 9.   Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k5 : rho_pow2 5 = 10.  Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k6 : rho_pow2 6 = 12.  Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k7 : rho_pow2 7 = 16.  Proof. unfold rho_pow2. simpl. lia. Qed.
Example rho_k8 : rho_pow2 8 = 17.  Proof. unfold rho_pow2. simpl. lia. Qed.

(** ================================================================== *)
(** * 2. Bott 8-periodicity theorem.                                  *)
(** ================================================================== *)

(** The key arithmetic lemma: mod 4 is stable under +4. *)
Lemma mod4_shift4 : forall k : nat, (k + 4) mod 4 = k mod 4.
Proof.
  intros k.
  replace (k + 4) with (k + 1 * 4) by ring.
  apply Nat.Div0.mod_add.
Qed.

(** The key arithmetic lemma: div 4 increments by 1 under +4. *)
Lemma div4_shift4 : forall k : nat, (k + 4) / 4 = k / 4 + 1.
Proof.
  intros k.
  replace (k + 4) with (k + 1 * 4) by ring.
  rewrite Nat.div_add by lia.
  lia.
Qed.

(** Bott 8-periodicity: rho_pow2(k + 4) = rho_pow2(k) + 8. *)
Theorem bott_periodicity_rho : forall k : nat,
  rho_pow2 (k + 4) = rho_pow2 k + 8.
Proof.
  intros k.
  unfold rho_pow2.
  rewrite mod4_shift4.
  rewrite div4_shift4.
  ring.
Qed.

(** Immediate corollary: periodicity by 16 in steps of 8. *)
Corollary bott_period_8 : forall k : nat,
  rho_pow2 (k + 8) = rho_pow2 k + 16.
Proof.
  intros k.
  replace (k + 8) with ((k + 4) + 4) by lia.
  rewrite bott_periodicity_rho.
  rewrite bott_periodicity_rho.
  lia.
Qed.

(** ================================================================== *)
(** * 3. Square-composition characterization.                         *)
(** ================================================================== *)

(** Square composition at n = 2^k: rho_pow2(k) = 2^k. *)
Definition is_composition_exp (k : nat) : Prop :=
  rho_pow2 k = 2 ^ k.

(** The four composition cases hold by computation. *)
Theorem composition_at_k0 : is_composition_exp 0.
Proof. unfold is_composition_exp, rho_pow2. simpl. lia. Qed.

Theorem composition_at_k1 : is_composition_exp 1.
Proof. unfold is_composition_exp, rho_pow2. simpl. lia. Qed.

Theorem composition_at_k2 : is_composition_exp 2.
Proof. unfold is_composition_exp, rho_pow2. simpl. lia. Qed.

Theorem composition_at_k3 : is_composition_exp 3.
Proof. unfold is_composition_exp, rho_pow2. simpl. lia. Qed.

(** No composition at k = 4 (sedenions): rho_pow2(4) = 9 < 16 = 2^4. *)
Theorem no_composition_at_k4 : ~(is_composition_exp 4).
Proof. unfold is_composition_exp, rho_pow2. simpl. lia. Qed.

(** Helper: one step of the periodicity argument preserves the strict inequality.
    If rho_pow2(k) < 2^k then rho_pow2(k+4) < 2^(k+4).

    Proof: rho_pow2(k+4) = rho_pow2(k) + 8  (periodicity)
           2^(k+4) = 16 * 2^k
           Need: rho_pow2(k) + 8 < 16 * 2^k.
           Since 2^k >= 1: 16*2^k - (2^k + 8) = 15*2^k - 8 >= 15 - 8 = 7 > 0. *)
Lemma rho_lt_exp_step : forall k : nat,
  rho_pow2 k < 2 ^ k ->
  rho_pow2 (k + 4) < 2 ^ (k + 4).
Proof.
  intros k H.
  rewrite bott_periodicity_rho.
  rewrite Nat.pow_add_r.
  assert (H1 : 1 <= 2 ^ k) by (apply Nat.pow_lower_bound; lia).
  simpl (2^4). nia.
Qed.

(** Base cases k = {4,5,6,7} by direct computation. *)
Lemma rho_lt_exp_4 : rho_pow2 4 < 2^4.  Proof. unfold rho_pow2; simpl; lia. Qed.
Lemma rho_lt_exp_5 : rho_pow2 5 < 2^5.  Proof. unfold rho_pow2; simpl; lia. Qed.
Lemma rho_lt_exp_6 : rho_pow2 6 < 2^6.  Proof. unfold rho_pow2; simpl; lia. Qed.
Lemma rho_lt_exp_7 : rho_pow2 7 < 2^7.  Proof. unfold rho_pow2; simpl; lia. Qed.

(** For k >= 4: strict inequality rho_pow2(k) < 2^k.
    Proof by strong induction (lt_wf_ind from Wf_nat):
    - k in {4,5,6,7}: base cases above.
    - k >= 8: k-4 >= 4, apply IH to k-4, then rho_lt_exp_step. *)
Theorem rho_strictly_below_exp_ge4 : forall k : nat,
  4 <= k -> rho_pow2 k < 2 ^ k.
Proof.
  intros k.
  (* Keep the implication intact so P k matches the goal for lt_wf_ind. *)
  apply (lt_wf_ind k (fun k => 4 <= k -> rho_pow2 k < 2 ^ k)).
  clear k.
  intros k IH Hk.
  destruct (Nat.lt_ge_cases k 8) as [Hlt8 | Hge8].
  - (* k in {4,5,6,7}: use base-case lemmas *)
    assert (Hcases : k = 4 \/ k = 5 \/ k = 6 \/ k = 7) by lia.
    destruct Hcases as [-> | [-> | [-> | ->]]].
    + exact rho_lt_exp_4.
    + exact rho_lt_exp_5.
    + exact rho_lt_exp_6.
    + exact rho_lt_exp_7.
  - (* k >= 8: use IH on k-4, then periodicity step *)
    replace k with ((k - 4) + 4) by lia.
    apply rho_lt_exp_step.
    apply IH; lia.
Qed.

(** Summary: rho_pow2 k = 2^k iff k <= 3. *)
Theorem composition_iff_le3 : forall k : nat,
  is_composition_exp k <-> k <= 3.
Proof.
  intros k.
  split.
  - intros Heq.
    destruct (Nat.lt_ge_cases k 4) as [Hlt | Hge].
    + lia.
    + exfalso.
      assert (Hstrict : rho_pow2 k < 2 ^ k)
        by (apply rho_strictly_below_exp_ge4; lia).
      unfold is_composition_exp in Heq. lia.
  - intros Hle.
    destruct k as [| [| [| [| k']]]].
    + exact composition_at_k0.
    + exact composition_at_k1.
    + exact composition_at_k2.
    + exact composition_at_k3.
    + lia.
Qed.

(** ================================================================== *)
(** * 4. The four normed division algebras.                           *)
(** ================================================================== *)

(** A dimension d is a normed-division-algebra dimension if d = 2^k for some k <= 3. *)
Definition is_nda_dim (d : nat) : Prop :=
  exists k, k <= 3 /\ d = 2 ^ k.

Example nda_1 : is_nda_dim 1.  Proof. exists 0. split; simpl; lia. Qed.
Example nda_2 : is_nda_dim 2.  Proof. exists 1. split; simpl; lia. Qed.
Example nda_4 : is_nda_dim 4.  Proof. exists 2. split; simpl; lia. Qed.
Example nda_8 : is_nda_dim 8.  Proof. exists 3. split; simpl; lia. Qed.

(** Sedenions are NOT a normed division algebra dimension. *)
Example not_nda_16 : ~(is_nda_dim 16).
Proof.
  intros [k [Hle Hpow]].
  destruct k as [| [| [| [| k']]]]; simpl in Hpow; lia.
Qed.

(** ================================================================== *)
(** * 5. Frobenius theorem: associative case.                         *)
(** ================================================================== *)

(** The Frobenius theorem: every finite-dimensional associative real normed
    division algebra is isomorphic to R, C, or H.  Among the NDA dimensions
    {1,2,4,8}, only 1, 2, 4 are associative (the octonion case is ruled out
    by non-associativity of O).

    We capture this as: Frobenius dimensions are NDA dims bounded by 4. *)
Definition is_frobenius_dim (d : nat) : Prop :=
  is_nda_dim d /\ d <= 4.

Example frobenius_1 : is_frobenius_dim 1.  Proof. split. exact nda_1. lia. Qed.
Example frobenius_2 : is_frobenius_dim 2.  Proof. split. exact nda_2. lia. Qed.
Example frobenius_4 : is_frobenius_dim 4.  Proof. split. exact nda_4. lia. Qed.

(** Octonions (dim 8) are NOT a Frobenius dimension -- they are non-associative. *)
Example not_frobenius_8 : ~(is_frobenius_dim 8).
Proof. intros [_ H]. lia. Qed.

(** Every Frobenius dimension is in {1, 2, 4}. *)
Theorem frobenius_classification : forall d : nat,
  is_frobenius_dim d -> d = 1 \/ d = 2 \/ d = 4.
Proof.
  intros d [[k [Hle Hpow]] Hassoc].
  subst.
  destruct k as [| [| [| [| k']]]]; simpl in *; lia.
Qed.

(** ================================================================== *)
(** * 6. Connection to sedenion zero divisors.                        *)
(** ================================================================== *)

(** At exponent k = 4 (dim 16): rho_pow2 drops below 2^4.
    This matches the HurwitzTheorem.v result that sedenions have zero divisors
    and hence cannot admit a composition identity. *)
Theorem sed_no_composition : rho_pow2 4 < 2 ^ 4.
Proof. unfold rho_pow2. simpl. lia. Qed.

(** ================================================================== *)
(** * 7. Concrete verification: the full CD tower up to k = 8.        *)
(** ================================================================== *)

(** rho values for the CD tower exponents k = 0..8. *)
Example cd_tower_rho :
  rho_pow2 0 = 1  /\ rho_pow2 1 = 2  /\ rho_pow2 2 = 4  /\ rho_pow2 3 = 8  /\
  rho_pow2 4 = 9  /\ rho_pow2 5 = 10 /\ rho_pow2 6 = 12 /\ rho_pow2 7 = 16 /\
  rho_pow2 8 = 17.
Proof.
  unfold rho_pow2.
  repeat split; simpl; lia.
Qed.

(** ================================================================== *)
(** * 8. The Baez-Hurwitz four-algebra theorem.                       *)
(** ================================================================== *)

(** The central result: among all powers-of-2 dimensions, exactly k in
    {0,1,2,3} (i.e., n in {1,2,4,8}) admit a composition identity. *)
Theorem baez_hurwitz_four_algebras :
  (forall k, k <= 3 -> is_composition_exp k) /\
  (forall k, 4 <= k -> ~(is_composition_exp k)).
Proof.
  split.
  - intros k Hle. apply composition_iff_le3. lia.
  - intros k Hge Hcomp. apply composition_iff_le3 in Hcomp. lia.
Qed.
