(** * CDSignBridge: Connecting Z-valued sign function to R-valued CDOct arithmetic.

    Bridges the cd_sign_fuel / oct_sign (Z-valued) to the concrete octonion
    multiplication (R-valued CDOct records).

    KEY THEOREM: for all basis indices i, j in {0..7},
      oct_mul (oct_e i) (oct_e j) = oct_scale (sign_to_R (oct_sign i j)) (oct_e (Nat.lxor i j))

    This formalizes the XOR rule: multiplying octonion basis elements produces
    another basis element with index = i XOR j, sign = oct_sign i j.

    PROOF STRATEGY NOTES (why these proofs are structured the way they are):
    ==========================================================================
    1. vm_compute cannot reduce R-valued goals -- R is axiomatic in Rocq.
       Use: cbv [whitelist] + f_equal decomposition + ring.

    2. The 64-case main theorem uses NESTED DESTRUCT patterns:
       `destruct i as [|[|[|...]]]` enumerates i=0,1,...,8 concretely.
       `try lia` closes the i>=8 branch (contradicts i < 8 hypothesis).
       This avoids the `do N` / `[tac1 | try tac2]` antipattern where
       tac2 silently fails and leaves CDOct goals open for the final lia.

    3. sign_to_R converts Z sign values to R without IZR infrastructure.
       Eliminates the need to unfold IZR's recursive definition in cbv.

    4. oct_basis_sign_norm uses oct_norm_mul (multiplicativity) to avoid
       a 64-case enumeration -- a tower-rewrite speedup.

    PERFORMANCE PROFILE:
    - oct_basis_mul_xor: ~3-10s (64 cases of cbv+ring on CDOct)
    - oct_basis_sign_norm: <1s (no enumeration, uses multiplicativity)
    - oct_self_xor_zero: <1s (structural induction, no R arithmetic)

    Self-contained: cd_sign_fuel defined locally (see M3IsAssociator.v for
    authoritative source and algebraic property proofs).

    Analog: Rust sparse_multiply uses XOR index + sign lookup.
    Supports: Phase B of CD sign infrastructure sprint.
    Claims: CDSignBridge extension of C-1505 series. *)

From Stdlib Require Import ZArith Bool Arith Reals Lra Lia List.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** ================================================================== *)
(** * Local sign function: canonical cd_sign_fuel at dim=8.           *)
(**                                                                     *)
(**   Self-contained copy from M3IsAssociator.v. The two copies share  *)
(**   the same definition -- any theorem proved here holds when       *)
(**   M3IsAssociator.cd_sign_fuel is in scope under its qualified name. *)
(** ================================================================== *)

Fixpoint cd_sign_fuel (fuel : nat) (dim p q : nat) : Z :=
  match fuel with
  | O => 1%Z
  | S fuel' =>
    let half := Nat.div dim 2 in
    if Nat.eqb half 0 then 1%Z
    else if andb (Nat.ltb p half) (Nat.ltb q half) then
      cd_sign_fuel fuel' half p q
    else if andb (Nat.ltb p half) (negb (Nat.ltb q half)) then
      cd_sign_fuel fuel' half (q - half)%nat p
    else if andb (negb (Nat.ltb p half)) (Nat.ltb q half) then
      let s := cd_sign_fuel fuel' half (p - half)%nat q in
      if Nat.eqb q 0 then s else Z.opp s
    else
      let qh := (q - half)%nat in
      let ph := (p - half)%nat in
      if Nat.eqb qh 0 then (-1)%Z
      else cd_sign_fuel fuel' half qh ph
  end.

Definition oct_sign (i j : nat) : Z := cd_sign_fuel 4 8 i j.

(** ================================================================== *)
(** * sign_to_R: convert Z sign value to R scalar.                    *)
(**                                                                     *)
(**   Uses a match rather than IZR to avoid unfolding IZR's recursive  *)
(**   definition inside the cbv whitelist -- a significant speedup.   *)
(**   sign_to_R s = 1 for s=1, and = -1 for any other Z value.       *)
(**   Since oct_sign always returns ±1, sign_to_R is correct here.   *)
(** ================================================================== *)

Definition sign_to_R (s : Z) : R :=
  match s with
  | 1%Z => 1
  | _ => -1
  end.

(** sign_to_R maps any Z to exactly +1 or -1 (by match structure).
    Proof by Z-case split: Z is 0 | Zpos p | Zneg n;
    Zpos p is xH (=1) | xO q (even, > 1) | xI q (odd, >= 3). *)
Lemma sign_to_R_pm1 : forall s : Z,
    sign_to_R s = 1 \/ sign_to_R s = -1.
Proof.
  intro s. unfold sign_to_R.
  destruct s as [| p | n].
  - right; reflexivity.
  - destruct p as [p'|p'|].
    + right; reflexivity.   (* xI p' -> odd positive > 1 -> _ -> -1 *)
    + right; reflexivity.   (* xO p' -> even positive -> _ -> -1 *)
    + left; reflexivity.    (* xH = 1%Z -> first branch -> 1 *)
  - right; reflexivity.
Qed.

(** ================================================================== *)
(** * oct_sign_to_oct: helper combining sign and index.               *)
(** ================================================================== *)

Definition oct_sign_to_oct (s : Z) (idx : nat) : CDOct :=
  oct_scale (sign_to_R s) (oct_e idx).

(** ================================================================== *)
(** * Reduction tactic: cd_cbv.                                        *)
(**                                                                     *)
(**   Unfolds exactly the definitions needed to reduce CDOct goals to  *)
(**   closed R arithmetic. Does NOT include nat.div or cd_sign_fuel's  *)
(**   recursive body -- those are pre-computed via oct_sign lookup.   *)
(**                                                                     *)
(**   PERFORMANCE NOTE: Including cd_sign_fuel in the cbv whitelist   *)
(**   causes the fixpoint to be INLINED for each case, generating     *)
(**   large proof terms. Instead, oct_sign i j is pre-evaluated by    *)
(**   including cd_sign_fuel, Nat.div, Nat.eqb, Nat.ltb, andb, negb  *)
(**   in the whitelist so they reduce to concrete Z values before the  *)
(**   CDOct arithmetic reduction runs.                                 *)
(** ================================================================== *)

Ltac cd_cbv :=
  cbv [oct_mul oct_conj oct_add oct_sub oct_neg oct_scale oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_scale quat_zero quat_one
       oct_lo oct_hi qa qb qc qd sign_to_R oct_sign cd_sign_fuel
       Z.opp
       Nat.div Nat.divmod fst snd Nat.sub
       Nat.eqb Nat.ltb Nat.leb andb negb
       Nat.lxor Nat.bitwise Nat.max Nat.even Nat.odd Nat.div2
       Nat.add Nat.mul xorb].

Ltac close_oct_case :=
  cd_cbv; apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.

(** ================================================================== *)
(** * oct_self_xor_zero: nat-level fact supporting the diagonal case. *)
(**                                                                     *)
(**   Nat.lxor i i = 0 for all i -- proved by structural induction,  *)
(**   NOT by enumeration. This is a non-enumerative proof.            *)
(**   (Alternative: Nat.lxor_nilpotent in Rocq stdlib if available.)  *)
(** ================================================================== *)

Lemma oct_self_xor_zero : forall i : nat, Nat.lxor i i = 0%nat.
Proof.
  intro i. apply Nat.bits_inj. intro n.
  rewrite Nat.lxor_spec. rewrite Nat.bits_0.
  apply Bool.xorb_nilpotent.
Qed.

(** ================================================================== *)
(** * oct_basis_mul_xor: THE KEY BRIDGE THEOREM.                      *)
(**                                                                     *)
(**   For all i, j in {0..7}:                                         *)
(**     oct_mul (oct_e i) (oct_e j)                                   *)
(**       = oct_scale (sign_to_R (oct_sign i j)) (oct_e (Nat.lxor i j)) *)
(**                                                                     *)
(**   PROOF STRATEGY (avoiding the do-N antipattern):                 *)
(**   Use nested destruct `[|[|[|...]]]` to enumerate i=0..8 and      *)
(**   j=0..8 concretely. `try lia` closes i>=8 and j>=8 branches      *)
(**   (contradicts Hi / Hj hypotheses). Each concrete (i,j) pair is   *)
(**   closed by close_oct_case = cd_cbv + f_equal + ring.             *)
(**                                                                     *)
(**   WHY NOT do N: `do N (destruct; [tac1 | try tac2])` silently    *)
(**   fails on CDOct goals in tac2, leaving them open. The final lia  *)
(**   then sees a CDOct equality (not arithmetic) and fails.          *)
(** ================================================================== *)

Theorem oct_basis_mul_xor : forall i j : nat,
    (i < 8)%nat -> (j < 8)%nat ->
    oct_mul (oct_e i) (oct_e j) =
      oct_scale (sign_to_R (oct_sign i j)) (oct_e (Nat.lxor i j)).
Proof.
  intros i j Hi Hj.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  close_oct_case.
Qed.

(** ================================================================== *)
(** * oct_basis_sign_norm: sign function preserves norm.              *)
(**                                                                     *)
(**   For all i, j in {0..7}: oct_norm_sq (oct_mul (oct_e i) (oct_e j)) = 1. *)
(**                                                                     *)
(**   TOWER-REWRITE PROOF: Uses oct_norm_mul (multiplicativity) and   *)
(**   oct_e_norm (basis elements have unit norm) to avoid a           *)
(**   64-case enumeration. This is the FAST path:                     *)
(**   norm(e_i * e_j) = norm(e_i) * norm(e_j) = 1 * 1 = 1.          *)
(**   Contrast with oct_basis_mul_xor which needs all 64 cases.       *)
(** ================================================================== *)

Lemma oct_e_norm : forall i : nat,
    (i < 8)%nat -> oct_norm_sq (oct_e i) = 1.
Proof.
  intros i Hi.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  cbv [oct_norm_sq quat_norm_sq oct_e oct_lo oct_hi qa qb qc qd
       quat_zero quat_one]; ring.
Qed.

Theorem oct_basis_sign_norm : forall i j : nat,
    (i < 8)%nat -> (j < 8)%nat ->
    oct_norm_sq (oct_mul (oct_e i) (oct_e j)) = 1.
Proof.
  intros i j Hi Hj.
  rewrite oct_norm_mul.
  rewrite oct_e_norm; [| exact Hi].
  rewrite oct_e_norm; [| exact Hj].
  ring.
Qed.

(** ================================================================== *)
(** * Corollaries of oct_basis_mul_xor.                               *)
(** ================================================================== *)

(** e_0 is the left identity: oct_mul (oct_e 0) (oct_e j) = oct_e j. *)
Lemma oct_mul_e0_left : forall j : nat,
    (j < 8)%nat ->
    oct_mul (oct_e 0) (oct_e j) = oct_e j.
Proof.
  intros j Hj.
  rewrite oct_basis_mul_xor; [| lia | exact Hj].
  destruct j as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  cbv [oct_sign cd_sign_fuel Nat.div Nat.divmod fst snd Nat.sub
       Nat.eqb Nat.ltb Nat.leb andb negb sign_to_R
       oct_scale quat_scale quat_zero quat_one oct_lo oct_hi oct_e
       Nat.lxor Nat.bitwise Nat.max Nat.even Nat.odd Nat.div2 Nat.add Nat.mul xorb
       qa qb qc qd]; f_equal; f_equal; ring.
Qed.

(** Self-square: imaginary basis elements square to -e_0. *)
Lemma oct_mul_self_neg : forall i : nat,
    (1 <= i)%nat -> (i < 8)%nat ->
    oct_mul (oct_e i) (oct_e i) = oct_neg (oct_e 0).
Proof.
  intros i Hlo Hhi.
  rewrite oct_basis_mul_xor; [| lia | lia].
  rewrite oct_self_xor_zero.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  cbv [oct_sign cd_sign_fuel Nat.div Nat.divmod fst snd Nat.sub
       Nat.eqb Nat.ltb Nat.leb andb negb sign_to_R
       oct_scale oct_neg oct_e quat_scale quat_neg quat_zero quat_one
       oct_lo oct_hi qa qb qc qd]; f_equal; f_equal; ring.
Qed.
