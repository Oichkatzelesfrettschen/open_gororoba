(** * ZD_Criterion: Brown (1972) Major Theorem 7.15 for Zero Divisors.

    Formalizes the criterion for detecting zero divisor pairs in sedenions
    from Brown's dissertation (1972), Chapter VII, Theorem 7.15.

    MAJOR THEOREM (7.15): For sedenions, A = a1 + e*a2 and B = b1 + e*b2
    where a1, a2, b1, b2 are octonions:

      A*B = 0  <==>  (i)   N(a1) = N(a2)
                    (ii)  b2 = [(a1*b1)*a2] / N(a1)
                    (iii) antiassociator(a1, b1, a2) = 0

    where antiassociator(x, y, z) = (x*y)*z + x*(y*z).

    Key results:
    - Formalization of the three necessary and sufficient conditions
    - Definition of the antiassociator at dim 8 (octonionic)
    - Characterization of the fundamental ZD pair (e1+e10, e4-e15)
    - Foundation for formal verification in Rocq

    Mirrors: crates/brown_1972/src/pl1_emulator.rs
    Supports claims: C-1548..C-1557 (Brown 1972 formalization series)

    STATUS: Phase C Step 21 of Brown 1972 Formalization Plan
*)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm CDFusedBilinear.

Open Scope R_scope.

(** ================================================================== *)
(** * The Antiassociator: measure of associativity violation.            *)
(** ================================================================== *)

(** The antiassociator of three octonions measures how far they are from
    satisfying the anti-associative law. It is defined as:
      antiassociator(x, y, z) = (x*y)*z + x*(y*z)

    This is the SUM of the two possible parenthesizations, not the difference.
    When antiassociator = 0, we have (x*y)*z = -x*(y*z).

    In the context of Brown's Major Theorem, the antiassociator condition
    encodes the constraint that the sedenion ZD pair satisfies a specific
    algebraic relation derived from the CD doubling structure. *)

Definition oct_antiassociator (x y z : CDOct) : CDOct :=
  let xy := oct_mul x y in
  let xyz_left := oct_mul xy z in
  let yz := oct_mul y z in
  let xyz_right := oct_mul x yz in
  oct_add xyz_left xyz_right.

(** The norm of the antiassociator measures its magnitude. For ZD pairs,
    this should equal zero (up to numerical precision). *)

Definition oct_antiassociator_norm_sq (x y z : CDOct) : R :=
  oct_norm_sq (oct_antiassociator x y z).

(** ================================================================== *)
(** * Brown's Major Theorem conditions (formalized).                    *)
(** ================================================================== *)

(** Condition (i): The norms of a1 and a2 must be equal. *)

Definition zd_condition_i (a1 a2 : CDOct) : Prop :=
  oct_norm_sq a1 = oct_norm_sq a2.

(** Condition (ii): The second half of B must equal a specific formula.

    Given a1, a2, b1, and N(a1), compute:
      b2 = [(a1*b1)*a2] / N(a1)

    To avoid division, we verify equivalently:
      N(a1) * b2 = (a1*b1)*a2

    This is the condition we check. *)

Definition zd_condition_ii (a1 a2 b1 b2 : CDOct) : Prop :=
  let n := oct_norm_sq a1 in
  let ab1 := oct_mul a1 b1 in
  let ab1_a2 := oct_mul ab1 a2 in
  let scaled_b2 := oct_scale n b2 in
  scaled_b2 = ab1_a2.

(** Condition (iii): The antiassociator of (a1, b1, a2) must equal zero. *)

Definition zd_condition_iii (a1 b1 a2 : CDOct) : Prop :=
  oct_antiassociator a1 b1 a2 = oct_zero.

(** ================================================================== *)
(** * The complete ZD criterion.                                        *)
(** ================================================================== *)

(** A sedenion pair (A, B) with A = a1 + e*a2 and B = b1 + e*b2 is a
    zero divisor pair if and only if all three conditions hold. *)

Definition is_zd_pair_major_theorem (a1 a2 b1 b2 : CDOct) : Prop :=
  zd_condition_i a1 a2 /\
  zd_condition_ii a1 a2 b1 b2 /\
  zd_condition_iii a1 b1 a2.

(** ================================================================== *)
(** * Concrete instantiation: the fundamental ZD pair.                  *)
(** ================================================================== *)

(** The fundamental sedenion ZD pair discovered computationally via the
    PL/1 emulator: A = e1 + e10, B = e4 - e15.

    In octonion decomposition:
      a1 = e1 (index 1 in octonion basis)
      a2 = e2 (index 2 in octonion basis)
      b1 = e4 (index 4 in octonion basis)
      b2 = -e7 (index 7 with sign flip)

    Both a1 and a2 have norm sqrt(2) = norm-sq of 2.
*)

(** a1 = e1 *)
Definition zd_a1_fundamental : CDOct := oct_e 1.

(** a2 = e2 *)
Definition zd_a2_fundamental : CDOct := oct_e 2.

(** b1 = e4 *)
Definition zd_b1_fundamental : CDOct := oct_e 4.

(** b2 = -e7 = (0, 0, 0, 0, 0, 0, 0, -1) in the basis representation *)
Definition zd_b2_fundamental : CDOct :=
  mkOct quat_zero (mkQuat 0 0 0 (-1)).

(** The fundamental pair satisfies the norm condition of criterion (i). *)
Lemma zd_fundamental_condition_i :
  zd_condition_i zd_a1_fundamental zd_a2_fundamental.
Proof.
  unfold zd_condition_i, zd_a1_fundamental, zd_a2_fundamental, oct_e, oct_norm_sq.
  cbv [oct_lo oct_hi quat_norm_sq qa qb qc qd quat_one quat_zero].
  ring.
Qed.

(** ================================================================== *)
(** * Strategy for full theorem proof (Phase C continuation).           *)
(** ================================================================== *)

(** The complete theorem would require:

    1. Prove zd_condition_ii for the fundamental pair
       (requires computation of (a1*b1)*a2 = oct_mul (oct_mul a1 b1) a2)

    2. Prove zd_condition_iii for the fundamental pair
       (requires showing oct_antiassociator a1 b1 a2 = oct_zero)

    3. Combine to show is_zd_pair_major_theorem holds for fundamental pair

    4. Prove generation: all 168 discovered pairs are scalar multiples
       and basis permutations of the fundamental pair

    5. Establish extraction equivalence: Rust emulator produces pairs
       satisfying the criterion

    These steps are deferred to Phase C continuation (Steps 22-30).
*)

(** ================================================================== *)
(** * Verification lemmas (computational support).                      *)
(** ================================================================== *)

(** Both a1 and a2 have unit norm in the fundamental pair. *)
Lemma zd_a1_norm : oct_norm_sq zd_a1_fundamental = 1.
Proof.
  unfold zd_a1_fundamental, oct_e, oct_norm_sq, oct_lo, oct_hi.
  cbv [quat_norm_sq qa qb qc qd quat_one quat_zero].
  ring.
Qed.

Lemma zd_a2_norm : oct_norm_sq zd_a2_fundamental = 1.
Proof.
  unfold zd_a2_fundamental, oct_e, oct_norm_sq, oct_lo, oct_hi.
  cbv [quat_norm_sq qa qb qc qd quat_one quat_zero].
  ring.
Qed.

(** b1 has unit norm. *)
Lemma zd_b1_norm : oct_norm_sq zd_b1_fundamental = 1.
Proof.
  unfold zd_b1_fundamental, oct_e, oct_norm_sq, oct_lo, oct_hi.
  cbv [quat_norm_sq qa qb qc qd quat_one quat_zero].
  ring.
Qed.

(** b2 has unit norm. *)
Lemma zd_b2_norm : oct_norm_sq zd_b2_fundamental = 1.
Proof.
  unfold zd_b2_fundamental, oct_norm_sq, oct_lo, oct_hi.
  cbv [quat_norm_sq qa qb qc qd quat_one quat_zero].
  ring.
Qed.

(** ================================================================== *)
(** * Condition (ii): b2 = [(a1*b1)*a2] / N(a1) for fundamental pair. *)
(** ================================================================== *)

(** The whitelist for cbv unfolding of concrete octonionic arithmetic. *)

(** Condition (ii) holds for the fundamental pair:
    oct_scale N(a1) b2 = oct_mul (oct_mul a1 b1) a2, i.e., 1 * (-e7) = (e1*e4)*e2.

    In octonion arithmetic: e1*e4 = e5 (Fano line {1,4,5}), then e5*e2 = -e7
    (from line {2,5,7}: e2*e5 = e7, so e5*e2 = -e7). And N(e1) = 1.
    Therefore 1 * (-e7) = -e7 = (e1*e4)*e2. Proof: cbv + ring. *)
Lemma zd_fundamental_condition_ii :
  zd_condition_ii zd_a1_fundamental zd_a2_fundamental
                  zd_b1_fundamental zd_b2_fundamental.
Proof.
  unfold zd_condition_ii,
    zd_a1_fundamental, zd_a2_fundamental, zd_b1_fundamental, zd_b2_fundamental, oct_e.
  cbv [oct_norm_sq quat_norm_sq oct_scale oct_mul oct_conj oct_add
       quat_mul quat_add quat_neg quat_conj quat_scale
       oct_zero quat_zero quat_one
       oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.

Lemma zd_fundamental_condition_ii_fused_aux :
  oct_scale (oct_norm_sq zd_a1_fundamental) zd_b2_fundamental =
  oct_mul_fused (oct_mul_fused zd_a1_fundamental zd_b1_fundamental)
                zd_a2_fundamental.
Proof.
  destruct oct_fused_bilinear_surface as [_ _ _ HscaleL HscaleR].
  unfold zd_a1_fundamental, zd_a2_fundamental, zd_b1_fundamental, zd_b2_fundamental.
  cbv [oct_norm_sq quat_norm_sq oct_scale oct_lo oct_hi qa qb qc qd quat_one quat_zero].
  rewrite oct_mul_fused_basis_xor with (i := 1%nat) (j := 4%nat) by lia.
  rewrite HscaleL.
  rewrite oct_mul_fused_basis_xor with (i := Nat.lxor 1%nat 4%nat) (j := 2%nat).
  2: { vm_compute; lia. }
  2: { lia. }
  vm_compute.
  f_equal; f_equal; ring.
Qed.

Lemma zd_fundamental_condition_ii_fused :
  zd_condition_ii zd_a1_fundamental zd_a2_fundamental
                  zd_b1_fundamental zd_b2_fundamental.
Proof.
  unfold zd_condition_ii.
  change
    (oct_scale (oct_norm_sq zd_a1_fundamental) zd_b2_fundamental =
     oct_mul_fused (oct_mul_fused zd_a1_fundamental zd_b1_fundamental)
                   zd_a2_fundamental).
  exact zd_fundamental_condition_ii_fused_aux.
Qed.

(** ================================================================== *)
(** * Condition (iii): antiassociator = 0 for the fundamental pair.   *)
(** ================================================================== *)

(** Sign convention note (verified by cbv computation):
    In this codebase's oct_mul CD-doubling convention:
      e1 * e4 = e5     (lo=0, hi=i)
      e5 * e2 = -e7    (lo=0, hi=-k)
      e4 * e2 = -e6    (lo=0, hi=-j)
      e1 * (-e6) = e7  (lo=0, hi=k)  -- note: e1*e6 = -e7 in this convention

    Therefore for the fundamental pair (a1=e1, b1=e4, a2=e2):
      (a1*b1)*a2 = (e1*e4)*e2 = e5*e2  = -e7
      a1*(b1*a2) = e1*(e4*e2) = e1*(-e6) = e7
      SUM  = -e7 + e7 = 0   (antiassociator = ZERO)
      DIFF = -e7 - e7 = -2*e7  (classical associator = non-zero)

    The ORIGINAL zd_condition_iii using the SUM form (oct_antiassociator)
    is therefore the correct condition and IS zero for this pair.
    The classical associator (DIFFERENCE form) is NOT used here. *)

(** Condition (iii) holds for the fundamental pair: the antiassociator
    (a1*b1)*a2 + a1*(b1*a2) = -e7 + e7 = 0.
    Strategy: unfold all definitions, cbv on arithmetic ops,
    then decompose the CDOct equality to R components via f_equal. *)
Lemma zd_fundamental_condition_iii :
  zd_condition_iii zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental.
Proof.
  unfold zd_condition_iii, oct_antiassociator,
    zd_a1_fundamental, zd_b1_fundamental, zd_a2_fundamental, oct_e.
  cbv [oct_mul oct_conj oct_add oct_lo oct_hi oct_zero
       quat_mul quat_add quat_neg quat_conj quat_one quat_zero qa qb qc qd].
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma zd_fundamental_condition_iii_fused_aux :
  oct_add
    (oct_mul_fused (oct_mul_fused zd_a1_fundamental zd_b1_fundamental)
                   zd_a2_fundamental)
    (oct_mul_fused zd_a1_fundamental
                   (oct_mul_fused zd_b1_fundamental zd_a2_fundamental)) =
  oct_zero.
Proof.
  destruct oct_fused_bilinear_surface as [_ _ _ HscaleL HscaleR].
  unfold zd_a1_fundamental, zd_b1_fundamental, zd_a2_fundamental.
  rewrite oct_mul_fused_basis_xor with (i := 1%nat) (j := 4%nat) by lia.
  rewrite HscaleL.
  rewrite oct_mul_fused_basis_xor with (i := Nat.lxor 1%nat 4%nat) (j := 2%nat).
  2: { vm_compute; lia. }
  2: { lia. }
  rewrite oct_mul_fused_basis_xor with (i := 4%nat) (j := 2%nat) by lia.
  rewrite HscaleR.
  rewrite oct_mul_fused_basis_xor with (i := 1%nat) (j := Nat.lxor 4%nat 2%nat).
  2: { lia. }
  2: { vm_compute; lia. }
  vm_compute.
  apply (f_equal2 mkOct); apply (f_equal4 mkQuat); ring.
Qed.

Lemma zd_fundamental_condition_iii_fused :
  zd_condition_iii zd_a1_fundamental zd_b1_fundamental zd_a2_fundamental.
Proof.
  unfold zd_condition_iii, oct_antiassociator.
  change
    (oct_add
       (oct_mul_fused (oct_mul_fused zd_a1_fundamental zd_b1_fundamental)
                      zd_a2_fundamental)
       (oct_mul_fused zd_a1_fundamental
                      (oct_mul_fused zd_b1_fundamental zd_a2_fundamental)) =
     oct_zero).
  exact zd_fundamental_condition_iii_fused_aux.
Qed.

(** ================================================================== *)
(** * Combined: fundamental pair satisfies all three conditions.       *)
(** ================================================================== *)

(** The fundamental ZD pair satisfies all three Brown Theorem 7.15 conditions.
    Conditions (i), (ii), (iii) are all formally proved for the concrete
    fundamental pair (a1=e1, a2=e2, b1=e4, b2=-e7). *)
Theorem zd_fundamental_major_theorem :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  unfold is_zd_pair_major_theorem.
  split. { exact zd_fundamental_condition_i. }
  split. { exact zd_fundamental_condition_ii. }
  exact zd_fundamental_condition_iii.
Qed.

Theorem zd_fundamental_major_theorem_fused :
  is_zd_pair_major_theorem
    zd_a1_fundamental zd_a2_fundamental
    zd_b1_fundamental zd_b2_fundamental.
Proof.
  unfold is_zd_pair_major_theorem.
  split. { exact zd_fundamental_condition_i. }
  split. { exact zd_fundamental_condition_ii_fused. }
  exact zd_fundamental_condition_iii_fused.
Qed.

(** ================================================================== *)
(** * Remarks on formalization strategy.                                *)
(** ================================================================== *)

(** With zd_fundamental_major_theorem proved, the Brown Thm 7.15 framework
    is grounded in a concrete verified instance. Remaining gaps (S2):

    1. ABSTRACT FORM: Brown's Thm 7.15 for arbitrary A, B in A_3 (not just
       basis elements) requires sedenion algebra over abstract R with symbolic
       variables. This needs a Module Type similar to CDAlgInner in
       C1538_MorZDSymmetry.v but with full sedenion multiplication axioms.

    2. SIGN CONVENTION: The antiassociator (SUM form) is correct for this
       codebase's CD-doubling convention. The classical associator (DIFF form)
       equals -2*e7 for the fundamental pair (non-zero). This codebase uses
       the opposite cyclic orientation from some textbook presentations.

    3. GENERATION PRINCIPLE: Once the fundamental pair is verified, all 168
       ZD pairs should be derivable as G2-orbit images. This requires
       G2OctonionAutomorphisms.v infrastructure.

    4. EQUIVALENCE CLASS (Brown Cor 7.16): All 168 ZD pairs reduce to one
       under the algebra's symmetry group. Requires group action theory.

    These remain structural gaps (S2, S3 in the lacunae audit). *)

(** ================================================================== *)
(** * End of ZD_Criterion.v                                             *)
(** ================================================================== *)
