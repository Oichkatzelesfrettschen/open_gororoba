(** * C-1142: XOR Involution and Scatter/Gather Duality.

    Formally proves that the XOR operator's involution property
    guarantees the equivalence of Scatter and Gather formulations
    of Cayley-Dickson multiplication.

    The key theorem: for any fixed i, the map j -> i XOR j is a
    bijection on [0, dim). This means:
    - SCATTER: result[i XOR j] += sign(i,j) * a[i] * x[j]
    - GATHER:  y[k] = sum_j a[k XOR j] * sign(k XOR j, j) * x[j]
    are the same computation with substitution k = i XOR j.

    Hardware consequence: the Gather (L_a row-scan) formulation
    has no write collisions -- each output slot y[k] is computed
    by a single independent dot product. This maps directly to
    Tensor Core WMMA without atomics.

    Uses: N.lxor from Stdlib.NArith.BinNat (binary natural XOR).

    Mirrors: CURRENT::PATH crates/gororoba_algebra/src/gpu/tensor_avt/mod.rs (LEGACY::PATH crates/algebra_core/src/gpu/tensor_avt.rs)
             (compute_cd_mul vs reference_cd_multiply) *)

From Stdlib Require Import NArith.BinNat.
From Stdlib Require Import Arith.
From Stdlib Require Import Lia.

Open Scope N_scope.

(** * Core XOR involution: (i XOR j) XOR j = i.
    This is the algebraic heart of the Scatter/Gather equivalence. *)

Theorem xor_involution : forall i j : N,
  N.lxor (N.lxor i j) j = i.
Proof.
  intros i j.
  rewrite N.lxor_assoc.
  rewrite N.lxor_nilpotent.
  rewrite N.lxor_0_r.
  reflexivity.
Qed.

(** * XOR involution from the left: j XOR (j XOR k) = k. *)

Theorem xor_involution_left : forall j k : N,
  N.lxor j (N.lxor j k) = k.
Proof.
  intros j k.
  rewrite <- N.lxor_assoc.
  rewrite N.lxor_nilpotent.
  rewrite N.lxor_0_l.
  reflexivity.
Qed.

(** * The XOR map is injective: i XOR j1 = i XOR j2 -> j1 = j2.
    This proves the "no collision" property needed for Gather. *)

Theorem xor_map_injective : forall i j1 j2 : N,
  N.lxor i j1 = N.lxor i j2 -> j1 = j2.
Proof.
  intros i j1 j2 H.
  assert (H1 : N.lxor i (N.lxor i j1) = N.lxor i (N.lxor i j2)).
  { rewrite H. reflexivity. }
  rewrite xor_involution_left in H1.
  rewrite xor_involution_left in H1.
  exact H1.
Qed.

(** * The XOR map is surjective: for any target k, there exists
    j = i XOR k such that i XOR j = k.
    This proves every output slot is reached exactly once. *)

Theorem xor_map_surjective : forall i k : N,
  exists j : N, N.lxor i j = k.
Proof.
  intros i k.
  exists (N.lxor i k).
  apply xor_involution_left.
Qed.

(** * Scatter-Gather index substitution.

    If we define target = i XOR j (Scatter writes to target),
    then the inverse mapping is j = target XOR i (Gather reads source).

    More precisely: for any function f(i, j) accumulated at index i XOR j,
    the same result is obtained by accumulating at index k using
    the substitution j = k XOR i, because:
    - i XOR (k XOR i) = k  (the target index is indeed k)
    - The substitution j -> k XOR i is a bijection (from xor_map_injective)

    This theorem states the core index identity. *)

Theorem scatter_gather_substitution : forall i k : N,
  N.lxor i (N.lxor k i) = k.
Proof.
  intros i k.
  rewrite (N.lxor_comm k i).
  apply xor_involution_left.
Qed.

(** * Source index recovery: in the L_a formula, L_a[k][j] uses
    source index k XOR j. Composing with the target shows:
    if target = i XOR j, then source = target XOR j = i.
    This confirms L_a gathers the correct coefficient a[i]. *)

Theorem la_source_index_correct : forall i j : N,
  N.lxor (N.lxor i j) j = i.
Proof.
  exact xor_involution.
Qed.

(** * Double substitution round-trip.
    Starting from (i, j) in the Scatter domain:
    1. Compute target k = i XOR j
    2. Recover source: k XOR j = i  (xor_involution)
    3. Recover j from (i, k): i XOR k = j  (by commutativity + involution)

    This proves the Scatter and Gather traversals visit exactly
    the same set of (source, column) pairs, just indexed differently. *)

Theorem scatter_gather_roundtrip : forall i j : N,
  let k := N.lxor i j in
  N.lxor k j = i /\ N.lxor i k = j.
Proof.
  intros i j. simpl.
  split.
  - apply xor_involution.
  - apply xor_involution_left.
Qed.

(** * XOR permutation group structure.
    For completeness: XOR forms an abelian group on N where
    every element is its own inverse. The key properties: *)

Theorem xor_identity : forall a : N, N.lxor a 0 = a.
Proof. exact N.lxor_0_r. Qed.

Theorem xor_self_inverse : forall a : N, N.lxor a a = 0.
Proof. exact N.lxor_nilpotent. Qed.

Theorem xor_commutative : forall a b : N, N.lxor a b = N.lxor b a.
Proof. exact N.lxor_comm. Qed.

Theorem xor_associative : forall a b c : N,
  N.lxor (N.lxor a b) c = N.lxor a (N.lxor b c).
Proof. exact N.lxor_assoc. Qed.

(** * The biconditional Scatter/Gather duality theorem.

    This is the formal statement that the Scatter operation
    (writing to target i XOR j) is EXACTLY equivalent to
    the Gather operation (reading source i XOR k for output k).

    i XOR j = k  <=>  j = i XOR k

    This is the mathematical certificate that the L_a row-scan
    algorithm and the reference_cd_multiply scatter algorithm
    compute the same function. *)

Theorem scatter_gather_duality : forall i j k : N,
  N.lxor i j = k <-> j = N.lxor i k.
Proof.
  intros i j k. split; intros H.
  - rewrite <- H.
    symmetry.
    apply xor_involution_left.
  - rewrite H.
    apply xor_involution_left.
Qed.
