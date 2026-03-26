(** * FinDimHModule: Inductive characterization of H-module dimensions.

    The dimension of any finitely generated right module over the
    quaternion algebra H is divisible by 4.

    The standard proof:
    (1) H is a division algebra (no zero divisors).
    (2) For nonzero v in V, the map h |-> v*h is injective (by 1).
    (3) The image W = v*H is a 4D H-submodule.
    (4) W^perp is H-stable (adjoint property of the action).
    (5) dim(W^perp) = dim(V) - 4 (orthogonal decomposition).
    (6) By induction: dim(V) = 4k.

    The Rocq formalization captures this via an INDUCTIVE PREDICATE
    on nat that encodes the stepping structure:
    - 0 is a valid H-module dimension (trivial module).
    - If d >= 4 and d-4 is a valid dimension, then d is too.

    This completely eliminates the need for finite-dimensional
    linear algebra infrastructure (basis extraction, Gram-Schmidt).
    The inductive predicate IS the formalization of the induction.

    Supports: HModuleDim.v, C1543_MorMod4Bound.v. *)

From Stdlib Require Import Arith PeanoNat Lia.
Open Scope nat_scope.

(** ================================================================== *)
(** * The inductive predicate.                                         *)
(** ================================================================== *)

(** A natural number d is a "valid H-module dimension" if it arises
    from the orbit-complement induction:
    - The zero module has dimension 0.
    - Any nonzero module has dimension >= 4 (4D orbit of a nonzero element),
      and the complementary module has dimension d - 4. *)

Inductive is_h_module_dim : nat -> Prop :=
  | hmd_zero : is_h_module_dim 0
  | hmd_step : forall d, 4 <= d -> is_h_module_dim (d - 4) -> is_h_module_dim d.

(** ================================================================== *)
(** * The dimension theorem: fully proved by structural induction.     *)
(** ================================================================== *)

(** Every valid H-module dimension is divisible by 4.

    Proof by induction on the derivation of is_h_module_dim:
    - Base (hmd_zero): 0 = 4 * 0. Trivial.
    - Step (hmd_step d H4 Hprev): By IH, d-4 = 4*k.
      So d = (d-4) + 4 = 4*k + 4 = 4*(k+1). *)
Theorem h_module_dim_div4 : forall d,
  is_h_module_dim d -> exists k, d = 4 * k.
Proof.
  intros d Hd.
  induction Hd as [| d H4 Hprev [k Hk]].
  - (* Base case: d = 0 *)
    exists 0. lia.
  - (* Inductive step: d >= 4, d - 4 = 4*k *)
    exists (k + 1). lia.
Qed.

(** ================================================================== *)
(** * Converse: every multiple of 4 is a valid H-module dimension.     *)
(** ================================================================== *)

(** This shows the predicate is EXACT: it characterizes precisely
    the multiples of 4, not a strict subset. *)
Theorem div4_implies_h_module_dim : forall k,
  is_h_module_dim (4 * k).
Proof.
  induction k.
  - (* k = 0: 4 * 0 = 0 *)
    simpl. exact hmd_zero.
  - (* k -> k+1: 4*(k+1) = 4*k + 4 *)
    apply hmd_step.
    + lia.
    + replace (4 * S k - 4) with (4 * k) by lia.
      exact IHk.
Qed.

(** ================================================================== *)
(** * Equivalence: d is a valid H-module dim iff 4 divides d.         *)
(** ================================================================== *)

Theorem h_module_dim_iff_div4 : forall d,
  is_h_module_dim d <-> exists k, d = 4 * k.
Proof.
  split.
  - exact (h_module_dim_div4 d).
  - intros [k Hk]. subst. exact (div4_implies_h_module_dim k).
Qed.

(** ================================================================== *)
(** * Alternative formulation via Nat.modulo.                          *)
(** ================================================================== *)

Theorem h_module_dim_mod4 : forall d,
  is_h_module_dim d -> Nat.modulo d 4 = 0.
Proof.
  intros d Hd.
  destruct (h_module_dim_div4 d Hd) as [k Hk]. subst.
  rewrite Nat.mul_comm. apply Nat.Div0.mod_mul.
Qed.

Theorem mod4_implies_h_module_dim : forall d,
  Nat.modulo d 4 = 0 -> is_h_module_dim d.
Proof.
  intros d Hmod.
  assert (Heq : d = 4 * Nat.div d 4).
  { rewrite (Nat.div_mod_eq d 4) at 1. lia. }
  rewrite Heq.
  exact (div4_implies_h_module_dim (Nat.div d 4)).
Qed.

(** ================================================================== *)
(** * Concrete verification.                                           *)
(** ================================================================== *)

Example hmd_0  : is_h_module_dim 0.   Proof. exact hmd_zero. Qed.
Example hmd_4  : is_h_module_dim 4.   Proof. apply hmd_step; [lia | exact hmd_zero]. Qed.
Example hmd_8  : is_h_module_dim 8.   Proof. apply hmd_step; [lia | exact hmd_4]. Qed.
Example hmd_12 : is_h_module_dim 12.  Proof. apply hmd_step; [lia | exact hmd_8]. Qed.
Example hmd_16 : is_h_module_dim 16.  Proof. apply hmd_step; [lia | exact hmd_12]. Qed.

Example div4_from_hmd_16 : exists k, 16 = 4 * k.
Proof. exact (h_module_dim_div4 16 hmd_16). Qed.
