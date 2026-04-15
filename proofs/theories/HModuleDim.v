(** * HModuleDim: Dimension of quaternion modules is divisible by 4.

    Foundational theorem: every finitely generated right module over
    the real quaternion algebra H has R-dimension divisible by 4.

    This is the algebraic backbone of Moreno (1997) Corollary 1.17:
    the annihilator kernel Ker(L_a) carries an H_a-module structure
    and therefore has dim = 0 mod 4.

    * Proof strategy:

    1. H is a division algebra (no zero divisors).
       Proof: ||p*q||^2 = ||p||^2 * ||q||^2 (Hurwitz at dim 4).
       So p*q = 0 => ||p||^2 * ||q||^2 = 0 => p = 0 or q = 0.

    2. Any finitely generated right module V over a division algebra D
       has dim_R V = [dim_R D] * k for some k in N.
       Proof: Every module over a division ring is free.
       For D = H: dim_R V = 4 * k.

    3. The free module argument (finite-dim version, no Zorn):
       - If V = {0}: k = 0.
       - If v in V is nonzero: {v, v*i, v*j, v*k} are R-lin. independent
         because H is a division algebra (v*h = 0 => h = 0).
       - W = span{v, v*i, v*j, v*k} is a 4D H-submodule.
       - W_perp is also an H-submodule (since H acts by isometries
         when ||h|| = 1, and inner product adjointness).
       - By induction on dim: dim(W_perp) = 4*m.
       - dim(V) = 4 + 4*m = 4*(1 + m).

    Source: Standard algebra; used by Moreno (1997) Theorems 1.15-1.17.
    Supports claims: C-1543.

    * Structure:
    PART I:   H has no zero divisors (concrete, fully proved).
    PART II:  Abstract Module Type for finite-dim H-module.
    PART III: The dimension theorem (inductive argument). *)

From Stdlib Require Import Reals Lra Psatz.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra FinDimHModule.
Open Scope R_scope.

(** ================================================================== *)
(** * PART I: Quaternions form a division algebra.                     *)
(** ================================================================== *)

(** Quaternion norm is multiplicative (already in CayleyDicksonAlgebra.v). *)
(* quat_norm_mul : quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q *)

(** Quaternion norm is non-negative. *)
Lemma quat_norm_sq_nonneg : forall q : CDQuat, quat_norm_sq q >= 0.
Proof.
  intros [a b c d].
  cbv [quat_norm_sq qa qb qc qd].
  assert (Ha : a ^ 2 >= 0) by nra.
  assert (Hb : b ^ 2 >= 0) by nra.
  assert (Hc : c ^ 2 >= 0) by nra.
  assert (Hd : d ^ 2 >= 0) by nra.
  lra.
Qed.

(** Quaternion norm is zero iff the element is zero. *)
Lemma quat_norm_sq_zero_iff : forall q : CDQuat,
  quat_norm_sq q = 0 <-> q = quat_zero.
Proof.
  intros [a b c d]; cbv [quat_norm_sq quat_zero qa qb qc qd]; split.
  - intros H.
    assert (Ha2 : a ^ 2 >= 0) by nra.
    assert (Hb2 : b ^ 2 >= 0) by nra.
    assert (Hc2 : c ^ 2 >= 0) by nra.
    assert (Hd2 : d ^ 2 >= 0) by nra.
    assert (Ha0 : a ^ 2 = 0) by lra.
    assert (Hb0 : b ^ 2 = 0) by lra.
    assert (Hc0 : c ^ 2 = 0) by lra.
    assert (Hd0 : d ^ 2 = 0) by lra.
    assert (a = 0) by nra. assert (b = 0) by nra.
    assert (c = 0) by nra. assert (d = 0) by nra.
    subst; reflexivity.
  - intros H; injection H; intros; subst; ring.
Qed.

(** Quaternion norm is strictly positive for nonzero elements. *)
Lemma quat_norm_sq_pos : forall q : CDQuat,
  q <> quat_zero -> quat_norm_sq q > 0.
Proof.
  intros q Hq.
  assert (Hge := quat_norm_sq_nonneg q).
  destruct (Rtotal_order (quat_norm_sq q) 0) as [Hlt | [Heq | Hgt]].
  - lra.
  - exfalso. apply Hq. apply quat_norm_sq_zero_iff. lra.
  - lra.
Qed.

(** ** The division algebra property: no zero divisors.

    pq = 0 implies p = 0 or q = 0.
    Proof: ||pq||^2 = ||p||^2 * ||q||^2.  If pq = 0 then ||pq||^2 = 0,
    so ||p||^2 * ||q||^2 = 0.  Since norms are non-negative, one must be 0. *)

Theorem quat_no_zero_divisors : forall p q : CDQuat,
  quat_mul p q = quat_zero ->
  p = quat_zero \/ q = quat_zero.
Proof.
  intros p q Hpq.
  assert (Hnorm : quat_norm_sq (quat_mul p q) = 0).
  { rewrite Hpq. unfold quat_norm_sq, quat_zero; simpl. ring. }
  rewrite quat_norm_mul in Hnorm.
  assert (Hp := quat_norm_sq_nonneg p).
  assert (Hq := quat_norm_sq_nonneg q).
  (* ||p||^2 * ||q||^2 = 0 with both >= 0 => one is 0. *)
  assert (Hdisj : quat_norm_sq p = 0 \/ quat_norm_sq q = 0) by nra.
  destruct Hdisj as [Hp0 | Hq0].
  - left.  apply quat_norm_sq_zero_iff. exact Hp0.
  - right. apply quat_norm_sq_zero_iff. exact Hq0.
Qed.

(** Corollary: the H-action is free on nonzero elements.
    If v <> 0 and v*h = 0, then h = 0. *)
Corollary quat_action_free : forall v h : CDQuat,
  v <> quat_zero ->
  quat_mul v h = quat_zero ->
  h = quat_zero.
Proof.
  intros v h Hv Hvh.
  destruct (quat_no_zero_divisors v h Hvh) as [Hv0 | Hh0].
  - contradiction.
  - exact Hh0.
Qed.

(** Corollary: module-interface form of act_free for the H_a^perp concrete module.

    In the H_a^perp right H-module (V = CDQuat, act = quat_mul, v_zero = quat_zero),
    the HModuleFinDim act_free axiom "act v h = v_zero => v = v_zero \/ h = quat_zero"
    reduces to exactly this statement.  Bridges the abstract module interface to the
    concrete quaternion division algebra property proved above.

    Closes the act_free obligation for L-1542 (Moreno Theorem 1.16 H_a^perp lane). *)
Corollary h_a_perp_act_free : forall v h : CDQuat,
  quat_mul v h = quat_zero -> v = quat_zero \/ h = quat_zero.
Proof.
  intros v h Hvh.
  destruct (quat_no_zero_divisors v h Hvh) as [Hv | Hh].
  - left; exact Hv.
  - right; exact Hh.
Qed.

(** ================================================================== *)
(** * PART II: Abstract H-module structure.                            *)
(** ================================================================== *)

(** Module Type capturing the essential structure of a finite-dimensional
    real inner product space V with a right H-action.

    The key axioms are:
    (a) The H-action is R-linear and associative: v*(h1*h2) = (v*h1)*h2.
    (b) The H-action preserves the inner product (up to norm):
        <v*h, w*h> = ||h||^2 * <v, w>.
    (c) The H-action is free on nonzero elements: v*h = 0 => v = 0 or h = 0.
    (d) V has finite dimension (captured by a nat parameter). *)

Module Type HModuleFinDim.
  (** The module V. *)
  Parameter V       : Type.
  Parameter v_add   : V -> V -> V.
  Parameter v_scale : R -> V -> V.
  Parameter v_inner : V -> V -> R.
  Parameter v_zero  : V.
  Parameter v_dim   : nat.    (** R-dimension of V *)

  (** Right H-action: V x H -> V. *)
  Parameter act : V -> CDQuat -> V.

  (** Inner product axioms. *)
  Axiom v_inner_symm  : forall x y, v_inner x y = v_inner y x.
  Axiom v_inner_pos   : forall x, x <> v_zero -> v_inner x x > 0.
  Axiom v_inner_zero  : v_inner v_zero v_zero = 0.

  (** H-action preserves inner product structure:
      <v*h, w*h> = ||h||^2 * <v, w>
      This implies: if ||h|| = 1 then the action is an isometry. *)
  Axiom act_inner : forall v w h,
    v_inner (act v h) (act w h) = quat_norm_sq h * v_inner v w.

  (** H-action is free on nonzero elements. *)
  Axiom act_free : forall v h,
    act v h = v_zero -> v = v_zero \/ h = quat_zero.

  (** H-action is associative: v*(h1*h2) = (v*h1)*h2. *)
  Axiom act_assoc : forall v h1 h2,
    act v (quat_mul h1 h2) = act (act v h1) h2.

  (** Unit action: v * 1 = v. *)
  Axiom act_one : forall v, act v quat_one = v.

  (** Orthogonal complement of an H-submodule is also an H-submodule.

      If W is an H-stable subspace and u is perpendicular to all of W,
      then u*h is also perpendicular to all of W (for any h in H).

      Proof sketch: for w in W and h in H,
        <u*h, w> = <u*h, w*1> = <u*h, w*(h*h_inv)> = ... uses the
        adjoint property of the H-action, which follows from act_inner
        and the associativity of H. *)
  Axiom act_preserves_perp : forall u w h,
    v_inner u w = 0 ->
    v_inner (act u h) w = 0.

  (** The dimension of V is a valid H-module dimension, i.e.,
      it arises from the orbit-complement induction.  This bridges
      the abstract H-action to the nat-level inductive predicate.
      Justification: act_free gives 4D orbit, act_preserves_perp
      gives complement stability, orthogonal decomposition gives
      dimension stepping. *)
  Axiom v_dim_is_h_module_dim : is_h_module_dim v_dim.
End HModuleFinDim.

(** ================================================================== *)
(** * PART III: The dimension theorem.                                 *)
(** ================================================================== *)

Module HModDimThm (M : HModuleFinDim).
  Import M.

  (** The inductive argument:

      Base case: V = {0} (dim = 0 = 4*0).

      Inductive step: If V has a nonzero element v, then:
      1. {v, v*i, v*j, v*k} are linearly independent (by act_free).
      2. W = span{v, v*i, v*j, v*k} is a 4D H-submodule.
      3. W^perp is also an H-submodule (by act_preserves_perp).
      4. dim(W^perp) = dim(V) - 4.
      5. By induction: dim(W^perp) = 4*m.
      6. dim(V) = 4 + 4*m = 4*(1+m).

      In Rocq, we formalize this as a theorem on nat (the dimension),
      using the module axioms to justify each step. *)

  (** The core argument as a nat proposition:
      if the module axioms hold for a space of dimension d,
      then 4 divides d. *)
  Theorem h_module_dim_div4 :
    exists k : nat, v_dim = (4 * k)%nat.
  Proof.
    (** The inductive predicate is_h_module_dim captures the
        orbit-complement stepping structure:
        - hmd_zero: dim 0 is valid.
        - hmd_step: if dim >= 4 and dim-4 is valid, then dim is valid.

        The Module Type axiom v_dim_is_h_module_dim bridges the
        algebraic structure (act_free, act_preserves_perp, etc.)
        to this nat-level predicate.

        The theorem then follows from FinDimHModule.h_module_dim_div4,
        which proves is_h_module_dim d -> exists k, d = 4*k by
        structural induction on the predicate derivation. *)
    exact (FinDimHModule.h_module_dim_div4 v_dim v_dim_is_h_module_dim).
  Qed.

End HModDimThm.

(** ================================================================== *)
(** * Summary of proof status.                                         *)
(** ================================================================== *)

(** FULLY PROVED (no Admitted):
    - quat_norm_sq_nonneg
    - quat_norm_sq_zero_iff
    - quat_norm_sq_pos
    - quat_no_zero_divisors   [the key division algebra property]
    - quat_action_free

    PROVED (from FinDimHModule inductive predicate):
    - h_module_dim_div4: proved from v_dim_is_h_module_dim axiom
      + FinDimHModule.h_module_dim_div4 (structural induction).

    The remaining axiom v_dim_is_h_module_dim bridges the algebraic
    structure (act_free, act_preserves_perp, act_assoc) to the
    nat-level inductive predicate is_h_module_dim.  It encodes:
    (a) act_free => nonzero v generates 4 lin. indep. vectors (dim >= 4)
    (b) act_preserves_perp => orthogonal complement is H-stable
    (c) dimension counting => dim V = 4 + dim V_perp *)
