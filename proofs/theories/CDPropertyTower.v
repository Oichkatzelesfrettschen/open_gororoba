(** * CDPropertyTower: Property loss at each Cayley-Dickson doubling.

    The Cayley-Dickson construction doubles dimension at each step,
    and at each doubling (for n >= 2) at least one algebraic property
    is lost:

    - dim 2 (C): commutative, associative, alternative, division algebra
    - dim 4 (H): NON-commutative, associative, alternative, division algebra
    - dim 8 (O): NON-commutative, NON-associative, alternative, division algebra
    - dim 16 (S): NON-commutative, NON-associative, NON-alternative, has zero divisors

    This file consolidates the property tower witnesses into a unified
    framework. Each loss is proved by explicit counterexample (witness).

    References:
    - Schafer (1966): On the algebras formed by the Cayley-Dickson process
    - Baez (2002): The Octonions, Bull. AMS 39(2), 145-205

    Mirrors: cd_kernel/src/cayley_dickson.rs, gr_core/src/cd_ladder_force.rs *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Basis quaternion elements (local definitions for self-containment). *)
Definition tower_qi : CDQuat := mkQuat 0 1 0 0.
Definition tower_qj : CDQuat := mkQuat 0 0 1 0.

(** Basis octonion elements (local definitions). *)
Definition tower_oe1 : CDOct := oct_e 1.
Definition tower_oe2 : CDOct := oct_e 2.
Definition tower_oe4 : CDOct := oct_e 4.

(** Property predicates for the tower. *)

(** Loss 1: Commutativity lost at dim 4.
    Witness: i * j <> j * i. *)
Definition commutativity_lost_at_4 : Prop :=
  quat_mul tower_qi tower_qj <> quat_mul tower_qj tower_qi.

Lemma proof_commutativity_lost_at_4 : commutativity_lost_at_4.
Proof.
  unfold commutativity_lost_at_4, tower_qi, tower_qj, quat_mul.
  intro H. injection H. lra.
Qed.

(** Loss 2: Associativity lost at dim 8.
    Witness: (e1 * e2) * e4 <> e1 * (e2 * e4). *)
Definition associativity_lost_at_8 : Prop :=
  oct_mul (oct_mul tower_oe1 tower_oe2) tower_oe4 <>
  oct_mul tower_oe1 (oct_mul tower_oe2 tower_oe4).

Lemma proof_associativity_lost_at_8 : associativity_lost_at_8.
Proof.
  unfold associativity_lost_at_8.
  intro H.
  (* Same focused proof pattern as C909: project to the single distinguishing
     scalar instead of unfolding the entire octonion equality. *)
  assert (Hd := f_equal (fun x => qd (oct_hi x)) H).
  unfold tower_oe1, tower_oe2, tower_oe4, oct_e in Hd.
  cbv [oct_mul oct_conj oct_hi qd
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       qa qb qc qd oct_lo] in Hd.
  lra.
Qed.

(** Loss 3: Division algebra property lost at dim 16.
    Witness: sed_zd_a * sed_zd_b = 0 with both nonzero (zero divisors). *)
Definition division_lost_at_16 : Prop :=
  sed_mul sed_zd_a sed_zd_b = sed_zero /\
  sed_zd_a <> sed_zero /\ sed_zd_b <> sed_zero.

Lemma proof_division_lost_at_16 : division_lost_at_16.
Proof.
  unfold division_lost_at_16.
  split; [| split].
  - exact sed_zd_product_zero.
  - unfold sed_zd_a, sed_zero, oct_zero, quat_zero.
    intro H.
    assert (Hlo := f_equal sed_lo H). simpl in Hlo.
    assert (Hq := f_equal oct_lo Hlo). simpl in Hq.
    assert (Hd := f_equal qd Hq). simpl in Hd. lra.
  - unfold sed_zd_b, sed_zero, oct_zero, quat_zero.
    intro H.
    assert (Hhi := f_equal sed_hi H). simpl in Hhi.
    assert (Hq := f_equal oct_hi Hhi). simpl in Hq.
    assert (Hd := f_equal qd Hq). simpl in Hd. lra.
Qed.

(** The tower theorem: every doubling from n=2 onward loses at least one property.

    We state this as a conjunction of three concrete losses.
    This is intentionally structural rather than inductive:
    the Cayley-Dickson construction is not a single inductive type in our
    formalization, so we enumerate the first three loss events explicitly. *)
Theorem cd_property_tower :
  commutativity_lost_at_4 /\
  associativity_lost_at_8 /\
  division_lost_at_16.
Proof.
  split; [| split].
  - exact proof_commutativity_lost_at_4.
  - exact proof_associativity_lost_at_8.
  - exact proof_division_lost_at_16.
Qed.
