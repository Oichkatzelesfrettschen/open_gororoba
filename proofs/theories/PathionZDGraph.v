(** * PathionZDGraph: Zero-divisor graph structure of the Pathion algebra (dim 32).

    CONVENTION. A vertex is a 2-blade plane span{e_p, e_q} (1 <= p < q < 32)
    that takes part in a zero-product; two planes are adjacent when
    (e_p + s e_q)(e_r + t e_u) = 0 for some signs s, t (the relation is
    symmetric). Vertices without an edge are dropped. Two graphs matter:

    - The complete graph over all index pairs has 22 connected components:
      7 of 12 planes (a sedenion box-kite plus its copy shifted by 16) and
      15 of 14 planes, 294 planes in all.
    - The top-level cross-pair graph (p < 16 <= q) has 15 connected
      components of 14 planes. Each is the emanation table of one strut
      constant S = 1..15 (de Marrais, "Placeholder Substructures III",
      arXiv:math/0703745, section 6: 14 assessors per table, tables for
      S = 9..15 are the "sand mandalas", S <= 8 hold Pleiades of 7 box-kites).

    EPISTEMIC BOUNDARY: the two size lists below are definitions transcribed
    from the Rust census (algebra_analysis::boxkites, tests
    test_zd_plane_components_all_dims and test_motif_census_32d_summary);
    their agreement with the zero-product graph is established there, not
    kernel-checked here, because the Cayley-Dickson sign table at dim=32
    exceeds practical Rocq memory limits for vm_compute. Everything derived
    from the lists (counts, sums, information capacity) is kernel-checked,
    and the file introduces no axioms.

    References:
    - Moreno (1998): The zero divisors of the Cayley-Dickson algebras
    - de Marrais (2000): The 42 Assessors and the Box-Kites They Fly

    Mirrors: algebra_analysis/src/boxkites.rs *)

From Stdlib Require Import Reals Lra Lia.
From Stdlib Require Import ZArith List.
From OpenGororoba Require Import BekensteinEntropy.
Import ListNotations.
Open Scope R_scope.

Definition pathion_dim : nat := 32%nat.

(** Component sizes of the complete plane graph at dim=32. *)
Definition pathion_all_plane_component_sizes : list nat :=
  (repeat 12%nat 7 ++ repeat 14%nat 15)%list.

(** Component sizes of the top-level cross-pair graph at dim=32. *)
Definition pathion_cross_pair_component_sizes : list nat := repeat 14%nat 15.

Lemma pathion_all_plane_components_count :
  length pathion_all_plane_component_sizes = 22%nat.
Proof. reflexivity. Qed.

Lemma pathion_all_plane_vertex_count :
  fold_right Nat.add 0%nat pathion_all_plane_component_sizes = 294%nat.
Proof. reflexivity. Qed.

Lemma pathion_cross_pair_components_count :
  length pathion_cross_pair_component_sizes = 15%nat.
Proof. reflexivity. Qed.

(** Number of strut emanation tables (cross-pair components) at dim=32. *)
Definition pathion_n_components : nat := length pathion_cross_pair_component_sizes.

Lemma pathion_zd_components : pathion_n_components = 15%nat.
Proof. exact pathion_cross_pair_components_count. Qed.

(** The strut-table count equals dim/2 - 1, one table per strut constant. *)
Lemma pathion_components_formula :
  pathion_n_components = (Nat.div pathion_dim 2 - 1)%nat.
Proof.
  rewrite pathion_zd_components. reflexivity.
Qed.

(** Model choice: each strut emanation table is one binary channel, so the
    capacity is 15 * ln 2 nats. The count is the strut-table count of the
    cross-pair graph; the complete plane graph has 22 components. *)
Definition pathion_information_capacity : R :=
  INR pathion_n_components * ln 2.

Lemma pathion_information_positive :
  pathion_information_capacity > 0.
Proof.
  unfold pathion_information_capacity.
  rewrite pathion_zd_components.
  simpl (INR 15).
  assert (Hln2 := ln2_pos).
  nra.
Qed.

(** The Pathion has strictly more ZD channels than the sedenion (dim=16, 7 components). *)
Definition sedenion_n_components : nat := 7%nat.

Lemma pathion_more_channels_than_sedenion :
  (sedenion_n_components < pathion_n_components)%nat.
Proof.
  unfold sedenion_n_components. rewrite pathion_zd_components. lia.
Qed.
