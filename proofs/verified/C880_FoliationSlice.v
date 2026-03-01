(** * C-880: Single-slice foliation reduces to the underlying field.

    Formal proof that a foliation with exactly one time slice has
    its imbalance accessible at slice index 0.

    Mirrors: sedenion_foliation.rs test_single_slice_matches_sedenion_field. *)

From Stdlib Require Import List.
From OpenGororoba Require Import Prelude Foliation.
Import ListNotations.

(** CLAIM C-880: A single-slice foliation's first slice is accessible. *)
Theorem C880_single_slice_reduction :
  forall (s : FoliationSlice),
    nth_slice [s] 0 = Some s.
Proof.
  intro s. simpl. reflexivity.
Qed.

(** A single-slice foliation has no second slice. *)
Theorem C880_single_slice_no_second :
  forall (s : FoliationSlice),
    nth_slice [s] 1 = None.
Proof.
  intro s. simpl. reflexivity.
Qed.

(** Accessing a point in a single-slice foliation works. *)
Theorem C880_single_slice_point_access :
  forall (s : FoliationSlice) (f : Imbalance) (idx : nat),
    nth_error s idx = Some f ->
    imbalance_at [s] 0 idx = Some f.
Proof.
  intros s f idx H.
  unfold imbalance_at. simpl. exact H.
Qed.

(** Multi-slice indexing is consistent. *)
Theorem C880_multi_slice_indexing :
  forall (s1 s2 s3 : FoliationSlice),
    nth_slice [s1; s2; s3] 0 = Some s1 /\
    nth_slice [s1; s2; s3] 1 = Some s2 /\
    nth_slice [s1; s2; s3] 2 = Some s3.
Proof.
  intros. repeat split; reflexivity.
Qed.
