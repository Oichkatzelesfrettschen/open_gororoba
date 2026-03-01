(** * Sedenion foliation: abstract model of time slices.

    Models the SedenionFoliation from sedenion_foliation.rs.
    A foliation is a sequence of imbalance densities on hypersurfaces.
    The key property: a single-slice foliation trivially reduces to
    the underlying field. *)

From Stdlib Require Import List.
From OpenGororoba Require Import Prelude.
Import ListNotations.

(** Abstract imbalance value on a spatial point. *)
Definition Imbalance := R.

(** A foliation slice is a list of imbalance values (one per grid point). *)
Definition FoliationSlice := list Imbalance.

(** A foliation is a list of slices (one per time step). *)
Definition Foliation := list FoliationSlice.

(** Access a specific slice of a foliation (0-indexed). *)
Fixpoint nth_slice (fol : Foliation) (n : nat) : option FoliationSlice :=
  match fol, n with
  | s :: _, O => Some s
  | _ :: rest, S m => nth_slice rest m
  | [], _ => None
  end.

(** Access a specific point within a slice. *)
Definition imbalance_at (fol : Foliation) (slice_idx point_idx : nat)
  : option Imbalance :=
  match nth_slice fol slice_idx with
  | Some s => nth_error s point_idx
  | None => None
  end.
