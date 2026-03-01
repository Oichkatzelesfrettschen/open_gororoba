(** * SpatialAlgebra: 3x3 symmetric tensor operations.

    Defines the trace, trace-free part, and raise/lower operations for
    symmetric 3x3 tensors. Proofs are specialized to the identity metric
    (flat space) for simplicity, matching the unit tests in spatial.rs.

    Mirrors: gr_core/src/spatial.rs lines 177-193 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** A symmetric 3x3 tensor stored as 6 independent components:
    (a00, a01, a02, a11, a12, a22).
    Symmetry: a_{ij} = a_{ji}. *)
Record Sym3 := mkSym3 {
  s00 : R; s01 : R; s02 : R;
  s11 : R; s12 : R;
  s22 : R;
}.

(** Trace of a symmetric tensor w.r.t. the identity metric:
    tr(A) = A_{00} + A_{11} + A_{22}. *)
Definition trace_id (a : Sym3) : R :=
  s00 a + s11 a + s22 a.

(** Trace-free part w.r.t. the identity metric:
    A^TF_{ij} = A_{ij} - (1/3) delta_{ij} tr(A). *)
Definition tracefree_id (a : Sym3) : Sym3 :=
  let tr := trace_id a in
  mkSym3
    (s00 a - tr / 3) (s01 a) (s02 a)
    (s11 a - tr / 3) (s12 a)
    (s22 a - tr / 3).

(** Raise an index: A^{ij} = gamma^{ik} gamma^{jl} A_{kl}.
    For the identity metric, this is the identity operation. *)
Definition raise_id (a : Sym3) : Sym3 := a.

(** Lower an index: A_{ij} = gamma_{ik} gamma_{jl} A^{kl}.
    For the identity metric, this is the identity operation. *)
Definition lower_id (a : Sym3) : Sym3 := a.

(** The trace of the trace-free part vanishes. *)
Theorem trace_tracefree_vanishes : forall a,
  trace_id (tracefree_id a) = 0.
Proof.
  intros a. destruct a.
  unfold trace_id, tracefree_id; simpl.
  unfold trace_id; simpl.
  field.
Qed.

(** Raise then lower is the identity (for flat metric). *)
Theorem raise_lower_roundtrip : forall a,
  lower_id (raise_id a) = a.
Proof. intros. reflexivity. Qed.

(** Lower then raise is the identity (for flat metric). *)
Theorem lower_raise_roundtrip : forall a,
  raise_id (lower_id a) = a.
Proof. intros. reflexivity. Qed.

(** The trace-free part of a traceless tensor is itself. *)
Theorem tracefree_idempotent : forall a,
  trace_id a = 0 -> tracefree_id a = a.
Proof.
  intros a Htr.
  destruct a. unfold tracefree_id, trace_id in *. simpl in *.
  f_equal; lra.
Qed.
