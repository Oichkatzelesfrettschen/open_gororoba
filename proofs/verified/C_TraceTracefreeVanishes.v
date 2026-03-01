(** * C_TraceTracefreeVanishes: trace of trace-free part is zero.

    For the identity metric, the trace-free part of any symmetric tensor
    has vanishing trace. This is the fundamental property that makes
    the trace-free decomposition well-defined. *)

From OpenGororoba Require Import Prelude SpatialAlgebra.

Theorem claim_trace_tracefree_vanishes :
  forall a, trace_id (tracefree_id a) = 0.
Proof. exact trace_tracefree_vanishes. Qed.

Theorem claim_raise_lower_roundtrip :
  forall a, lower_id (raise_id a) = a.
Proof. exact raise_lower_roundtrip. Qed.
