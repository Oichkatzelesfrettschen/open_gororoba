(** * ExtractQuaternion: verified-to-OCaml extraction of quaternion operations.

    Extracts the QuatOps functor to OCaml. The functor is parameterized over
    FLOAT_OPS, so the extracted code is a genuine OCaml functor: supply any
    module matching the FLOAT_OPS signature and get verified quaternion ops.

    WHY functor-only: Extracting the concrete RQuat instantiation at R pulls
    in the entire Dedekind-cut reals infrastructure (~1000 lines of dead code).
    Since the trust boundary is the FLOAT_OPS-to-float mapping anyway, the
    functor alone is the correct extraction target.

    Trust boundary: the mapping from FLOAT_OPS axioms to concrete OCaml float
    operations is NOT verified by the kernel. It is the single assumption
    bridging formal proofs to executable code.

    Mirrors: algebra_core/src/physics/quat_rotation.rs *)

From OpenGororoba Require Import FloatAxioms FloatQuaternion.

Require Extraction.
Require Import ExtrOcamlBasic.

(** Extract the abstract functor only -- no dependency on Coq reals. *)
Extraction "extraction/extracted_quat.ml" QuatOps.
