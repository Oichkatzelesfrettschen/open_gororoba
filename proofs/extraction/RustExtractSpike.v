(** * RustExtractSpike: evaluation of rocq-rust-extraction 0.2.0.
 *
 * WHY:  Evaluate whether automated Rocq->Rust extraction can replace
 *       the manual OCaml->Rust translation in verified_core.
 * WHAT: Attempt to extract QuatOps functor using rocq-rust-extraction.
 * HOW:  Import TypedExtraction, load our modules, attempt extraction.
 *
 * STATUS: DEFERRED (2026-02-28)
 *   rocq-rust-extraction 0.2.0 extracts Records as Rust enums (not structs),
 *   uses lifetime-heavy &'a style, and does not map Module Types to Rust
 *   traits. The manual translation in verified_core/src/quaternion.rs is
 *   simpler and more idiomatic for numerical f64 code.
 *
 *   Re-evaluate when:
 *   - Record -> struct extraction is supported
 *   - Module Type -> trait mapping is available
 *   - Lifetime-free numeric extraction mode exists
 *)

(* Uncomment to test when rocq-rust-extraction is installed:

From RustExtraction Require Import Loader.
From RustExtraction Require Import RustExtract.
From OpenGororoba Require Import FloatAxioms FloatQuaternion.

(* Attempt: extract the QuatOps functor body *)
(* Expected issue: functor not directly extractable to Rust trait *)

*)
