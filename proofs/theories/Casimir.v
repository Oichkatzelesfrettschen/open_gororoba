(** * Casimir energy formula for parallel plates.

    The exact Casimir energy density per unit area between infinite
    parallel conducting plates separated by distance a:

      E/A = -pi^2 / (720 * a^3)

    Mirrors: casimir_parallel_plates_exact() in casimir_core/energy.rs:185-188. *)

From OpenGororoba Require Import Prelude.

(** Exact Casimir energy per unit area for parallel plates. *)
Definition casimir_parallel_plates (a : R) : R :=
  - (PI * PI) / (720 * a ^ 3).
