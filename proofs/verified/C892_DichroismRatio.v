(** * C-892: Euler-Heisenberg dichroism ratio is 4:7.

    The birefringence coefficients for vacuum in a strong field are
    8/45 (parallel) and 14/45 (perpendicular).  Their ratio is 4/7.

    Mirrors C-823/C-831 in registry. *)

From OpenGororoba Require Import Prelude.

Theorem C892_eh_dichroism_ratio :
  (8 / 45) / (14 / 45) = 4 / 7.
Proof. field. Qed.

Theorem C892_eh_ratio_simplified :
  8 / 14 = 4 / 7.
Proof. field. Qed.
