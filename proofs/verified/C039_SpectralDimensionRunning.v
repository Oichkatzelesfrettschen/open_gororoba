(** * C-039: Spectral dimension runs from 4 (IR) to 2 (UV).

    Calcagni's d_S(s) = 4 - 2/(1+s) has range (2, 4) for s > 0,
    and is strictly increasing (larger s = deeper IR gives larger d_S).

    Kernel-checked via SpectralDimension.v (C-883 reformulation). *)

From OpenGororoba Require Import Prelude SpectralDimension.
From OpenGororobaVerified Require Import C883_CalcagniMonotoneRange.

(** Range: d_S in (2, 4) for all s > 0. *)
Theorem C039_range : forall s : R, s > 0 -> 2 < calcagni_d_s s < 4.
Proof. exact calcagni_range. Qed.

(** Monotonicity: d_S is strictly increasing for s > 0. *)
Theorem C039_monotone : forall s1 s2 : R,
  0 < s1 -> s1 < s2 -> calcagni_d_s s1 < calcagni_d_s s2.
Proof. exact calcagni_decreasing. Qed.
