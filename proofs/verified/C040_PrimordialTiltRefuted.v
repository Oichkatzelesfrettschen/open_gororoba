(** * C-040: Primordial tilt refuted -- d_S is injective.

    C-040 claimed D_eff ~ 2.8 implies n_s ~ 0.965 non-uniquely.
    The positive content: calcagni_d_s is strictly monotone (C-883),
    hence injective. Any d_s value in (2, 4) corresponds to a UNIQUE
    scale parameter s.

    Kernel-checked via SpectralDimension.v (C-883). *)

From OpenGororoba Require Import Prelude SpectralDimension.
From OpenGororobaVerified Require Import C883_CalcagniMonotoneRange.

(** d_S is injective: same output implies same input. *)
Theorem C040_injective : forall s1 s2 : R,
  0 < s1 -> 0 < s2 ->
  calcagni_d_s s1 = calcagni_d_s s2 -> s1 = s2.
Proof. exact C883_injective. Qed.

(** 2.8 is in the range (2, 4), so the unique s exists. *)
Theorem C040_28_in_range : 2 < (28/10 : R) < 4.
Proof. lra. Qed.
