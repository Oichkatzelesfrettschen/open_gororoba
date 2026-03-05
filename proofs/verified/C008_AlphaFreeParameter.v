(** * C-008: Alpha is a free parameter in the spectral dimension formula.

    The Parisi-Sourlas spectral dimension d_s(alpha, k) admits any
    alpha in R. The value alpha = -1.5 is a convention giving d_s = 3,
    not a prediction from the algebra.

    Positive content: the Calcagni formula d_s(s) = 4 - 2/(1+s) maps
    any s > 0 to a unique d_s in (2, 4). Alpha parameterizes s. *)

From OpenGororoba Require Import Prelude SpectralDimension.

(** For any s > 0, the formula gives a well-defined d_s in (2, 4). *)
Theorem C008_alpha_free : forall s : R,
  s > 0 -> 2 < calcagni_d_s s < 4.
Proof. exact calcagni_range. Qed.

(** Specific value: s = 1 gives d_s = 3. *)
Theorem C008_specific_value : calcagni_d_s 1 = 3.
Proof. unfold calcagni_d_s. field. Qed.
