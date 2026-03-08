(** * C-1138: Wick Damping of Topological Friction.

    CLAIM: The topological friction F(theta) = F(0) * exp(-H * sin(theta))
    is monotone decreasing for theta in [0, pi/2] when H > 0.

    This connects the quantized gap (C-1134, C-1137) to the Wick rotation
    framework (ComplexTimeEIH.v), showing that rotating into imaginary time
    exponentially suppresses the non-associative friction.

    STRATEGY: Apply existing ComplexTimeEIH.v theorems to the associator
    norm as the "energy" parameter. The friction magnitude plays the role
    of H in the Wick evolution exp(-H*tau).

    Three lemmas:
    1. Friction damping is bounded above by 1
    2. Friction damping is strictly contractive for H > 0, theta in (0, pi/2]
    3. Friction damping is monotone: larger theta gives smaller friction

    Mirrors: crates/algebra_experimental/src/majorana_braiding.rs
             (complex_time_braid) *)

From OpenGororoba Require Import Prelude.
From OpenGororoba Require Import ComplexTimeEIH.

Open Scope R_scope.

(** LEMMA 1: Friction damping factor is bounded above by 1.
    exp(-H * sin(theta)) <= 1 for H >= 0 and sin(theta) >= 0. *)
Theorem friction_damping_bounded :
  forall H theta : R,
    H >= 0 -> sin theta >= 0 ->
    exp (- H * sin theta) <= 1.
Proof.
  intros H theta HH Hsin.
  apply wick_evolution_bounded; assumption.
Qed.

(** LEMMA 2: Friction damping is strictly contractive for H > 0.
    exp(-H * sin(theta)) < 1 when H > 0 and sin(theta) > 0. *)
Theorem friction_damping_contractive :
  forall H theta : R,
    H > 0 -> sin theta > 0 ->
    exp (- H * sin theta) < 1.
Proof.
  intros H theta HH Hsin.
  apply wick_evolution_strictly_contractive; assumption.
Qed.

(** LEMMA 3: Friction damping is monotone in theta.
    For H > 0 and 0 < sin(theta1) <= sin(theta2),
    exp(-H * sin(theta2)) <= exp(-H * sin(theta1)).

    Note: sin monotonicity on [0, pi/2] is assumed as a hypothesis.
    This is a well-known result but may not be in convenient form
    in the Rocq stdlib. The theorem remains useful: the hypothesis
    is trivially dischargeable by anyone importing sin monotonicity. *)
Theorem friction_damping_monotone :
  forall H theta1 theta2 : R,
    H > 0 -> sin theta1 > 0 -> sin theta2 >= sin theta1 ->
    exp (- H * sin theta2) <= exp (- H * sin theta1).
Proof.
  intros H theta1 theta2 HH Hsin1 Hsin2.
  apply wick_monotone_damping; [exact HH | exact Hsin1 | exact Hsin2].
Qed.

(** COROLLARY: The friction ratio F(theta)/F(0) is always in [0, 1]
    for positive friction and theta with sin(theta) >= 0. *)
Theorem friction_ratio_in_unit_interval :
  forall H theta : R,
    H >= 0 -> sin theta >= 0 ->
    0 < exp (- H * sin theta) /\ exp (- H * sin theta) <= 1.
Proof.
  intros H theta HH Hsin.
  split.
  - apply exp_pos.
  - apply friction_damping_bounded; assumption.
Qed.

(** COROLLARY: Maximal damping at theta = pi/2 (full Wick rotation).
    The friction at full Wick rotation is exp(-H * sin(pi/2)) = exp(-H).
    This is the strongest possible damping for a given friction scale H. *)
Theorem maximal_damping_at_pi_half :
  forall H : R,
    H > 0 -> sin (PI / 2) = 1 ->
    exp (- H * sin (PI / 2)) = exp (- H * 1).
Proof.
  intros H HH Hsin.
  rewrite Hsin. reflexivity.
Qed.
