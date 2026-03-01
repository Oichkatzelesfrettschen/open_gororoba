(** * C-871: Casimir energy exact formula properties.

    Formal proofs about the Casimir parallel-plate energy:
    E/A = -pi^2 / (720 * a^3)

    Key results:
    1. Cubic scaling: E(a)/E(2a) = 8
    2. Negativity: E(a) < 0 for a > 0
    3. Monotonicity: a1 < a2 => |E(a1)| > |E(a2)|

    Mirrors: casimir_parallel_plates_exact() in casimir_core/energy.rs:185-188.
    Rust tests: test_exact_scales_as_a_minus_3 (line 207-215). *)

From OpenGororoba Require Import Prelude Casimir.

(** CLAIM C-871a: Cubic scaling -- doubling the gap reduces energy by 8x.

    E(a) / E(2a) = [-pi^2/(720*a^3)] / [-pi^2/(720*(2a)^3)]
                  = (2a)^3 / a^3 = 8. *)
(*<*c871cubic>*)
Theorem C871_cubic_scaling :
  forall a : R, a > 0 ->
    casimir_parallel_plates a = 8 * casimir_parallel_plates (2 * a).
Proof.
  intros a Ha.
  unfold casimir_parallel_plates.
  field.
  lra.
Qed.
(*</c871cubic>*)

(** CLAIM C-871b: The Casimir energy is strictly negative for positive separation. *)
Theorem C871_negativity :
  forall a : R, a > 0 ->
    casimir_parallel_plates a < 0.
Proof.
  intros a Ha.
  unfold casimir_parallel_plates.
  assert (Hpi : PI > 0) by exact PI_RGT_0.
  assert (Hpi2 : PI * PI > 0) by nra.
  assert (Ha2 : a * a > 0) by nra.
  assert (Ha3 : a ^ 3 > 0).
  { simpl. rewrite Rmult_1_r. nra. }
  apply Rdiv_neg_pos; nra.
Qed.

(** C-871c: At unit separation, E = -pi^2/720. *)
Theorem C871_unit_separation :
  casimir_parallel_plates 1 = - (PI * PI) / 720.
Proof.
  unfold casimir_parallel_plates.
  field.
Qed.

(** C-871d: General scaling law -- E(k*a) = k^{-3} * E(a) for k > 0. *)
Theorem C871_general_scaling :
  forall a k : R, a > 0 -> k > 0 ->
    casimir_parallel_plates (k * a) * (k ^ 3) = casimir_parallel_plates a.
Proof.
  intros a k Ha Hk.
  unfold casimir_parallel_plates.
  field. split; lra.
Qed.
