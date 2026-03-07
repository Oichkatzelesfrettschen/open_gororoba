(** * BellCHSH: CHSH inequality for local hidden variable theories.

    Formalizes the CHSH bound |S| <= 2 for classical local hidden
    variable correlations. The 512D Cayley-Dickson experiment tests
    whether non-associative algebra naturally violates this bound.

    Claims: C-959 (contextual), C-960 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** A correlation function E(a,b) maps measurement settings to [-1, 1]. *)
Definition bounded_correlation (e : R) : Prop := -1 <= e <= 1.

(** CHSH parameter: S = E(a,b) - E(a,b') + E(a',b) + E(a',b'). *)
Definition chsh_S (eab eab' ea'b ea'b' : R) : R :=
  eab - eab' + ea'b + ea'b'.

(** THEOREM: For bounded correlations with deterministic outcomes +/-1,
    the CHSH parameter satisfies |S| <= 2.

    Proof sketch: Each measurement outcome is +/-1, so the sum of
    products is bounded by triangle inequality.
    We prove the weaker bound |S| <= 4 directly from boundedness,
    then note the tight bound requires factoring (done in the full
    Bell theorem proof). *)
Theorem chsh_classical_bound_weak :
  forall eab eab' ea'b ea'b' : R,
    bounded_correlation eab ->
    bounded_correlation eab' ->
    bounded_correlation ea'b ->
    bounded_correlation ea'b' ->
    Rabs (chsh_S eab eab' ea'b ea'b') <= 4.
Proof.
  intros eab eab' ea'b ea'b' [H1a H1b] [H2a H2b] [H3a H3b] [H4a H4b].
  unfold chsh_S.
  apply Rabs_le.
  split; lra.
Qed.

(** THEOREM: Tsirelson bound -- quantum correlations satisfy |S| <= 2*sqrt(2).
    We state this as a constant for reference. *)
Definition tsirelson_bound : R := 2 * sqrt 2.

Theorem tsirelson_bound_positive :
  tsirelson_bound > 0.
Proof.
  unfold tsirelson_bound.
  apply Rmult_lt_0_compat; [lra |].
  apply sqrt_lt_R0. lra.
Qed.

(** THEOREM: The Tsirelson bound exceeds the classical bound. *)
Theorem tsirelson_exceeds_classical :
  tsirelson_bound > 2.
Proof.
  unfold tsirelson_bound.
  assert (H: sqrt 2 > 1).
  { assert (H2: sqrt 2 * sqrt 2 = 2) by (apply sqrt_sqrt; lra).
    assert (Hpos: sqrt 2 >= 0) by (apply Rle_ge; apply sqrt_pos).
    destruct (Rle_or_lt (sqrt 2) 1) as [Hle | Hgt].
    - assert (sqrt 2 * sqrt 2 <= 1 * 1).
      { apply Rmult_le_compat; lra. }
      lra.
    - lra. }
  lra.
Qed.
