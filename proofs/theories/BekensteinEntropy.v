(** * BekensteinEntropy: Bekenstein-Hawking entropy and combinatorial channels.

    Defines the Bekenstein-Hawking entropy S = A / (4 l_P^2) and proves
    basic structural properties: positivity, monotonicity in area,
    and the Schwarzschild special case.

    The combinatorial entropy of n discrete channels is n * ln 2,
    providing an information-theoretic interpretation of algebraic
    structure (e.g., box-kite connected components in CD algebras).

    References:
    - Bekenstein (1973): Phys. Rev. D 7, 2333
    - Hawking (1975): Commun. Math. Phys. 43, 199
    - Strominger & Vafa (1996): Phys. Lett. B 379, 99

    Mirrors: gr_core/src/hawking.rs *)

From Stdlib Require Import Reals Lra Psatz.
Open Scope R_scope.

(** ln 2 > 0, derived from ln_lt_2 : 1/2 < ln 2. *)
Lemma ln2_pos : ln 2 > 0.
Proof.
  assert (H := ln_lt_2). lra.
Qed.

(** Bekenstein-Hawking entropy: S_BH = A / (4 * l_P^2). *)
Definition bekenstein_entropy (area l_P_sq : R) : R :=
  area / (4 * l_P_sq).

(** Positivity: S_BH > 0 when A > 0 and l_P^2 > 0. *)
Lemma bekenstein_entropy_positive : forall area l_P_sq,
  area > 0 -> l_P_sq > 0 ->
  bekenstein_entropy area l_P_sq > 0.
Proof.
  intros area l_P_sq Ha Hl.
  unfold bekenstein_entropy.
  apply Rdiv_lt_0_compat; lra.
Qed.

(** Monotonicity: larger area => larger entropy. *)
Lemma bekenstein_entropy_monotone : forall a1 a2 l_P_sq,
  a1 < a2 -> l_P_sq > 0 ->
  bekenstein_entropy a1 l_P_sq < bekenstein_entropy a2 l_P_sq.
Proof.
  intros a1 a2 l_P_sq Hlt Hl.
  unfold bekenstein_entropy, Rdiv.
  apply Rmult_lt_compat_r.
  - apply Rinv_0_lt_compat. lra.
  - exact Hlt.
Qed.

(** Schwarzschild entropy: for M > 0, A = 16 pi M^2 / c^4 (natural units: c=G=1).
    We state this structurally: S = 4 pi M^2 in natural units. *)
Definition schwarzschild_entropy (mass : R) : R :=
  4 * PI * mass ^ 2.

Lemma schwarzschild_entropy_positive : forall mass,
  mass > 0 -> schwarzschild_entropy mass > 0.
Proof.
  intros mass Hm.
  unfold schwarzschild_entropy.
  assert (Hpi : PI > 0) by (exact PI_RGT_0).
  assert (Hm2 : mass * mass > 0) by nra.
  simpl. nra.
Qed.

(** Combinatorial entropy of n discrete information channels.
    S_comb = n * ln 2 (each channel contributes 1 bit). *)
Definition combinatorial_entropy (n : R) : R :=
  n * ln 2.

(** Combinatorial entropy is positive for n > 0. *)
Lemma combinatorial_entropy_positive : forall n,
  n > 0 -> combinatorial_entropy n > 0.
Proof.
  intros n Hn.
  unfold combinatorial_entropy.
  assert (Hln2 := ln2_pos).
  nra.
Qed.

(** Combinatorial entropy is monotone in channel count. *)
Lemma combinatorial_entropy_monotone : forall n1 n2,
  n1 < n2 -> combinatorial_entropy n1 < combinatorial_entropy n2.
Proof.
  intros n1 n2 Hlt.
  unfold combinatorial_entropy.
  assert (Hln2 := ln2_pos).
  nra.
Qed.
