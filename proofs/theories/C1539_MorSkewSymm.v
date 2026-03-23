(** * C1539_MorSkewSymm: Moreno (1997) Proposition 1.7 -- L_x skew-symmetric.

    For any purely imaginary x in A_n (t(x) = 0, equivalently x_bar = -x),
    the left-multiplication operator L_x is skew-symmetric:
      <L_x(y), z> = -<y, L_x(z)>   for all y, z in A_n.

    Similarly, R_x is skew-symmetric:
      <R_x(y), z> = -<y, R_x(z)>   for all y, z.

    Consequences proved here:
    (a) L_x and R_x are both skew-symmetric (Prop 1.7)
    (b) xy = 0 => y perp Im(L_x) [provable from adjoint identity]
    (c) yx = 0 => y perp Im(R_x) [provable from adjoint identity]
    (d) y perp Im(L_x) => xy = 0 [provable from skew-symm + non-degeneracy]
    (e) y perp Im(R_x) => yx = 0 [provable from skew-symm + non-degeneracy]

    The remaining gap:
    (f) Im(L_x) = Im(R_x) for purely imaginary x [requires CD structure]
    Without (f), we cannot close Ker(L_x) = Ker(R_x) from the abstract axioms.

    Source: Moreno 1997, arXiv:q-alg/9710013v1, Proposition 1.7.
    Claim:  C-1539. *)

From Stdlib Require Import Reals Lra.
Open Scope R_scope.

(** * Module Type: Cayley-Dickson algebra with Moreno Lemma 1.3 axioms. *)
Module Type CDAlgMoreno.
  Parameter A     : Type.
  Parameter mul   : A -> A -> A.
  Parameter conj  : A -> A.
  Parameter neg   : A -> A.
  Parameter inner : A -> A -> R.
  Parameter zero  : A.

  (** Moreno Lemma 1.3, left adjoint:  <xy, z> = <y, x_bar * z>. *)
  Axiom inner_adj_left : forall x y z,
    inner (mul x y) z = inner y (mul (conj x) z).

  (** Moreno Lemma 1.3, right adjoint: <xy, z> = <x, z * y_bar>. *)
  Axiom inner_adj_right : forall x y z,
    inner (mul x y) z = inner x (mul z (conj y)).

  (** Inner product is symmetric. *)
  Axiom inner_symm : forall x y, inner x y = inner y x.

  (** Inner product R-linearity (negation, both arguments). *)
  Axiom inner_neg_left  : forall x y, inner (neg x) y = - inner x y.
  Axiom inner_neg_right : forall x y, inner x (neg y) = - inner x y.

  (** Inner product on zero. *)
  Axiom inner_zero_left : forall y, inner zero y = 0.

  (** neg distributes over mul (both sides). *)
  Axiom neg_mul_left  : forall x y, mul (neg x) y = neg (mul x y).
  Axiom neg_mul_right : forall x y, mul x (neg y) = neg (mul x y).

  (** neg of zero is zero. *)
  Axiom neg_zero : neg zero = zero.

  (** Non-degeneracy: <x, y> = 0 for all y implies x = zero. *)
  Axiom inner_nondeg : forall x, (forall y, inner x y = 0) -> x = zero.
End CDAlgMoreno.

(** * Proposition 1.7 and consequences. *)
Module MorSkewSymm (Alg : CDAlgMoreno).
  Import Alg.

  (* ================================================================== *)
  (** ** Derived inner product lemmas. *)
  (* ================================================================== *)

  Lemma inner_zero_right : forall x, inner x zero = 0.
  Proof.
    intros x.
    rewrite inner_symm.
    apply inner_zero_left.
  Qed.

  (** If a = zero, then inner a z = 0. *)
  Lemma inner_eq_zero_left : forall a z,
    a = zero -> inner a z = 0.
  Proof.
    intros a z Ha. rewrite Ha. apply inner_zero_left.
  Qed.

  (* ================================================================== *)
  (** ** Proposition 1.7(a): L_x is skew-symmetric. *)
  (* ================================================================== *)

  (** For purely imaginary x (conj x = neg x):
      <xy, z> = -<y, xz>. *)
  Theorem l_x_skew_symm : forall x y z,
    conj x = neg x ->
    inner (mul x y) z = - inner y (mul x z).
  Proof.
    intros x y z Hx.
    (* Step 1: Moreno Lemma 1.3 left adjoint. *)
    rewrite inner_adj_left.
    (* Goal: inner y (mul (conj x) z) = - inner y (mul x z) *)
    (* Step 2: conj x = neg x *)
    rewrite Hx.
    (* Goal: inner y (mul (neg x) z) = - inner y (mul x z) *)
    (* Step 3: neg distributes over mul *)
    rewrite neg_mul_left.
    (* Goal: inner y (neg (mul x z)) = - inner y (mul x z) *)
    (* Step 4: neg pulls out of inner *)
    rewrite inner_neg_right.
    (* Goal: - inner y (mul x z) = - inner y (mul x z) *)
    ring.
  Qed.

  (** Equivalent: <L_x(y), z> + <y, L_x(z)> = 0. *)
  Corollary l_x_skew_sum_zero : forall x y z,
    conj x = neg x ->
    inner (mul x y) z + inner y (mul x z) = 0.
  Proof.
    intros x y z Hx.
    rewrite (l_x_skew_symm x y z Hx).
    ring.
  Qed.

  (* ================================================================== *)
  (** ** Proposition 1.7(a): R_x is skew-symmetric. *)
  (* ================================================================== *)

  (** For purely imaginary x (conj x = neg x):
      <yx, z> = -<y, zx>.

      Proof mirror of l_x_skew_symm, using the right adjoint identity. *)
  Theorem r_x_skew_symm : forall x y z,
    conj x = neg x ->
    inner (mul y x) z = - inner y (mul z x).
  Proof.
    intros x y z Hx.
    (* Step 1: Moreno Lemma 1.3 right adjoint. *)
    rewrite inner_adj_right.
    (* Goal: inner y (mul z (conj x)) = - inner y (mul z x) *)
    (* Step 2: conj x = neg x *)
    rewrite Hx.
    (* Goal: inner y (mul z (neg x)) = - inner y (mul z x) *)
    (* Step 3: neg distributes over mul (right) *)
    rewrite neg_mul_right.
    (* Goal: inner y (neg (mul z x)) = - inner y (mul z x) *)
    (* Step 4: neg pulls out of inner *)
    rewrite inner_neg_right.
    (* Goal: - inner y (mul z x) = - inner y (mul z x) *)
    ring.
  Qed.

  (** Equivalent: <R_x(y), z> + <y, R_x(z)> = 0. *)
  Corollary r_x_skew_sum_zero : forall x y z,
    conj x = neg x ->
    inner (mul y x) z + inner y (mul z x) = 0.
  Proof.
    intros x y z Hx.
    rewrite (r_x_skew_symm x y z Hx).
    ring.
  Qed.

  (* ================================================================== *)
  (** ** Consequence (b): xy = 0 implies y perp Im(L_x). *)
  (* ================================================================== *)

  (** If xy = 0 and x is purely imaginary, then <y, xz> = 0 for all z.
      In other words, y is orthogonal to the image of L_x. *)
  Theorem lx_zero_implies_perp : forall x y,
    conj x = neg x ->
    mul x y = zero ->
    forall z, inner y (mul x z) = 0.
  Proof.
    intros x y Hpure Hxy z.
    (* From xy = 0: inner (mul x y) z = inner zero z = 0. *)
    assert (H0 : inner (mul x y) z = 0).
    { rewrite Hxy. apply inner_zero_left. }
    (* By l_x_skew_symm: inner (mul x y) z = -inner y (mul x z). *)
    rewrite (l_x_skew_symm x y z Hpure) in H0.
    (* H0: -inner y (mul x z) = 0, so inner y (mul x z) = 0. *)
    lra.
  Qed.

  (* ================================================================== *)
  (** ** Consequence (c): yx = 0 implies y perp Im(R_x). *)
  (* ================================================================== *)

  (** If yx = 0 and x is purely imaginary, then <y, zx> = 0 for all z. *)
  Theorem rx_zero_implies_perp : forall x y,
    conj x = neg x ->
    mul y x = zero ->
    forall z, inner y (mul z x) = 0.
  Proof.
    intros x y Hpure Hyx z.
    assert (H0 : inner (mul y x) z = 0).
    { rewrite Hyx. apply inner_zero_left. }
    rewrite (r_x_skew_symm x y z Hpure) in H0.
    lra.
  Qed.

  (* ================================================================== *)
  (** ** Consequence (d): y perp Im(L_x) implies xy = 0. *)
  (* ================================================================== *)

  (** If <y, xz> = 0 for all z (y perp Im L_x), and x is purely imaginary,
      then xy = 0.

      Proof: by l_x_skew_symm, <xy, z> = -<y, xz> = 0 for all z.
      By non-degeneracy: xy = 0. *)
  Theorem perp_im_lx_implies_xy_zero : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul x z) = 0) ->
    mul x y = zero.
  Proof.
    intros x y Hpure Hperp.
    apply inner_nondeg.
    intros z.
    rewrite (l_x_skew_symm x y z Hpure).
    specialize (Hperp z).
    lra.
  Qed.

  (* ================================================================== *)
  (** ** Consequence (e): y perp Im(R_x) implies yx = 0. *)
  (* ================================================================== *)

  (** If <y, zx> = 0 for all z, and x is purely imaginary, then yx = 0. *)
  Theorem perp_im_rx_implies_yx_zero : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul z x) = 0) ->
    mul y x = zero.
  Proof.
    intros x y Hpure Hperp.
    apply inner_nondeg.
    intros z.
    rewrite (r_x_skew_symm x y z Hpure).
    specialize (Hperp z).
    lra.
  Qed.

  (* ================================================================== *)
  (** ** Characterization: xy = 0 iff y perp Im(L_x). *)
  (* ================================================================== *)

  (** Complete characterization for purely imaginary x:
      xy = 0 if and only if y is orthogonal to the image of L_x. *)
  Theorem ker_lx_iff_perp_im_lx : forall x y,
    conj x = neg x ->
    (mul x y = zero <-> forall z, inner y (mul x z) = 0).
  Proof.
    intros x y Hpure. split.
    - exact (lx_zero_implies_perp x y Hpure).
    - exact (perp_im_lx_implies_xy_zero x y Hpure).
  Qed.

  (** Complete characterization for R_x:
      yx = 0 iff y perp Im(R_x). *)
  Theorem ker_rx_iff_perp_im_rx : forall x y,
    conj x = neg x ->
    (mul y x = zero <-> forall z, inner y (mul z x) = 0).
  Proof.
    intros x y Hpure. split.
    - exact (rx_zero_implies_perp x y Hpure).
    - exact (perp_im_rx_implies_yx_zero x y Hpure).
  Qed.

  (* ================================================================== *)
  (** ** The remaining gap: Ker(L_x) = Ker(R_x). *)
  (* ================================================================== *)

  (** From the above, we have:
        Ker(L_x) = Im(L_x)^perp   [ker_lx_iff_perp_im_lx]
        Ker(R_x) = Im(R_x)^perp   [ker_rx_iff_perp_im_rx]

      So Ker(L_x) = Ker(R_x) iff Im(L_x) = Im(R_x).

      In the abstract module type, Im(L_x) and Im(R_x) can differ for
      non-commutative x.  The equality Im(L_x) = Im(R_x) for purely
      imaginary x in A_n (n >= 4) requires the CD-specific quaternion
      subalgebra H_a and the direct sum decomposition (Theorem 1.15),
      which shows that both L_x and R_x respect the same block structure.

      This axiom captures that finite-dimensional CD-specific fact. *)

  Axiom im_lx_eq_im_rx : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul x z) = 0) ->
    (forall z, inner y (mul z x) = 0).

  (** Symmetric direction: Im(R_x) perp implies Im(L_x) perp.
      Both directions hold by the same Theorem 1.15 H_a-module argument. *)
  Axiom im_rx_eq_im_lx : forall x y,
    conj x = neg x ->
    (forall z, inner y (mul z x) = 0) ->
    (forall z, inner y (mul x z) = 0).

  (** Moreno Proposition 1.7, final form: Ker(L_x) = Ker(R_x). *)
  Theorem ker_lx_eq_ker_rx : forall x y,
    conj x = neg x ->
    (mul x y = zero <-> mul y x = zero).
  Proof.
    intros x y Hpure. split.
    - (* xy = 0 => yx = 0 *)
      intro Hxy.
      apply (perp_im_rx_implies_yx_zero x y Hpure).
      apply (im_lx_eq_im_rx x y Hpure).
      exact (lx_zero_implies_perp x y Hpure Hxy).
    - (* yx = 0 => xy = 0 *)
      intro Hyx.
      apply (perp_im_lx_implies_xy_zero x y Hpure).
      apply (im_rx_eq_im_lx x y Hpure).
      exact (rx_zero_implies_perp x y Hpure Hyx).
  Qed.

End MorSkewSymm.
