(** * CardanoEigensolver: Formal verification of the Cardano eigensolver.

    Theorem 9a (depressed_cubic_q_sign):
      The depressed cubic substitution lambda = t + tr/3 transforms
      lambda^3 - tr*lambda^2 + s2*lambda - det = 0 into t^3 + p*t + q = 0
      with q = s2*tr/3 - 2*tr^3/27 - det (CORRECTED sign, fixed 2026-03-22).

    Theorem 9b (cardano_trigonometric_root):
      If p = -3*r^2 and 2*r^3*C3 + q = 0 and C3 = 4*c^3 - 3*c,
      then t = 2*r*c satisfies t^3 + p*t + q = 0.
      (C3 is the value of cos(3u) when c = cos(u), via the triple-angle
      identity.  We verify algebraically without importing trig.)

    Mirrors: hermitian_3x3_eig in neutrino_sector.rs.
*)

From Stdlib Require Import Reals Lra Psatz.
Open Scope R_scope.

(** * Theorem 9a: Depressed cubic q-sign verification.

    The characteristic polynomial of a 3x3 matrix with trace tr,
    second symmetric function s2, and determinant det is:
      lambda^3 - tr*lambda^2 + s2*lambda - det = 0

    The depressed cubic substitution lambda = t + tr/3 gives:
      t^3 + p*t + q = 0
    where p = s2 - tr^2/3 and q = s2*tr/3 - 2*tr^3/27 - det.

    This verifies the CORRECTED code (the original had q negated). *)
Lemma depressed_cubic_q_sign :
  forall tr s2 det : R,
    let shift := tr / 3 in
    let p := s2 - tr ^ 2 / 3 in
    let q := s2 * tr / 3 - 2 * tr ^ 3 / 27 - det in
    forall t : R,
      let lam := t + shift in
      lam ^ 3 - tr * lam ^ 2 + s2 * lam - det = t ^ 3 + p * t + q.
Proof.
  intros tr s2 det shift p q t lam.
  unfold lam, shift, p, q.
  field.
Qed.

(** * Theorem 9b: Trigonometric Cardano root (algebraic core).

    The triple-angle cosine identity cos(3u) = 4*cos^3(u) - 3*cos(u)
    is the algebraic backbone of the Vieta trigonometric solution.

    We abstract: let c be any real number, define C3 := 4*c^3 - 3*c
    (this equals cos(3u) when c = cos(u)).

    If p = -3*r^2, and the "cosine equation" 2*r^3*C3 + q = 0 holds,
    then t = 2*r*c satisfies the depressed cubic t^3 + p*t + q = 0.

    This is a purely algebraic verification -- no trig imports needed. *)
Theorem cardano_trigonometric_root :
  forall r c q : R,
    let C3 := 4 * c ^ 3 - 3 * c in
    let p := - 3 * r ^ 2 in
    2 * r ^ 3 * C3 + q = 0 ->
    let t := 2 * r * c in
    t ^ 3 + p * t + q = 0.
Proof.
  intros r c q C3 p Hcos t.
  unfold t, p, C3 in *.
  (* t^3 + p*t + q
     = (2*r*c)^3 + (-3*r^2)*(2*r*c) + q
     = 8*r^3*c^3 - 6*r^3*c + q
     = 2*r^3*(4*c^3 - 3*c) + q
     = 2*r^3*C3 + q
     = 0   by Hcos *)
  nra.
Qed.

(** * Eigenvalue sum equals trace.

    If the three depressed cubic roots sum to zero (Vieta's formula for
    a monic cubic with zero quadratic coefficient), then adding back the
    shift tr/3 recovers the trace. *)
Lemma eigenvalue_sum_equals_trace :
  forall tr t0 t1 t2 : R,
    t0 + t1 + t2 = 0 ->
    (t0 + tr / 3) + (t1 + tr / 3) + (t2 + tr / 3) = tr.
Proof.
  intros. lra.
Qed.

(** * Eigenvalue product reconstruction.

    The product of the eigenvalues equals the determinant (up to sign).
    Vieta's formula for lambda^3 - tr*lambda^2 + s2*lambda - det = 0:
      lambda_1 * lambda_2 * lambda_3 = det. *)
Lemma vieta_product :
  forall l1 l2 l3 tr s2 det : R,
    l1 ^ 3 - tr * l1 ^ 2 + s2 * l1 - det = 0 ->
    l2 ^ 3 - tr * l2 ^ 2 + s2 * l2 - det = 0 ->
    l3 ^ 3 - tr * l3 ^ 2 + s2 * l3 - det = 0 ->
    l1 + l2 + l3 = tr ->
    l1 * l2 + l1 * l3 + l2 * l3 = s2 ->
    l1 * l2 * l3 = det.
Proof.
  intros l1 l2 l3 tr s2 det H1 H2 H3 Hsum Hpair.
  (* From Vieta: the cubic factors as (l-l1)(l-l2)(l-l3) = 0
     Expanding: l^3 - (l1+l2+l3)*l^2 + (l1l2+l1l3+l2l3)*l - l1l2l3 = 0
     Matching coefficients: det = l1*l2*l3 *)
  (* Use H1 at l = l1: l1^3 - tr*l1^2 + s2*l1 - det = 0 *)
  (* Substitute tr = l1+l2+l3 and s2 = l1l2+l1l3+l2l3: *)
  (* l1^3 - (l1+l2+l3)*l1^2 + (l1l2+l1l3+l2l3)*l1 - det = 0 *)
  (* = l1^3 - l1^3 - l1^2*(l2+l3) + l1^2*(l2+l3) + l1*l2*l3 - det *)
  (* = l1*l2*l3 - det = 0 *)
  nra.
Qed.
