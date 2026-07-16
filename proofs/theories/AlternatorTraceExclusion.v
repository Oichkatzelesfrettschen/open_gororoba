(** AlternatorTraceExclusion.v

    The "easy half" of Biss-Christensen-Dugger-Isaksen Question 9.2, as a
    fully axiom-free, kernel-checked arithmetic theorem.

    Setup (proved elsewhere in the eigentheory literature; here taken as the
    HYPOTHESES of the theorem, NOT as axioms of this file):
      - M_a := L_{a*} L_a / |a|^2 is symmetric, diagonalizable, PSD, with
        integer trace tr(M_a) = 2^n  (n = doubling stage, dim = 2^n);
      - every eigenspace of M_a has real dimension divisible by 4;
      - Eig_1(a) = ker of the alternator Delta_a, so
        dim Eig_1(a) = 2^n - rank(Delta_a).

    We encode an eigenvalue multiset abstractly by: the dimension N = 2^n,
    the multiplicity m1 of the eigenvalue 1, and the fact that the remaining
    (N - m1) dimensions carry eigenvalues != 1 whose multiplicities are all
    divisible by 4 and whose weighted sum closes the trace to N.

    CLAIM (rank Delta_a != 4, i.e. dim Eig_1 != N - 4):
      if the complement N - m1 = 4 is a single eigenspace (forced, since 4 is
      the minimal positive multiple of 4), its eigenvalue lambda satisfies the
      trace identity m1*1 + 4*lambda = N with m1 = N-4, forcing lambda = 1 --
      contradiction with lambda != 1.

    We prove the integer core: there is no integer lambda-numerator making the
    trace close when the complement has total dimension 4 and eigenvalue != 1.
    The eigenvalues of M_a are rational with denominator dividing |a|^2; scaling
    the trace identity by |a|^2 =: q gives an exact integer equation, which is
    what we formalize. *)

From Stdlib Require Import ZArith Lia.
Open Scope Z_scope.

(** Trace identity, cleared of denominators.
    q = |a|^2 (positive). Eigenvalues of q*M_a are integers; the eigenvalue 1
    scales to q. Total trace of q*M_a is q*N. If the 1-eigenspace has
    multiplicity m1 and the complementary 4 dimensions form one eigenspace with
    integer eigenvalue-numerator L (representing eigenvalue L/q), then:
        m1 * q + 4 * L = q * N. *)

Theorem alternator_no_codim4 :
  forall (N m1 q L : Z),
    q > 0 ->                    (* |a|^2 > 0 *)
    m1 = N - 4 ->              (* complement has dimension 4 *)
    m1 * q + 4 * L = q * N ->  (* trace identity (denominators cleared) *)
    L = q.                      (* forces eigenvalue L/q = 1 *)
Proof.
  intros N m1 q L Hq Hm1 Htr.
  subst m1.
  (* (N-4)*q + 4*L = q*N  ==>  4*L = 4*q  ==>  L = q *)
  nia.
Qed.

(** Corollary in words: the complementary eigenvalue equals 1, contradicting
    the requirement that a complementary eigenspace carry eigenvalue != 1.
    Hence dim Eig_1(a) = N - 4 is impossible. The theorem below packages the
    contradiction explicitly. *)

Theorem dim_eig1_neq_codim4 :
  forall (N m1 q L : Z),
    q > 0 ->
    m1 = N - 4 ->
    m1 * q + 4 * L = q * N ->
    L <> q ->        (* eigenvalue of the complement is NOT 1 *)
    False.
Proof.
  intros N m1 q L Hq Hm1 Htr Hne.
  apply Hne. eapply alternator_no_codim4; eauto.
Qed.
