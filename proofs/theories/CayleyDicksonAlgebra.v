(** * CayleyDicksonAlgebra: Cayley-Dickson construction at dims 2 and 4.

    Defines complex and quaternion types via the Cayley-Dickson doubling
    construction and proves two fundamental algebraic identities:
    - Conjugate involution: conj(conj(z)) = z
    - Norm-conjugate identity: |z|^2 = z * conj(z)  (Brahmagupta-Fibonacci)

    These hold at EVERY Cayley-Dickson level (R, C, H, O, S, ...),
    though we formalize only dims 2 and 4 for extraction efficiency.

    Mirrors: algebra_core/src/construction/cayley_dickson.rs *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** Complex number as a Cayley-Dickson pair (a, b). *)
Record CDComplex := mkComplex {
  cre : R;
  cim : R;
}.

(** Quaternion as a Cayley-Dickson pair of pairs ((a,b),(c,d)). *)
Record CDQuat := mkQuat {
  qa : R;
  qb : R;
  qc : R;
  qd : R;
}.

(** Complex conjugate: (a, b) -> (a, -b). *)
Definition complex_conj (z : CDComplex) : CDComplex :=
  mkComplex (cre z) (- cim z).

(** Complex norm squared: a^2 + b^2. *)
Definition complex_norm_sq (z : CDComplex) : R :=
  (cre z) ^ 2 + (cim z) ^ 2.

(** Complex multiplication: (a,b)*(c,d) = (ac - db*, a*d + cb).
    Standard Cayley-Dickson formula at level 1. *)
Definition complex_mul (z w : CDComplex) : CDComplex :=
  mkComplex
    (cre z * cre w - cim z * cim w)
    (cre z * cim w + cim z * cre w).

(** Quaternion conjugate: (a, -b, -c, -d). *)
Definition quat_conj (q : CDQuat) : CDQuat :=
  mkQuat (qa q) (- qb q) (- qc q) (- qd q).

(** Quaternion norm squared: a^2 + b^2 + c^2 + d^2. *)
Definition quat_norm_sq (q : CDQuat) : R :=
  (qa q) ^ 2 + (qb q) ^ 2 + (qc q) ^ 2 + (qd q) ^ 2.

(** Quaternion multiplication (Hamilton product). *)
Definition quat_mul (p q : CDQuat) : CDQuat :=
  mkQuat
    (qa p * qa q - qb p * qb q - qc p * qc q - qd p * qd q)
    (qa p * qb q + qb p * qa q + qc p * qd q - qd p * qc q)
    (qa p * qc q - qb p * qd q + qc p * qa q + qd p * qb q)
    (qa p * qd q + qb p * qc q - qc p * qb q + qd p * qa q).

(** Conjugate involution for complex numbers. *)
Theorem complex_conj_involution : forall z,
  complex_conj (complex_conj z) = z.
Proof. intros. destruct z. unfold complex_conj. simpl. f_equal; ring. Qed.

(** Conjugate involution for quaternions. *)
Theorem quat_conj_involution : forall q,
  quat_conj (quat_conj q) = q.
Proof. intros. destruct q. unfold quat_conj. simpl. f_equal; ring. Qed.

(** Norm-conjugate identity for complex numbers:
    z * conj(z) = (|z|^2, 0). *)
Theorem complex_norm_conjugate : forall z,
  complex_mul z (complex_conj z) = mkComplex (complex_norm_sq z) 0.
Proof.
  intros. destruct z. unfold complex_mul, complex_conj, complex_norm_sq. simpl.
  f_equal; ring.
Qed.

(*<*normconj>*)
(** Norm-conjugate identity for quaternions:
    q * conj(q) = (|q|^2, 0, 0, 0). *)
Theorem quat_norm_conjugate : forall q,
  quat_mul q (quat_conj q) = mkQuat (quat_norm_sq q) 0 0 0.
Proof.
  intros. destruct q. unfold quat_mul, quat_conj, quat_norm_sq. simpl.
  f_equal; ring.
Qed.
(*</normconj>*)
