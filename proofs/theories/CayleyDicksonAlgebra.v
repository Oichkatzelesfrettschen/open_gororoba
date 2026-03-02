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

(** * Extended arithmetic operations. *)

(** Complex addition. *)
Definition complex_add (z w : CDComplex) : CDComplex :=
  mkComplex (cre z + cre w) (cim z + cim w).

(** Complex scalar multiplication. *)
Definition complex_scale (r : R) (z : CDComplex) : CDComplex :=
  mkComplex (r * cre z) (r * cim z).

(** Complex zero. *)
Definition complex_zero : CDComplex := mkComplex 0 0.

(** Complex one. *)
Definition complex_one : CDComplex := mkComplex 1 0.

(** Quaternion addition. *)
Definition quat_add (p q : CDQuat) : CDQuat :=
  mkQuat (qa p + qa q) (qb p + qb q) (qc p + qc q) (qd p + qd q).

(** Quaternion scalar multiplication. *)
Definition quat_scale (r : R) (q : CDQuat) : CDQuat :=
  mkQuat (r * qa q) (r * qb q) (r * qc q) (r * qd q).

(** Quaternion zero. *)
Definition quat_zero : CDQuat := mkQuat 0 0 0 0.

(** Quaternion one. *)
Definition quat_one : CDQuat := mkQuat 1 0 0 0.

(** Quaternion negation. *)
Definition quat_neg (q : CDQuat) : CDQuat :=
  mkQuat (- qa q) (- qb q) (- qc q) (- qd q).

(** * Extended theorems. *)

(** Complex multiplication is commutative. *)
Theorem complex_mul_comm : forall z w, complex_mul z w = complex_mul w z.
Proof. intros; destruct z, w; unfold complex_mul; simpl; f_equal; ring. Qed.

(** Complex multiplication is associative. *)
Theorem complex_mul_assoc : forall x y z,
  complex_mul (complex_mul x y) z = complex_mul x (complex_mul y z).
Proof. intros; destruct x, y, z; unfold complex_mul; simpl; f_equal; ring. Qed.

(*<*conjanti>*)
(** Complex conjugate is an anti-automorphism: conj(z*w) = conj(w)*conj(z). *)
Theorem complex_conj_antimorphism : forall z w,
  complex_conj (complex_mul z w) = complex_mul (complex_conj w) (complex_conj z).
Proof. intros; destruct z, w; unfold complex_conj, complex_mul; simpl; f_equal; ring. Qed.
(*</conjanti>*)

(** Complex norm is multiplicative: |z*w|^2 = |z|^2 * |w|^2. *)
Theorem complex_norm_mul : forall z w,
  complex_norm_sq (complex_mul z w) = complex_norm_sq z * complex_norm_sq w.
Proof. intros; destruct z, w; unfold complex_norm_sq, complex_mul; simpl; ring. Qed.

(** Quaternion conjugate is an anti-automorphism: conj(p*q) = conj(q)*conj(p). *)
Theorem quat_conj_antimorphism : forall p q,
  quat_conj (quat_mul p q) = quat_mul (quat_conj q) (quat_conj p).
Proof. intros; destruct p, q; unfold quat_conj, quat_mul; simpl; f_equal; ring. Qed.

(** Quaternion norm is multiplicative: |p*q|^2 = |p|^2 * |q|^2 (Hurwitz). *)
Theorem quat_norm_mul : forall p q,
  quat_norm_sq (quat_mul p q) = quat_norm_sq p * quat_norm_sq q.
Proof. intros; destruct p, q; unfold quat_norm_sq, quat_mul; simpl; ring. Qed.

(** Quaternion multiplication is associative. *)
Theorem quat_mul_assoc : forall x y z,
  quat_mul (quat_mul x y) z = quat_mul x (quat_mul y z).
Proof. intros; destruct x, y, z; unfold quat_mul; simpl; f_equal; ring. Qed.

(** Real-part commutativity: Re(p*q) = Re(q*p). *)
Theorem quat_re_mul_comm : forall p q,
  qa (quat_mul p q) = qa (quat_mul q p).
Proof. intros; destruct p, q; unfold quat_mul; simpl; ring. Qed.

(** Left norm-conjugate: conj(q)*q = (|q|^2, 0, 0, 0). *)
Theorem quat_conj_norm_left : forall q,
  quat_mul (quat_conj q) q = mkQuat (quat_norm_sq q) 0 0 0.
Proof. intros; destruct q; unfold quat_mul, quat_conj, quat_norm_sq; simpl; f_equal; ring. Qed.

(*<*quatquad>*)
(** Quadratic identity: q^2 = 2*Re(q)*q - |q|^2 * 1.
    Every quaternion satisfies a quadratic over R. *)
Theorem quat_quadratic_identity : forall q,
  quat_mul q q = quat_add (quat_scale (2 * qa q) q)
                           (quat_scale (- quat_norm_sq q) quat_one).
Proof. intros; destruct q; unfold quat_mul, quat_add, quat_scale, quat_one, quat_norm_sq; simpl; f_equal; ring. Qed.
(*</quatquad>*)

(** Pure imaginary square: if Re(q)=0 then q^2 = -|q|^2 * 1. *)
Theorem quat_imaginary_square : forall q,
  qa q = 0 ->
  quat_mul q q = quat_scale (- quat_norm_sq q) quat_one.
Proof.
  intros q Hre; destruct q; unfold quat_mul, quat_scale, quat_one, quat_norm_sq; simpl in *;
  subst; f_equal; ring.
Qed.

(** Real part of square: Re(q^2) = 2*Re(q)^2 - |q|^2. *)
Theorem quat_re_square : forall q,
  qa (quat_mul q q) = 2 * (qa q)^2 - quat_norm_sq q.
Proof. intros; destruct q; unfold quat_mul, quat_norm_sq; simpl; ring. Qed.
