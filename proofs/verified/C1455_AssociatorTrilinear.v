(** * C-1455: Sedenion Associator Trilinearity (Proposition 3).

    The associator [a,b,c] := (a*b)*c - a*(b*c) is trilinear over R:

    1. Scalar homogeneity: [r*a, b, c] = r * [a, b, c]
    2. Additivity:         [a+a', b, c] = [a,b,c] + [a',b,c]

    (and similarly for the second and third arguments).

    Strategy: Destruct all sedenions to their 16 R-components,
    unfold all CD definitions, then close with f_equal + ring.

    Mirrors: crates/algebra_experimental/src/lepton_mass_hierarchy.rs *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion
                                 OctonionNorm CDAssociator SedenionAssociator.
Open Scope R_scope.

(** * Quaternion-level trilinearity (dim=4).

    The quaternion associator is identically zero, so trilinearity
    is trivially satisfied. Stated for the record. *)

Theorem quat_assoc_trilinear_scale_1 : forall r a b c,
  quat_assoc (quat_scale r a) b c = quat_scale r (quat_assoc a b c).
Proof.
  intros. destruct a, b, c.
  unfold quat_assoc, quat_scale, quat_mul, quat_add, quat_neg.
  simpl. f_equal; ring.
Qed.

Theorem quat_assoc_trilinear_add_1 : forall a a' b c,
  quat_assoc (quat_add a a') b c =
  quat_add (quat_assoc a b c) (quat_assoc a' b c).
Proof.
  intros. destruct a, a', b, c.
  unfold quat_assoc, quat_add, quat_mul, quat_neg.
  simpl. f_equal; ring.
Qed.

(** Helper tactic: destruct a sedenion into its 16 real components. *)
Ltac dest_sed s :=
  let lo := fresh "lo" in
  let hi := fresh "hi" in
  destruct s as [lo hi];
  let llo := fresh "llo" in
  let lhi := fresh "lhi" in
  destruct lo as [llo lhi];
  let hlo := fresh "hlo" in
  let hhi := fresh "hhi" in
  destruct hi as [hlo hhi];
  destruct llo; destruct lhi; destruct hlo; destruct hhi.

(** Helper tactic: unfold all CD layers and close with ring. *)
Ltac sed_ring :=
  unfold sed_assoc, sed_scale, sed_add, sed_mul, sed_sub, sed_neg;
  unfold oct_scale, oct_mul, oct_add, oct_neg, oct_sub, oct_conj;
  unfold quat_scale, quat_mul, quat_add, quat_neg, quat_conj;
  unfold sed_lo, sed_hi, oct_lo, oct_hi, qa, qb, qc, qd;
  f_equal; f_equal; f_equal; ring.

(** * Sedenion trilinearity: scalar homogeneity in first argument.
    [r*a, b, c] = r * [a, b, c] *)

Theorem sed_assoc_trilinear_scale_1 : forall r a b c,
  sed_assoc (sed_scale r a) b c = sed_scale r (sed_assoc a b c).
Proof.
  intros r a b c.
  dest_sed a; dest_sed b; dest_sed c.
  sed_ring.
Qed.

(** * Sedenion trilinearity: additivity in first argument.
    [a + a', b, c] = [a, b, c] + [a', b, c] *)

Theorem sed_assoc_trilinear_add_1 : forall a a' b c,
  sed_assoc (sed_add a a') b c =
  sed_add (sed_assoc a b c) (sed_assoc a' b c).
Proof.
  intros a a' b c.
  dest_sed a; dest_sed a'; dest_sed b; dest_sed c.
  sed_ring.
Qed.

(** * Sedenion trilinearity: scalar homogeneity in second argument.
    [a, r*b, c] = r * [a, b, c] *)

Theorem sed_assoc_trilinear_scale_2 : forall r a b c,
  sed_assoc a (sed_scale r b) c = sed_scale r (sed_assoc a b c).
Proof.
  intros r a b c.
  dest_sed a; dest_sed b; dest_sed c.
  sed_ring.
Qed.

(** * Sedenion trilinearity: scalar homogeneity in third argument.
    [a, b, r*c] = r * [a, b, c] *)

Theorem sed_assoc_trilinear_scale_3 : forall r a b c,
  sed_assoc a b (sed_scale r c) = sed_scale r (sed_assoc a b c).
Proof.
  intros r a b c.
  dest_sed a; dest_sed b; dest_sed c.
  sed_ring.
Qed.

(** * Sedenion trilinearity: additivity in second argument. *)

Theorem sed_assoc_trilinear_add_2 : forall a b b' c,
  sed_assoc a (sed_add b b') c =
  sed_add (sed_assoc a b c) (sed_assoc a b' c).
Proof.
  intros a b b' c.
  dest_sed a; dest_sed b; dest_sed b'; dest_sed c.
  sed_ring.
Qed.

(** * Sedenion trilinearity: additivity in third argument. *)

Theorem sed_assoc_trilinear_add_3 : forall a b c c',
  sed_assoc a b (sed_add c c') =
  sed_add (sed_assoc a b c) (sed_assoc a b c').
Proof.
  intros a b c c'.
  dest_sed a; dest_sed b; dest_sed c; dest_sed c'.
  sed_ring.
Qed.
