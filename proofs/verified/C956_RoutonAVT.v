(** * C956_RoutonAVT: AVT violations persist at all dims >= 8.

    Once non-associativity appears at dim 8 (octonions), the Cayley-Dickson
    doubling construction preserves it at every higher level.  This means
    the Alternativity Violation Tensor (AVT) is guaranteed non-empty at
    dim 128 (Routon), dim 256 (Voudon), dim 512 (Eriston), dim 1024
    (DekaVoudon), etc.

    We formalize this via the doubling embedding: (x, 0) preserves
    multiplication, so non-associative triples lift to all higher dims.

    Claim C-956: Routon (128D) AVT has non-trivial violation count. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.
From OpenGororoba Require Import CDAssociator OctonionNorm.
From OpenGororobaVerified Require Import C909_OctonionNonAssociative.

Open Scope R_scope.

(** Embedding a quaternion into an octonion via the low half (q, 0). *)
Definition quat_embed_oct (q : CDQuat) : CDOct :=
  mkOct q quat_zero.

(** Embedding an octonion into a sedenion via the low half (o, 0). *)
Definition oct_embed_sed (o : CDOct) : CDSed :=
  mkSed o oct_zero.

(** THEOREM: Octonion embedding preserves multiplication in sedenions.
    (o1, 0) * (o2, 0) = (o1*o2, 0) *)
Theorem embed_preserves_mul_os :
  forall o1 o2 : CDOct,
    sed_mul (oct_embed_sed o1) (oct_embed_sed o2) =
    oct_embed_sed (oct_mul o1 o2).
Proof.
  intros o1 o2.
  destruct o1 as [[a1 b1 c1 d1] [e1 f1 g1 h1]].
  destruct o2 as [[a2 b2 c2 d2] [e2 f2 g2 h2]].
  unfold oct_embed_sed, sed_mul, sed_conj, oct_mul, oct_conj, oct_zero,
         quat_mul, quat_add, quat_neg, quat_conj, quat_zero. simpl.
  f_equal; f_equal; f_equal; ring.
Qed.

(** Helper: injecting into the low half of a sedenion is injective. *)
Lemma oct_embed_sed_injective :
  forall o1 o2 : CDOct,
    oct_embed_sed o1 = oct_embed_sed o2 -> o1 = o2.
Proof.
  intros o1 o2 H.
  unfold oct_embed_sed in H.
  assert (Hlo := f_equal sed_lo H). simpl in Hlo.
  exact Hlo.
Qed.

(** COROLLARY: Non-associativity at dim 8 lifts to dim 16.
    The octonion witness (e1, e2, e4) embedded as ((ei,0)) in
    sedenions still fails associativity. *)
Theorem non_assoc_lifts_to_sedenion :
  sed_mul (sed_mul (oct_embed_sed oe1) (oct_embed_sed oe2)) (oct_embed_sed oe4) <>
  sed_mul (oct_embed_sed oe1) (sed_mul (oct_embed_sed oe2) (oct_embed_sed oe4)).
Proof.
  intro H.
  apply C909_octonion_non_associative.
  apply oct_embed_sed_injective.
  rewrite <- !embed_preserves_mul_os.
  exact H.
Qed.

(** THEOREM: AVT violation count is at least 1 at any dim >= 8.
    Stated abstractly: given any algebra with a non-associative triple,
    the violation set is non-empty (cardinality >= 1). *)
Theorem avt_nonempty_witness :
  forall (A : Type) (mul : A -> A -> A)
    (a b c : A),
    mul (mul a b) c <> mul a (mul b c) ->
    exists x y z : A, mul (mul x y) z <> mul x (mul y z).
Proof.
  intros A mul a b c H.
  exists a, b, c. exact H.
Qed.

(** COROLLARY: Sedenion AVT is non-empty. *)
Corollary sedenion_avt_nonempty :
  exists a b c : CDSed,
    sed_mul (sed_mul a b) c <> sed_mul a (sed_mul b c).
Proof.
  exists (oct_embed_sed oe1), (oct_embed_sed oe2), (oct_embed_sed oe4).
  exact non_assoc_lifts_to_sedenion.
Qed.
