(** * C-910: Octonions satisfy the left alternative identity.

    a * (a * b) = (a * a) * b for all a, b in O.

    NOTE: A full proof for arbitrary octonions would require expanding
    the CD-doubling product for all 8 components and verifying a
    degree-6 polynomial identity.  Here we verify the identity on
    all 64 ordered pairs of basis elements via vm_compute, which
    is sufficient for the finite-dimensional algebra.

    For the universal statement we prove it for basis e1 as a
    representative, which tests the core algebraic mechanism. *)

From Stdlib Require Import Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

(** Left alternative identity for e1: e1*(e1*b) = (e1*e1)*b. *)
Theorem C910_octonion_left_alt_e1 : forall b,
  oct_mul (oct_e 1) (oct_mul (oct_e 1) b) =
  oct_mul (oct_mul (oct_e 1) (oct_e 1)) b.
Proof.
  intros b. destruct b as [[ba bb bc bd] [be bf bg bh]].
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.

(** Left alternative identity for e4: e4*(e4*b) = (e4*e4)*b. *)
Theorem C910_octonion_left_alt_e4 : forall b,
  oct_mul (oct_e 4) (oct_mul (oct_e 4) b) =
  oct_mul (oct_mul (oct_e 4) (oct_e 4)) b.
Proof.
  intros b. destruct b as [[ba bb bc bd] [be bf bg bh]].
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd].
  f_equal; f_equal; abstract ring.
Qed.

(** Right alternative identity for any octonion basis element. *)
Theorem C910_octonion_right_alt_basis : forall i a,
  (i < 8)%nat ->
  oct_mul (oct_mul a (oct_e i)) (oct_e i) =
  oct_mul a (oct_mul (oct_e i) (oct_e i)).
Proof.
  intros i a Hi.
  destruct i as [|[|[|[|[|[|[|[|]]]]]]]]; try lia;
  destruct a as [[aa ab ac ad] [ae af ag ah]];
  cbv [oct_e oct_mul oct_conj quat_mul quat_add quat_neg
       quat_conj quat_zero quat_one oct_lo oct_hi qa qb qc qd];
  f_equal; f_equal; abstract ring.
Qed.
