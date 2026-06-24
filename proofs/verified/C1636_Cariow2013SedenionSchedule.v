(** * C-1636: Cariow 2013 sedenion schedule.

    Cariow's 16D sedenion multiplier factors the row-flipped product as a
    block-symmetric Hadamard-diagonal term plus a sparse correction:

      Y = Bcheck X - 2 * Bhat X.

    The schedule below formalizes the scalar computation over the same
    coordinate order as OpenGororoba::Sedenion and proves that it is
    extensionally equal to the ordinary Cayley-Dickson product.  The count
    theorem records the hardware multiplier boundary: 16 diagonal products plus
    106 sparse correction products, excluding powers-of-two scaling. *)

From Stdlib Require Import Reals Lra Lia.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion.

Open Scope R_scope.

Record R16 := mkR16 {
  r16_0 : R; r16_1 : R; r16_2 : R; r16_3 : R;
  r16_4 : R; r16_5 : R; r16_6 : R; r16_7 : R;
  r16_8 : R; r16_9 : R; r16_10 : R; r16_11 : R;
  r16_12 : R; r16_13 : R; r16_14 : R; r16_15 : R
}.

Definition r16_to_sed (x : R16) : CDSed :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkSed
        (mkOct (mkQuat x0 x1 x2 x3) (mkQuat x4 x5 x6 x7))
        (mkOct (mkQuat x8 x9 x10 x11) (mkQuat x12 x13 x14 x15))
  end.

Definition r16_hadamard_stage0 (x : R16) : R16 :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkR16
        (x0 + x8) (x1 + x9) (x2 + x10) (x3 + x11)
        (x4 + x12) (x5 + x13) (x6 + x14) (x7 + x15)
        (x0 - x8) (x1 - x9) (x2 - x10) (x3 - x11)
        (x4 - x12) (x5 - x13) (x6 - x14) (x7 - x15)
  end.

Definition r16_hadamard_stage1 (x : R16) : R16 :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkR16
        (x0 + x4) (x1 + x5) (x2 + x6) (x3 + x7)
        (x0 - x4) (x1 - x5) (x2 - x6) (x3 - x7)
        (x8 + x12) (x9 + x13) (x10 + x14) (x11 + x15)
        (x8 - x12) (x9 - x13) (x10 - x14) (x11 - x15)
  end.

Definition r16_hadamard_stage2 (x : R16) : R16 :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkR16
        (x0 + x2) (x1 + x3) (x0 - x2) (x1 - x3)
        (x4 + x6) (x5 + x7) (x4 - x6) (x5 - x7)
        (x8 + x10) (x9 + x11) (x8 - x10) (x9 - x11)
        (x12 + x14) (x13 + x15) (x12 - x14) (x13 - x15)
  end.

Definition r16_hadamard_stage3 (x : R16) : R16 :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkR16
        (x0 + x1) (x0 - x1) (x2 + x3) (x2 - x3)
        (x4 + x5) (x4 - x5) (x6 + x7) (x6 - x7)
        (x8 + x9) (x8 - x9) (x10 + x11) (x10 - x11)
        (x12 + x13) (x12 - x13) (x14 + x15) (x14 - x15)
  end.

Definition r16_hadamard_forward (x : R16) : R16 :=
  r16_hadamard_stage3
    (r16_hadamard_stage2
      (r16_hadamard_stage1
        (r16_hadamard_stage0 x))).

Definition r16_hadamard_reverse (x : R16) : R16 :=
  r16_hadamard_stage0
    (r16_hadamard_stage1
      (r16_hadamard_stage2
        (r16_hadamard_stage3 x))).

Definition r16_scale (k : R) (x : R16) : R16 :=
  match x with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 =>
      mkR16
        (k * x0) (k * x1) (k * x2) (k * x3)
        (k * x4) (k * x5) (k * x6) (k * x7)
        (k * x8) (k * x9) (k * x10) (k * x11)
        (k * x12) (k * x13) (k * x14) (k * x15)
  end.

Definition r16_pointwise_mul (x y : R16) : R16 :=
  match x, y with
  | mkR16 x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15,
    mkR16 y0 y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 =>
      mkR16
        (x0 * y0) (x1 * y1) (x2 * y2) (x3 * y3)
        (x4 * y4) (x5 * y5) (x6 * y6) (x7 * y7)
        (x8 * y8) (x9 * y9) (x10 * y10) (x11 * y11)
        (x12 * y12) (x13 * y13) (x14 * y14) (x15 * y15)
  end.

Definition r16_sparse_correction (a b : R16) : R16 :=
  match a, b with
  | mkR16 a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15,
    mkR16 b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 =>
      mkR16
        (2 * (b0 * a0))
        (2 * (b2 * a3 + b4 * a5 + b7 * a6 + b8 * a9 +
              b11 * a10 + b13 * a12 + b14 * a15))
        (2 * (b3 * a1 + b4 * a6 + b5 * a7 + b8 * a10 +
              b9 * a11 + b14 * a12 + b15 * a13))
        (2 * (b1 * a2 + b6 * a5 + b4 * a7 + b10 * a9 +
              b8 * a11 + b15 * a12 + b13 * a14))
        (2 * (b5 * a1 + b6 * a2 + b7 * a3 + b8 * a12 +
              b9 * a13 + b10 * a14 + b11 * a15))
        (2 * (b7 * a2 + b1 * a4 + b3 * a6 + b12 * a9 +
              b14 * a11 + b8 * a13 + b10 * a15))
        (2 * (b5 * a3 + b2 * a4 + b1 * a7 + b15 * a9 +
              b12 * a10 + b11 * a13 + b8 * a14))
        (2 * (b6 * a1 + b3 * a4 + b2 * a5 + b13 * a10 +
              b12 * a11 + b9 * a14 + b8 * a15))
        (2 * (b9 * a1 + b10 * a2 + b11 * a3 + b12 * a4 +
              b13 * a5 + b14 * a6 + b15 * a7))
        (2 * (b11 * a2 + b13 * a4 + b14 * a7 + b1 * a8 +
              b3 * a10 + b5 * a12 + b6 * a15))
        (2 * (b9 * a3 + b14 * a4 + b15 * a5 + b2 * a8 +
              b1 * a11 + b6 * a12 + b7 * a13))
        (2 * (b10 * a1 + b15 * a4 + b13 * a6 + b3 * a8 +
              b2 * a9 + b7 * a12 + b5 * a14))
        (2 * (b9 * a5 + b10 * a6 + b11 * a7 + b4 * a8 +
              b1 * a13 + b2 * a14 + b3 * a15))
        (2 * (b12 * a1 + b14 * a3 + b10 * a7 + b5 * a8 +
              b4 * a9 + b6 * a11 + b2 * a15))
        (2 * (b15 * a1 + b12 * a2 + b11 * a5 + b6 * a8 +
              b7 * a9 + b4 * a10 + b3 * a13))
        (2 * (b13 * a2 + b12 * a3 + b9 * a6 + b7 * a8 +
              b5 * a10 + b4 * a11 + b1 * a14))
  end.

Definition r16_cariow2013_toeplitz_part (a b : R16) : R16 :=
  r16_hadamard_reverse
    (r16_pointwise_mul
      (r16_hadamard_forward a)
      (r16_scale (/ 16) (r16_hadamard_forward b))).

Definition r16_cariow2013_sedenion_mul (a b : R16) : R16 :=
  let t := r16_cariow2013_toeplitz_part a b in
  let s := r16_sparse_correction a b in
  match t, s with
  | mkR16 t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15,
    mkR16 s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 =>
      mkR16
        (-(t0 - s0))
        (t1 - s1) (t2 - s2) (t3 - s3) (t4 - s4) (t5 - s5)
        (t6 - s6) (t7 - s7) (t8 - s8) (t9 - s9) (t10 - s10)
        (t11 - s11) (t12 - s12) (t13 - s13) (t14 - s14) (t15 - s15)
  end.

Theorem C1636_cariow2013_multiplier_count :
  (16 + 106 = 122 /\ 122 < 256)%nat.
Proof.
  split; lia.
Qed.

Ltac close_r16_cariow_component :=
  cbv [r16_to_sed r16_cariow2013_sedenion_mul
       r16_cariow2013_toeplitz_part r16_sparse_correction
       r16_pointwise_mul r16_scale r16_hadamard_forward
       r16_hadamard_reverse r16_hadamard_stage0 r16_hadamard_stage1
       r16_hadamard_stage2 r16_hadamard_stage3 sed_mul oct_mul
       oct_conj quat_conj quat_neg quat_add quat_mul sed_lo sed_hi
       oct_lo oct_hi qa qb qc qd];
  apply (f_equal2 mkSed);
  apply (f_equal2 mkOct);
  apply (f_equal4 mkQuat);
  field; lra.

Theorem C1636_cariow2013_sedenion_mul_eq :
  forall a b : R16,
    r16_to_sed (r16_cariow2013_sedenion_mul a b) =
    sed_mul (r16_to_sed a) (r16_to_sed b).
Proof.
  intros [a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15]
         [b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15].
  close_r16_cariow_component.
Qed.

Record Cariow2013SedenionScheduleSurface := {
  c1636_multiplier_count :
    (16 + 106 = 122 /\ 122 < 256)%nat;
  c1636_schedule_eq :
    forall a b : R16,
      r16_to_sed (r16_cariow2013_sedenion_mul a b) =
      sed_mul (r16_to_sed a) (r16_to_sed b)
}.

Definition C1636_cariow2013_sedenion_schedule_surface :
  Cariow2013SedenionScheduleSurface :=
  {| c1636_multiplier_count := C1636_cariow2013_multiplier_count;
     c1636_schedule_eq := C1636_cariow2013_sedenion_mul_eq |}.
