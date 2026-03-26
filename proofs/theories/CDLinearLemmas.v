(** * CDLinearLemmas: Linearity of Cayley-Dickson multiplication.

    Proves that multiplication distributes over addition and commutes
    with scalar multiplication at each CD level.  These lemmas are the
    foundation for proving associator trilinearity WITHOUT triggering
    the monolithic `ring` explosion at dim >= 8.

    Tower strategy:
      quat_mul_scale_left  ->  oct_mul_scale_left  ->  sed_mul_scale_left
      quat_mul_add_left    ->  oct_mul_add_left    ->  sed_mul_add_left

    Each level lifts to the next via the CD doubling formula:
      (a,b)*(c,d) = (ac - conj(d)*b, da + b*conj(c))

    Mirrors: crates/cd_kernel/src/cayley_dickson.rs *)

From Stdlib Require Import Reals Lra.
From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.
Open Scope R_scope.

(** * Quaternion level: direct destruct + ring. *)

Theorem quat_mul_scale_left : forall r a b,
  quat_mul (quat_scale r a) b = quat_scale r (quat_mul a b).
Proof.
  intros r a b; destruct a, b.
  unfold quat_mul, quat_scale; simpl.
  f_equal; ring.
Qed.

Theorem quat_mul_scale_right : forall r a b,
  quat_mul a (quat_scale r b) = quat_scale r (quat_mul a b).
Proof.
  intros r a b; destruct a, b.
  unfold quat_mul, quat_scale; simpl.
  f_equal; ring.
Qed.

Theorem quat_mul_add_left : forall a a' b,
  quat_mul (quat_add a a') b = quat_add (quat_mul a b) (quat_mul a' b).
Proof.
  intros a a' b; destruct a, a', b.
  unfold quat_mul, quat_add; simpl.
  f_equal; ring.
Qed.

Theorem quat_mul_add_right : forall a b b',
  quat_mul a (quat_add b b') = quat_add (quat_mul a b) (quat_mul a b').
Proof.
  intros a b b'; destruct a, b, b'.
  unfold quat_mul, quat_add; simpl.
  f_equal; ring.
Qed.

Theorem quat_conj_scale : forall r a,
  quat_conj (quat_scale r a) = quat_scale r (quat_conj a).
Proof.
  intros r a; destruct a.
  unfold quat_conj, quat_scale; simpl.
  f_equal; ring.
Qed.

Theorem quat_neg_scale : forall r a,
  quat_neg (quat_scale r a) = quat_scale r (quat_neg a).
Proof.
  intros r a; destruct a.
  unfold quat_neg, quat_scale; simpl.
  f_equal; ring.
Qed.

Theorem quat_conj_add : forall a b,
  quat_conj (quat_add a b) = quat_add (quat_conj a) (quat_conj b).
Proof.
  intros a b; destruct a, b.
  unfold quat_conj, quat_add; simpl.
  f_equal; ring.
Qed.

(** * Octonion level: lift from quaternion via CD doubling.

    oct_mul(a,b)(c,d) = (quat_mul a c - quat_mul (quat_conj d) b,
                         quat_mul d a + quat_mul b (quat_conj c))

    oct_scale r (a,b) = (quat_scale r a, quat_scale r b) *)

Theorem oct_mul_scale_left : forall r x y,
  oct_mul (oct_scale r x) y = oct_scale r (oct_mul x y).
Proof.
  intros r x y.
  destruct x as [xa xb], y as [ya yb].
  unfold oct_mul, oct_scale.
  simpl oct_lo; simpl oct_hi.
  f_equal.
  - (* lo component: quat_mul (quat_scale r xa) ya -
       quat_mul (quat_conj yb) (quat_scale r xb)
       = quat_scale r (quat_mul xa ya - quat_mul (quat_conj yb) xb) *)
    rewrite quat_mul_scale_left.
    rewrite quat_mul_scale_right.
    destruct xa, xb, ya, yb.
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl.
    f_equal; ring.
  - (* hi component *)
    rewrite quat_mul_scale_right.
    rewrite quat_mul_scale_left.
    destruct xa, xb, ya, yb.
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl.
    f_equal; ring.
Qed.

Theorem oct_mul_scale_right : forall r x y,
  oct_mul x (oct_scale r y) = oct_scale r (oct_mul x y).
Proof.
  intros r x y.
  destruct x as [xa xb], y as [ya yb].
  unfold oct_mul, oct_scale.
  simpl oct_lo; simpl oct_hi.
  f_equal.
  - rewrite quat_mul_scale_right.
    rewrite quat_conj_scale.
    rewrite quat_mul_scale_left.
    destruct xa, xb, ya, yb.
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl.
    f_equal; ring.
  - rewrite quat_mul_scale_left.
    rewrite quat_conj_scale.
    rewrite quat_mul_scale_right.
    destruct xa, xb, ya, yb.
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl.
    f_equal; ring.
Qed.

Theorem oct_conj_scale : forall r x,
  oct_conj (oct_scale r x) = oct_scale r (oct_conj x).
Proof.
  intros r [lo hi].
  unfold oct_conj, oct_scale, oct_neg.
  simpl.
  f_equal.
  - apply quat_conj_scale.
  - apply quat_neg_scale.
Qed.

Theorem oct_conj_add : forall x y,
  oct_conj (oct_add x y) = oct_add (oct_conj x) (oct_conj y).
Proof.
  intros [xlo xhi] [ylo yhi].
  unfold oct_conj, oct_add, oct_neg.
  simpl.
  f_equal.
  - apply quat_conj_add.
  - destruct xhi, yhi.
    unfold quat_add, quat_neg.
    simpl.
    f_equal; ring.
Qed.

Theorem oct_mul_add_left : forall x x' y,
  oct_mul (oct_add x x') y = oct_add (oct_mul x y) (oct_mul x' y).
Proof.
  intros x x' y.
  destruct x as [xa xb], x' as [xa' xb'], y as [ya yb].
  unfold oct_mul, oct_add.
  simpl oct_lo; simpl oct_hi.
  f_equal.
  - rewrite quat_mul_add_left.
    rewrite quat_mul_add_right.
    destruct xa, xb, xa', xb', ya, yb.
    unfold quat_mul, quat_add, quat_neg, quat_conj.
    simpl.
    f_equal; ring.
  - rewrite quat_mul_add_right.
    rewrite quat_mul_add_left.
    destruct xa, xb, xa', xb', ya, yb.
    unfold quat_mul, quat_add, quat_neg, quat_conj.
    simpl.
    f_equal; ring.
Qed.

Theorem oct_mul_add_right : forall x y y',
  oct_mul x (oct_add y y') = oct_add (oct_mul x y) (oct_mul x y').
Proof.
  intros x y y'.
  destruct x as [xa xb], y as [ya yb], y' as [ya' yb'].
  unfold oct_mul, oct_add.
  simpl oct_lo; simpl oct_hi.
  f_equal.
  - rewrite quat_mul_add_right.
    rewrite quat_conj_add.
    rewrite quat_mul_add_left.
    destruct xa, xb, ya, yb, ya', yb'.
    unfold quat_mul, quat_add, quat_neg, quat_conj.
    simpl.
    f_equal; ring.
  - rewrite quat_mul_add_left.
    rewrite quat_conj_add.
    rewrite quat_mul_add_right.
    destruct xa, xb, ya, yb, ya', yb'.
    unfold quat_mul, quat_add, quat_neg, quat_conj.
    simpl.
    f_equal; ring.
Qed.

(** * Sedenion level: lift from octonion via CD doubling.

    sed_mul(a,b)(c,d) = (oct_mul a c - oct_mul (oct_conj d) b,
                         oct_mul d a + oct_mul b (oct_conj c))

    sed_scale r (a,b) = (oct_scale r a, oct_scale r b) *)

Theorem sed_mul_scale_left : forall r x y,
  sed_mul (sed_scale r x) y = sed_scale r (sed_mul x y).
Proof.
  intros r x y.
  destruct x as [xa xb], y as [ya yb].
  unfold sed_mul, sed_scale.
  simpl sed_lo; simpl sed_hi.
  f_equal.
  - (* lo: oct_mul (oct_scale r xa) ya - oct_mul (oct_conj yb) (oct_scale r xb) *)
    rewrite oct_mul_scale_left.
    rewrite oct_mul_scale_right.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_scale, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
  - (* hi *)
    rewrite oct_mul_scale_right.
    rewrite oct_mul_scale_left.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_scale, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
Qed.

Theorem sed_mul_scale_right : forall r x y,
  sed_mul x (sed_scale r y) = sed_scale r (sed_mul x y).
Proof.
  intros r x y.
  destruct x as [xa xb], y as [ya yb].
  unfold sed_mul, sed_scale.
  simpl sed_lo; simpl sed_hi.
  f_equal.
  - rewrite oct_mul_scale_right.
    (* oct_conj distributes over oct_scale *)
    assert (Hcs: forall s v, oct_conj (oct_scale s v) = oct_scale s (oct_conj v)).
    { intros s v; destruct v as [vlo vhi].
      unfold oct_conj, oct_scale, oct_neg; simpl.
      f_equal. apply quat_conj_scale. apply quat_neg_scale. }
    rewrite Hcs.
    rewrite oct_mul_scale_left.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_scale, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
  - rewrite oct_mul_scale_left.
    assert (Hcs: forall s v, oct_conj (oct_scale s v) = oct_scale s (oct_conj v)).
    { intros s v; destruct v as [vlo vhi].
      unfold oct_conj, oct_scale, oct_neg; simpl.
      f_equal. apply quat_conj_scale. apply quat_neg_scale. }
    rewrite Hcs.
    rewrite oct_mul_scale_right.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_scale, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_scale, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
Qed.

Theorem sed_conj_scale : forall r x,
  sed_conj (sed_scale r x) = sed_scale r (sed_conj x).
Proof.
  intros r [lo hi].
  unfold sed_conj, sed_scale, sed_neg.
  simpl.
  f_equal.
  - apply oct_conj_scale.
  - destruct hi as [hilo hihi].
    unfold oct_scale, oct_neg.
    simpl.
    f_equal; apply quat_neg_scale.
Qed.

Theorem sed_conj_add : forall x y,
  sed_conj (sed_add x y) = sed_add (sed_conj x) (sed_conj y).
Proof.
  intros [xlo xhi] [ylo yhi].
  unfold sed_conj, sed_add, sed_neg.
  simpl.
  f_equal.
  - apply oct_conj_add.
  - destruct xhi as [xhilo xhihi], yhi as [yhilo yhihi].
    unfold oct_add, oct_neg.
    simpl.
    f_equal; destruct xhilo, xhihi, yhilo, yhihi;
    unfold quat_add, quat_neg; simpl; f_equal; ring.
Qed.

Theorem sed_mul_add_left : forall x x' y,
  sed_mul (sed_add x x') y = sed_add (sed_mul x y) (sed_mul x' y).
Proof.
  intros x x' y.
  destruct x as [xa xb], x' as [xa' xb'], y as [ya yb].
  unfold sed_mul, sed_add.
  simpl sed_lo; simpl sed_hi.
  f_equal.
  - rewrite oct_mul_add_left.
    rewrite oct_mul_add_right.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             xa' as [xa'lo xa'hi], xb' as [xb'lo xb'hi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, xa'lo, xa'hi, xb'lo, xb'hi,
                    yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
  - rewrite oct_mul_add_right.
    rewrite oct_mul_add_left.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             xa' as [xa'lo xa'hi], xb' as [xb'lo xb'hi],
             ya as [yalo yahi], yb as [yblo ybhi].
    unfold oct_mul, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, xa'lo, xa'hi, xb'lo, xb'hi,
                    yalo, yahi, yblo, ybhi;
    unfold quat_mul, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
Qed.

Theorem sed_mul_add_right : forall x y y',
  sed_mul x (sed_add y y') = sed_add (sed_mul x y) (sed_mul x y').
Proof.
  intros x y y'.
  destruct x as [xa xb], y as [ya yb], y' as [ya' yb'].
  unfold sed_mul, sed_add.
  simpl sed_lo; simpl sed_hi.
  f_equal.
  - rewrite oct_mul_add_right.
    rewrite oct_conj_add.
    rewrite oct_mul_add_left.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi],
             ya' as [ya'lo ya'hi], yb' as [yb'lo yb'hi].
    unfold oct_mul, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi,
                    ya'lo, ya'hi, yb'lo, yb'hi;
    unfold quat_mul, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
  - rewrite oct_mul_add_left.
    rewrite oct_conj_add.
    rewrite oct_mul_add_right.
    destruct xa as [xalo xahi], xb as [xblo xbhi],
             ya as [yalo yahi], yb as [yblo ybhi],
             ya' as [ya'lo ya'hi], yb' as [yb'lo yb'hi].
    unfold oct_mul, oct_add, oct_neg, oct_sub, oct_conj.
    simpl oct_lo; simpl oct_hi.
    f_equal; destruct xalo, xahi, xblo, xbhi, yalo, yahi, yblo, ybhi,
                    ya'lo, ya'hi, yb'lo, yb'hi;
    unfold quat_mul, quat_add, quat_neg, quat_conj; simpl;
    f_equal; ring.
Qed.
