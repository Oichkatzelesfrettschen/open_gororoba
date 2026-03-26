(** * FloatQuaternion: quaternion operations parameterized over FLOAT_OPS.

    This functor implements quaternion multiplication, conjugation, and
    rotation for any field satisfying FLOAT_OPS. The key property --
    q*v*conj(q) = R(q)*v for unit quaternions -- is proved abstractly
    via the ring axioms.

    Mirrors: quat_rotation.rs lines 40-61 (rotate), 69-86 (matrix). *)

From Stdlib.setoid_ring Require Import Ring_theory Ring.
From OpenGororoba Require Import Prelude FloatAxioms.

Module QuatOps (F : FLOAT_OPS).

  Import F.

  (** Quaternion as a 4-tuple over F.t *)
  Record Quat := mkQuat {
    qw : t;
    qx : t;
    qy : t;
    qz : t;
  }.

  (** 3-vector as a 3-tuple over F.t *)
  Record Vec3 := mkVec3 {
    vx : t;
    vy : t;
    vz : t;
  }.

  Definition quat_zero : Quat := mkQuat zero zero zero zero.
  Definition quat_one : Quat := mkQuat one zero zero zero.

  Definition quat_add (p q : Quat) : Quat :=
    mkQuat (add (qw p) (qw q))
           (add (qx p) (qx q))
           (add (qy p) (qy q))
           (add (qz p) (qz q)).

  Definition quat_scale (a : t) (q : Quat) : Quat :=
    mkQuat (mul a (qw q)) (mul a (qx q)) (mul a (qy q)) (mul a (qz q)).

  Definition quat_neg (q : Quat) : Quat :=
    mkQuat (opp (qw q)) (opp (qx q)) (opp (qy q)) (opp (qz q)).

  Definition qi : Quat := mkQuat zero one zero zero.
  Definition qj : Quat := mkQuat zero zero one zero.
  Definition qk : Quat := mkQuat zero zero zero one.

  Definition quat_imag_basis (k : nat) : Quat :=
    match k with
    | 1%nat => qi
    | 2%nat => qj
    | 3%nat => qk
    | _ => quat_zero
    end.

  Definition quat_near_identity (k : nat) (eps : t) : Quat :=
    quat_add quat_one (quat_scale eps (quat_imag_basis k)).

  Lemma add_zero_r : forall x : t, add x zero = x.
  Proof.
    intro x.
    rewrite add_comm.
    exact (add_zero_l x).
  Qed.

  Lemma add_opp_l : forall x : t, add (opp x) x = zero.
  Proof.
    intro x.
    rewrite add_comm.
    exact (add_opp_r x).
  Qed.

  Lemma add_cancel_l : forall a b c : t, add a b = add a c -> b = c.
  Proof.
    intros a b c H.
    pose proof (f_equal (fun u => add (opp a) u) H) as H'.
    rewrite <- add_assoc in H'.
    rewrite <- add_assoc in H'.
    repeat rewrite add_opp_l in H'.
    repeat rewrite add_zero_l in H'.
    exact H'.
  Qed.

  Lemma opp_involutive : forall x : t, opp (opp x) = x.
  Proof.
    intro x.
    apply (add_cancel_l (opp x)).
    rewrite add_opp_r.
    symmetry.
    exact (add_opp_l x).
  Qed.

  Lemma opp_zero : opp zero = zero.
  Proof.
    pose proof (add_opp_r zero) as H.
    rewrite add_zero_l in H.
    exact H.
  Qed.

  Lemma mul_zero_l : forall x : t, mul x zero = zero.
  Proof.
    intro x.
    pose proof (mul_add_distr_l x zero zero) as H.
    rewrite add_zero_l in H.
    pose proof (f_equal (fun u => add (opp (mul x zero)) u) H) as H'.
    rewrite add_opp_l in H'.
    rewrite <- add_assoc in H'.
    rewrite add_opp_l in H'.
    rewrite add_zero_l in H'.
    symmetry.
    exact H'.
  Qed.

  Lemma mul_zero_r : forall x : t, mul zero x = zero.
  Proof.
    intro x.
    rewrite mul_comm.
    exact (mul_zero_l x).
  Qed.

  Lemma mul_one_r : forall x : t, mul x one = x.
  Proof.
    intro x.
    rewrite mul_comm.
    exact (mul_one_l x).
  Qed.

  Lemma opp_mul_l : forall x y : t, mul (opp x) y = opp (mul x y).
  Proof.
    intros x y.
    pose proof (mul_add_distr_l y (opp x) x) as H.
    rewrite add_opp_l in H.
    rewrite mul_zero_l in H.
    rewrite mul_comm in H.
    rewrite (mul_comm y x) in H.
    rewrite add_comm in H.
    symmetry in H.
    apply (add_cancel_l (mul x y)).
    rewrite add_opp_r.
    exact H.
  Qed.

  Lemma opp_mul_r : forall x y : t, mul x (opp y) = opp (mul x y).
  Proof.
    intros x y.
    rewrite mul_comm.
    rewrite opp_mul_l.
    rewrite mul_comm.
    reflexivity.
  Qed.

  Lemma opp_mul_opp : forall x : t, mul (opp x) (opp x) = mul x x.
  Proof.
    intro x.
    rewrite opp_mul_l.
    rewrite opp_mul_r.
    exact (opp_involutive (mul x x)).
  Qed.

  Lemma mul_add_distr_r : forall x y z : t,
    mul (add x y) z = add (mul x z) (mul y z).
  Proof.
    intros x y z.
    rewrite mul_comm.
    rewrite mul_add_distr_l.
    rewrite (mul_comm z x).
    rewrite (mul_comm z y).
    reflexivity.
  Qed.

  Lemma float_ring_theory : ring_theory zero one add mul sub opp (@eq t).
  Proof.
    constructor.
    - exact add_zero_l.
    - exact add_comm.
    - intros x y z. symmetry. exact (add_assoc x y z).
    - exact mul_one_l.
    - exact mul_comm.
    - intros x y z. symmetry. exact (mul_assoc x y z).
    - exact mul_add_distr_r.
    - exact sub_def.
    - exact add_opp_r.
  Qed.

  Add Ring float_ring : float_ring_theory.

  (** Hamilton product *)
  Definition quat_mul (p q : Quat) : Quat :=
    mkQuat
      (sub (sub (sub (mul (qw p) (qw q)) (mul (qx p) (qx q)))
                (mul (qy p) (qy q)))
           (mul (qz p) (qz q)))
      (add (add (add (mul (qw p) (qx q)) (mul (qx p) (qw q)))
                (mul (qy p) (qz q)))
           (opp (mul (qz p) (qy q))))
      (add (add (sub (mul (qw p) (qy q)) (mul (qx p) (qz q)))
                (mul (qy p) (qw q)))
           (mul (qz p) (qx q)))
      (add (add (add (mul (qw p) (qz q)) (mul (qx p) (qy q)))
                (opp (mul (qy p) (qx q))))
           (mul (qz p) (qw q))).

  (** Quaternion conjugate *)
  Definition quat_conj (q : Quat) : Quat :=
    mkQuat (qw q) (opp (qx q)) (opp (qy q)) (opp (qz q)).

  (** Embed a 3-vector as a pure quaternion *)
  Definition embed_vec (v : Vec3) : Quat :=
    mkQuat zero (vx v) (vy v) (vz v).

  (** Extract the vector part of a quaternion *)
  Definition extract_vec (q : Quat) : Vec3 :=
    mkVec3 (qx q) (qy q) (qz q).

  (*<*quatrotate>*)
  (** Quaternion rotation: v' = Im(q * embed(v) * conj(q)) *)
  Definition quat_rotate (q : Quat) (v : Vec3) : Vec3 :=
    extract_vec (quat_mul (quat_mul q (embed_vec v)) (quat_conj q)).
  (*</quatrotate>*)

  (** Squared norm of a quaternion *)
  Definition quat_norm_sq (q : Quat) : t :=
    add (add (add (mul (qw q) (qw q)) (mul (qx q) (qx q)))
             (mul (qy q) (qy q)))
        (mul (qz q) (qz q)).

  Theorem quat_mul_one_left : forall q : Quat,
    quat_mul quat_one q = q.
  Proof.
    intros [w x y z].
    unfold quat_mul, quat_one.
    simpl.
    repeat rewrite sub_def.
    repeat rewrite mul_one_l.
    repeat rewrite mul_zero_r.
    repeat rewrite opp_zero.
    repeat rewrite add_zero_r.
    repeat rewrite add_zero_l.
    reflexivity.
  Qed.

  Theorem quat_mul_one_right : forall q : Quat,
    quat_mul q quat_one = q.
  Proof.
    intros [w x y z].
    unfold quat_mul, quat_one.
    simpl.
    repeat rewrite sub_def.
    repeat rewrite mul_one_r.
    repeat rewrite mul_zero_l.
    repeat rewrite opp_zero.
    repeat rewrite add_zero_r.
    repeat rewrite add_zero_l.
    reflexivity.
  Qed.

  Theorem quat_conj_involution : forall q : Quat,
    quat_conj (quat_conj q) = q.
  Proof.
    intros [w x y z].
    unfold quat_conj.
    simpl.
    f_equal;
      try exact (opp_involutive x);
      try exact (opp_involutive y);
      try exact (opp_involutive z).
  Qed.

  Theorem quat_norm_sq_conj : forall q : Quat,
    quat_norm_sq (quat_conj q) = quat_norm_sq q.
  Proof.
    intros [w x y z].
    unfold quat_norm_sq, quat_conj.
    simpl.
    repeat rewrite opp_mul_opp.
    reflexivity.
  Qed.

  Theorem quat_norm_conjugate : forall q : Quat,
    quat_mul q (quat_conj q) = quat_scale (quat_norm_sq q) quat_one.
  Proof.
    intros [w x y z].
    unfold quat_mul, quat_conj, quat_scale, quat_norm_sq, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_conj_norm_left : forall q : Quat,
    quat_mul (quat_conj q) q = quat_scale (quat_norm_sq q) quat_one.
  Proof.
    intros [w x y z].
    unfold quat_mul, quat_conj, quat_scale, quat_norm_sq, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_norm_mul : forall p q : Quat,
    quat_norm_sq (quat_mul p q) = mul (quat_norm_sq p) (quat_norm_sq q).
  Proof.
    intros [a b c d] [e f g h].
    unfold quat_norm_sq, quat_mul.
    simpl.
    ring.
  Qed.

  Theorem quat_mul_assoc : forall x y z : Quat,
    quat_mul (quat_mul x y) z = quat_mul x (quat_mul y z).
  Proof.
    intros [a b c d] [e f g h] [i j k l].
    unfold quat_mul.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_i_sq : quat_mul qi qi = quat_scale (opp one) quat_one.
  Proof.
    unfold quat_mul, qi, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_j_sq : quat_mul qj qj = quat_scale (opp one) quat_one.
  Proof.
    unfold quat_mul, qj, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_k_sq : quat_mul qk qk = quat_scale (opp one) quat_one.
  Proof.
    unfold quat_mul, qk, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_ij : quat_mul qi qj = qk.
  Proof.
    unfold quat_mul, qi, qj, qk.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_ji : quat_mul qj qi = quat_neg qk.
  Proof.
    unfold quat_mul, qi, qj, qk, quat_neg.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_jk : quat_mul qj qk = qi.
  Proof.
    unfold quat_mul, qi, qj, qk.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_kj : quat_mul qk qj = quat_neg qi.
  Proof.
    unfold quat_mul, qi, qj, qk, quat_neg.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_ki : quat_mul qk qi = qj.
  Proof.
    unfold quat_mul, qi, qj, qk.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_ik : quat_mul qi qk = quat_neg qj.
  Proof.
    unfold quat_mul, qi, qj, qk, quat_neg.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_basis_table :
    quat_mul qi qi = quat_scale (opp one) quat_one /\
    quat_mul qj qj = quat_scale (opp one) quat_one /\
    quat_mul qk qk = quat_scale (opp one) quat_one /\
    quat_mul qi qj = qk /\
    quat_mul qj qk = qi /\
    quat_mul qk qi = qj /\
    quat_mul qj qi = quat_neg qk /\
    quat_mul qk qj = quat_neg qi /\
    quat_mul qi qk = quat_neg qj.
  Proof.
    repeat split.
    - exact quat_i_sq.
    - exact quat_j_sq.
    - exact quat_k_sq.
    - exact quat_ij.
    - exact quat_jk.
    - exact quat_ki.
    - exact quat_ji.
    - exact quat_kj.
    - exact quat_ik.
  Qed.

  Definition quat_coord (q : Quat) (i : nat) : t :=
    match i with
    | 0%nat => qw q
    | 1%nat => qx q
    | 2%nat => qy q
    | 3%nat => qz q
    | _ => zero
    end.

  Definition quat_identity_entry (i j : nat) : t :=
    if Nat.eqb i j then one else zero.

  Definition quat_hamilton_basis_entry (k i j : nat) : t :=
    match k, i, j with
    | 0%nat, 0%nat, 0%nat => one
    | 0%nat, 1%nat, 1%nat => one
    | 0%nat, 2%nat, 2%nat => one
    | 0%nat, 3%nat, 3%nat => one
    | 1%nat, 0%nat, 1%nat => opp one
    | 1%nat, 1%nat, 0%nat => one
    | 1%nat, 2%nat, 3%nat => opp one
    | 1%nat, 3%nat, 2%nat => one
    | 2%nat, 0%nat, 2%nat => opp one
    | 2%nat, 1%nat, 3%nat => one
    | 2%nat, 2%nat, 0%nat => one
    | 2%nat, 3%nat, 1%nat => opp one
    | 3%nat, 0%nat, 3%nat => opp one
    | 3%nat, 1%nat, 2%nat => opp one
    | 3%nat, 2%nat, 1%nat => one
    | 3%nat, 3%nat, 0%nat => one
    | _, _, _ => zero
    end.

  Definition quat_hamilton_entry (x : Quat) (i j : nat) : t :=
    qw x * quat_hamilton_basis_entry 0%nat i j +
    qx x * quat_hamilton_basis_entry 1%nat i j +
    qy x * quat_hamilton_basis_entry 2%nat i j +
    qz x * quat_hamilton_basis_entry 3%nat i j.

  Definition quat_hamilton_apply (x xi : Quat) : Quat :=
    mkQuat
      (quat_hamilton_entry x 0%nat 0%nat * quat_coord xi 0%nat +
       quat_hamilton_entry x 0%nat 1%nat * quat_coord xi 1%nat +
       quat_hamilton_entry x 0%nat 2%nat * quat_coord xi 2%nat +
       quat_hamilton_entry x 0%nat 3%nat * quat_coord xi 3%nat)
      (quat_hamilton_entry x 1%nat 0%nat * quat_coord xi 0%nat +
       quat_hamilton_entry x 1%nat 1%nat * quat_coord xi 1%nat +
       quat_hamilton_entry x 1%nat 2%nat * quat_coord xi 2%nat +
       quat_hamilton_entry x 1%nat 3%nat * quat_coord xi 3%nat)
      (quat_hamilton_entry x 2%nat 0%nat * quat_coord xi 0%nat +
       quat_hamilton_entry x 2%nat 1%nat * quat_coord xi 1%nat +
       quat_hamilton_entry x 2%nat 2%nat * quat_coord xi 2%nat +
       quat_hamilton_entry x 2%nat 3%nat * quat_coord xi 3%nat)
      (quat_hamilton_entry x 3%nat 0%nat * quat_coord xi 0%nat +
       quat_hamilton_entry x 3%nat 1%nat * quat_coord xi 1%nat +
       quat_hamilton_entry x 3%nat 2%nat * quat_coord xi 2%nat +
       quat_hamilton_entry x 3%nat 3%nat * quat_coord xi 3%nat).

  Definition quat_matrix_transpose (m : nat -> nat -> t) (i j : nat) : t :=
    m j i.

  Definition quat_matrix_mul_entry (m n : nat -> nat -> t) (i j : nat) : t :=
    m i 0%nat * n 0%nat j +
    m i 1%nat * n 1%nat j +
    m i 2%nat * n 2%nat j +
    m i 3%nat * n 3%nat j.

  Definition quat_hamilton_gram_entry (x : Quat) (i j : nat) : t :=
    quat_matrix_mul_entry (quat_matrix_transpose (quat_hamilton_entry x))
                          (quat_hamilton_entry x) i j.

  Theorem quat_hamilton_apply_eq_mul : forall x xi : Quat,
    quat_hamilton_apply x xi = quat_mul x xi.
  Proof.
    intros [a b c d] [e f g h].
    unfold quat_hamilton_apply, quat_hamilton_entry, quat_hamilton_basis_entry,
           quat_coord, quat_mul.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_eq13_dim4 : forall x xi : Quat,
    quat_norm_sq (quat_hamilton_apply x xi) = quat_norm_sq x * quat_norm_sq xi.
  Proof.
    intros x xi.
    rewrite quat_hamilton_apply_eq_mul.
    exact (quat_norm_mul x xi).
  Qed.

  Theorem quat_eq15_dim4 : forall x (i j : nat),
    (i < 4)%nat ->
    (j < 4)%nat ->
    quat_hamilton_gram_entry x i j = quat_norm_sq x * quat_identity_entry i j.
  Proof.
    intros [a b c d] i j Hi Hj.
    unfold quat_hamilton_gram_entry, quat_matrix_mul_entry, quat_matrix_transpose,
           quat_hamilton_entry, quat_hamilton_basis_entry, quat_identity_entry,
           quat_norm_sq.
    destruct i as [|[|[|[|i]]]];
    destruct j as [|[|[|[|j]]]];
    simpl in Hi, Hj; try lia; simpl; ring.
  Qed.

  Theorem quat_eq16_basis_skew : forall (k i j : nat),
    (1 <= k <= 3)%nat ->
    (i < 4)%nat ->
    (j < 4)%nat ->
    quat_matrix_transpose (quat_hamilton_basis_entry k) i j =
    opp (quat_hamilton_basis_entry k i j).
  Proof.
    intros k i j Hk Hi Hj.
    unfold quat_matrix_transpose, quat_hamilton_basis_entry.
    destruct k as [|[|[|[|k]]]];
    destruct i as [|[|[|[|i]]]];
    destruct j as [|[|[|[|j]]]];
    simpl in Hk, Hi, Hj; try lia; simpl; ring.
  Qed.

  Theorem quat_eq16_basis_square : forall (k i j : nat),
    (1 <= k <= 3)%nat ->
    (i < 4)%nat ->
    (j < 4)%nat ->
    quat_matrix_mul_entry (quat_hamilton_basis_entry k)
                          (quat_hamilton_basis_entry k) i j =
    opp (quat_identity_entry i j).
  Proof.
    intros k i j Hk Hi Hj.
    unfold quat_matrix_mul_entry, quat_hamilton_basis_entry, quat_identity_entry.
    destruct k as [|[|[|[|k]]]];
    destruct i as [|[|[|[|i]]]];
    destruct j as [|[|[|[|j]]]];
    simpl in Hk, Hi, Hj; try lia; simpl; ring.
  Qed.

  Theorem quat_eq16_basis_anticommute : forall (k l i j : nat),
    (1 <= k <= 3)%nat ->
    (1 <= l <= 3)%nat ->
    k <> l ->
    (i < 4)%nat ->
    (j < 4)%nat ->
    quat_matrix_mul_entry (quat_hamilton_basis_entry k)
                          (quat_hamilton_basis_entry l) i j +
    quat_matrix_mul_entry (quat_hamilton_basis_entry l)
                          (quat_hamilton_basis_entry k) i j = zero.
  Proof.
    intros k l i j Hk Hl Hneq Hi Hj.
    unfold quat_matrix_mul_entry, quat_hamilton_basis_entry.
    destruct k as [|[|[|[|k]]]];
    destruct l as [|[|[|[|l]]]];
    destruct i as [|[|[|[|i]]]];
    destruct j as [|[|[|[|j]]]];
    simpl in Hk, Hl, Hneq, Hi, Hj; try lia; simpl in *; try contradiction; ring.
  Qed.

  Definition quat_param_mul (c2 c3 : t) (p q : Quat) : Quat :=
    mkQuat
      (qw p * qw q + c2 * qx p * qx q + c3 * qy p * qy q -
       c2 * c3 * qz p * qz q)
      (qw p * qx q + qx p * qw q - c3 * qy p * qz q +
       c3 * qz p * qy q)
      (qw p * qy q + qy p * qw q + c2 * qx p * qz q -
       c2 * qz p * qx q)
      (qw p * qz q + qz p * qw q + qx p * qy q -
       qy p * qx q).

  Definition quat_param_norm (c2 c3 : t) (q : Quat) : t :=
    qw q * qw q - c2 * (qx q * qx q) - c3 * (qy q * qy q) +
    c2 * c3 * (qz q * qz q).

  Theorem quat_param_mul_one_left : forall c2 c3 q,
    quat_param_mul c2 c3 quat_one q = q.
  Proof.
    intros c2 c3 [a b c d].
    unfold quat_param_mul, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_mul_one_right : forall c2 c3 q,
    quat_param_mul c2 c3 q quat_one = q.
  Proof.
    intros c2 c3 [a b c d].
    unfold quat_param_mul, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_norm_conj_preserved : forall c2 c3 q,
    quat_param_norm c2 c3 (quat_conj q) = quat_param_norm c2 c3 q.
  Proof.
    intros c2 c3 [a b c d].
    unfold quat_param_norm, quat_conj.
    simpl.
    ring.
  Qed.

  Theorem quat_param_norm_conj_right : forall c2 c3 q,
    quat_param_mul c2 c3 q (quat_conj q) =
    quat_scale (quat_param_norm c2 c3 q) quat_one.
  Proof.
    intros c2 c3 [a b c d].
    unfold quat_param_mul, quat_param_norm, quat_conj, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_norm_conj_left : forall c2 c3 q,
    quat_param_mul c2 c3 (quat_conj q) q =
    quat_scale (quat_param_norm c2 c3 q) quat_one.
  Proof.
    intros c2 c3 [a b c d].
    unfold quat_param_mul, quat_param_norm, quat_conj, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_norm_mul : forall c2 c3 p q,
    quat_param_norm c2 c3 (quat_param_mul c2 c3 p q) =
    quat_param_norm c2 c3 p * quat_param_norm c2 c3 q.
  Proof.
    intros c2 c3 [a b c d] [e f g h].
    unfold quat_param_norm, quat_param_mul.
    simpl.
    ring.
  Qed.

  Theorem quat_param_mul_assoc : forall c2 c3 x y z,
    quat_param_mul c2 c3 (quat_param_mul c2 c3 x y) z =
    quat_param_mul c2 c3 x (quat_param_mul c2 c3 y z).
  Proof.
    intros c2 c3 [a b c d] [e f g h] [i j k l].
    unfold quat_param_mul.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qi_sq : forall c2 c3,
    quat_param_mul c2 c3 qi qi = quat_scale c2 quat_one.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qj_sq : forall c2 c3,
    quat_param_mul c2 c3 qj qj = quat_scale c3 quat_one.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qj, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qk_sq : forall c2 c3,
    quat_param_mul c2 c3 qk qk = quat_scale (opp c2 * c3) quat_one.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qk, quat_scale, quat_one.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qi_qj : forall c2 c3,
    quat_param_mul c2 c3 qi qj = qk.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qj, qk.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qj_qi : forall c2 c3,
    quat_param_mul c2 c3 qj qi = quat_neg qk.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qj, qk, quat_neg.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qi_qk : forall c2 c3,
    quat_param_mul c2 c3 qi qk = quat_scale c2 qj.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qk, qj, quat_scale.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qk_qi : forall c2 c3,
    quat_param_mul c2 c3 qk qi = quat_scale (opp c2) qj.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qk, qj, quat_scale.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qj_qk : forall c2 c3,
    quat_param_mul c2 c3 qj qk = quat_scale (opp c3) qi.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qj, qk, quat_scale.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_qk_qj : forall c2 c3,
    quat_param_mul c2 c3 qk qj = quat_scale c3 qi.
  Proof.
    intros c2 c3.
    unfold quat_param_mul, qi, qj, qk, quat_scale.
    simpl.
    f_equal; ring.
  Qed.

  Definition quat_param_form_entry (c2 c3 : t) (i j : nat) : t :=
    match i, j with
    | 0%nat, 0%nat => one
    | 1%nat, 1%nat => opp c2
    | 2%nat, 2%nat => opp c3
    | 3%nat, 3%nat => c2 * c3
    | _, _ => zero
    end.

  Definition quat_param_matrix_entry (c2 c3 : t) (x : Quat) (i j : nat) : t :=
    match i, j with
    | 0%nat, 0%nat => qw x
    | 0%nat, 1%nat => c2 * qx x
    | 0%nat, 2%nat => c3 * qy x
    | 0%nat, 3%nat => opp c2 * c3 * qz x
    | 1%nat, 0%nat => qx x
    | 1%nat, 1%nat => qw x
    | 1%nat, 2%nat => c3 * qz x
    | 1%nat, 3%nat => opp c3 * qy x
    | 2%nat, 0%nat => qy x
    | 2%nat, 1%nat => opp c2 * qz x
    | 2%nat, 2%nat => qw x
    | 2%nat, 3%nat => c2 * qx x
    | 3%nat, 0%nat => qz x
    | 3%nat, 1%nat => opp (qy x)
    | 3%nat, 2%nat => qx x
    | 3%nat, 3%nat => qw x
    | _, _ => zero
    end.

  Definition quat_param_matrix_apply (c2 c3 : t) (x xi : Quat) : Quat :=
    mkQuat
      (quat_param_matrix_entry c2 c3 x 0%nat 0%nat * quat_coord xi 0%nat +
       quat_param_matrix_entry c2 c3 x 0%nat 1%nat * quat_coord xi 1%nat +
       quat_param_matrix_entry c2 c3 x 0%nat 2%nat * quat_coord xi 2%nat +
       quat_param_matrix_entry c2 c3 x 0%nat 3%nat * quat_coord xi 3%nat)
      (quat_param_matrix_entry c2 c3 x 1%nat 0%nat * quat_coord xi 0%nat +
       quat_param_matrix_entry c2 c3 x 1%nat 1%nat * quat_coord xi 1%nat +
       quat_param_matrix_entry c2 c3 x 1%nat 2%nat * quat_coord xi 2%nat +
       quat_param_matrix_entry c2 c3 x 1%nat 3%nat * quat_coord xi 3%nat)
      (quat_param_matrix_entry c2 c3 x 2%nat 0%nat * quat_coord xi 0%nat +
       quat_param_matrix_entry c2 c3 x 2%nat 1%nat * quat_coord xi 1%nat +
       quat_param_matrix_entry c2 c3 x 2%nat 2%nat * quat_coord xi 2%nat +
       quat_param_matrix_entry c2 c3 x 2%nat 3%nat * quat_coord xi 3%nat)
      (quat_param_matrix_entry c2 c3 x 3%nat 0%nat * quat_coord xi 0%nat +
       quat_param_matrix_entry c2 c3 x 3%nat 1%nat * quat_coord xi 1%nat +
       quat_param_matrix_entry c2 c3 x 3%nat 2%nat * quat_coord xi 2%nat +
       quat_param_matrix_entry c2 c3 x 3%nat 3%nat * quat_coord xi 3%nat).

  Definition quat_param_gram_entry (c2 c3 : t) (x : Quat) (i j : nat) : t :=
    quat_param_form_entry c2 c3 0%nat 0%nat *
      quat_param_matrix_entry c2 c3 x 0%nat i *
      quat_param_matrix_entry c2 c3 x 0%nat j +
    quat_param_form_entry c2 c3 1%nat 1%nat *
      quat_param_matrix_entry c2 c3 x 1%nat i *
      quat_param_matrix_entry c2 c3 x 1%nat j +
    quat_param_form_entry c2 c3 2%nat 2%nat *
      quat_param_matrix_entry c2 c3 x 2%nat i *
      quat_param_matrix_entry c2 c3 x 2%nat j +
    quat_param_form_entry c2 c3 3%nat 3%nat *
      quat_param_matrix_entry c2 c3 x 3%nat i *
      quat_param_matrix_entry c2 c3 x 3%nat j.

  Definition quat_param_eq6_delta_coord
      (c2 c3 : t) (k : nat) (x : Quat) (i : nat) : t :=
    match k, i with
    | 1%nat, 0%nat => c2 * quat_coord x 1%nat
    | 1%nat, 1%nat => quat_coord x 0%nat
    | 1%nat, 2%nat => opp c2 * quat_coord x 3%nat
    | 1%nat, 3%nat => opp (quat_coord x 2%nat)
    | 2%nat, 0%nat => c3 * quat_coord x 2%nat
    | 2%nat, 1%nat => c3 * quat_coord x 3%nat
    | 2%nat, 2%nat => quat_coord x 0%nat
    | 2%nat, 3%nat => quat_coord x 1%nat
    | 3%nat, 0%nat => opp c2 * c3 * quat_coord x 3%nat
    | 3%nat, 1%nat => opp c3 * quat_coord x 2%nat
    | 3%nat, 2%nat => c2 * quat_coord x 1%nat
    | 3%nat, 3%nat => quat_coord x 0%nat
    | _, _ => zero
    end.

  Definition quat_param_metric_weight (c2 c3 : t) (i : nat) : t :=
    match i with
    | 0%nat => one
    | 1%nat => opp c2
    | 2%nat => opp c3
    | 3%nat => c2 * c3
    | _ => zero
    end.

  Definition quat_param_eq7_linear_form
      (c2 c3 : t) (k : nat) (x : Quat) : t :=
    quat_param_metric_weight c2 c3 0%nat *
      quat_coord x 0%nat * quat_param_eq6_delta_coord c2 c3 k x 0%nat +
    quat_param_metric_weight c2 c3 1%nat *
      quat_coord x 1%nat * quat_param_eq6_delta_coord c2 c3 k x 1%nat +
    quat_param_metric_weight c2 c3 2%nat *
      quat_coord x 2%nat * quat_param_eq6_delta_coord c2 c3 k x 2%nat +
    quat_param_metric_weight c2 c3 3%nat *
      quat_coord x 3%nat * quat_param_eq6_delta_coord c2 c3 k x 3%nat.

  Definition quat_param_near_identity_quadratic_factor
      (c2 c3 : t) (k : nat) : t :=
    match k with
    | 1%nat => opp c2
    | 2%nat => opp c3
    | 3%nat => c2 * c3
    | _ => zero
    end.

  Theorem quat_param_matrix_apply_eq_mul : forall c2 c3 x xi,
    quat_param_matrix_apply c2 c3 x xi = quat_param_mul c2 c3 x xi.
  Proof.
    intros c2 c3 [a b c d] [e f g h].
    unfold quat_param_matrix_apply, quat_param_matrix_entry,
           quat_param_mul, quat_coord.
    simpl.
    f_equal; ring.
  Qed.

  Theorem quat_param_eq13_dim4 : forall c2 c3 x xi,
    quat_param_norm c2 c3 (quat_param_matrix_apply c2 c3 x xi) =
    quat_param_norm c2 c3 x * quat_param_norm c2 c3 xi.
  Proof.
    intros c2 c3 x xi.
    rewrite quat_param_matrix_apply_eq_mul.
    exact (quat_param_norm_mul c2 c3 x xi).
  Qed.

  Theorem quat_param_eq15_dim4 : forall c2 c3 x i j,
    (i < 4)%nat ->
    (j < 4)%nat ->
    quat_param_gram_entry c2 c3 x i j =
    quat_param_norm c2 c3 x * quat_param_form_entry c2 c3 i j.
  Proof.
    intros c2 c3 [a b c d] i j Hi Hj.
    unfold quat_param_gram_entry, quat_param_form_entry, quat_param_matrix_entry,
           quat_param_norm.
    destruct i as [|[|[|[|i]]]];
    destruct j as [|[|[|[|j]]]];
    simpl in Hi, Hj; try lia; simpl; ring.
  Qed.

  Theorem quat_param_eq6_dim4 : forall c2 c3 x k eps i,
    (1 <= k <= 3)%nat ->
    (i < 4)%nat ->
    quat_coord (quat_param_mul c2 c3 x (quat_near_identity k eps)) i =
    quat_coord x i + eps * quat_param_eq6_delta_coord c2 c3 k x i.
  Proof.
    intros c2 c3 [a b c d] k eps i Hk Hi.
    destruct k as [|[|[|[|k]]]];
    destruct i as [|[|[|[|i]]]];
    simpl in Hk, Hi; try lia;
    unfold quat_param_mul, quat_near_identity, quat_imag_basis, quat_add,
           quat_one, quat_scale, quat_param_eq6_delta_coord, quat_coord,
           qi, qj, qk, quat_zero;
    simpl; ring.
  Qed.

  Theorem quat_param_near_identity_norm : forall c2 c3 k eps,
    (1 <= k <= 3)%nat ->
    quat_param_norm c2 c3 (quat_near_identity k eps) =
    one + quat_param_near_identity_quadratic_factor c2 c3 k * (eps * eps).
  Proof.
    intros c2 c3 k eps Hk.
    destruct k as [|[|[|[|k]]]];
    simpl in Hk; try lia;
    unfold quat_param_norm, quat_near_identity, quat_imag_basis, quat_add,
           quat_one, quat_scale, quat_param_near_identity_quadratic_factor,
           qi, qj, qk, quat_zero;
    simpl; ring.
  Qed.

  Theorem quat_param_infinitesimal_no_linear_term : forall c2 c3 k eps x,
    (1 <= k <= 3)%nat ->
    quat_param_norm c2 c3
      (quat_param_mul c2 c3 x (quat_near_identity k eps)) -
    quat_param_norm c2 c3 x =
    (eps * eps) * quat_param_near_identity_quadratic_factor c2 c3 k *
      quat_param_norm c2 c3 x.
  Proof.
    intros c2 c3 k eps x Hk.
    rewrite quat_param_norm_mul.
    rewrite quat_param_near_identity_norm by exact Hk.
    ring.
  Qed.

  Theorem quat_param_eq7_linear_form_vanish : forall c2 c3 k x,
    (1 <= k <= 3)%nat ->
    quat_param_eq7_linear_form c2 c3 k x = zero.
  Proof.
    intros c2 c3 k [a b c d] Hk.
    destruct k as [|[|[|[|k]]]];
    simpl in Hk; try lia;
    unfold quat_param_eq7_linear_form, quat_param_metric_weight,
           quat_param_eq6_delta_coord, quat_coord;
    simpl; ring.
  Qed.

End QuatOps.
