
type nat =
| O
| S of nat

module Nat =
 struct
  (** val eqb : nat -> nat -> bool **)

  let rec eqb n m =
    match n with
    | O -> (match m with
            | O -> true
            | S _ -> false)
    | S n' -> (match m with
               | O -> false
               | S m' -> eqb n' m')
 end

module type FLOAT_OPS =
 sig
  type t

  val zero : t

  val one : t

  val add : t -> t -> t

  val mul : t -> t -> t

  val sub : t -> t -> t

  val neg : t -> t -> t

  val opp : t -> t

  val div : t -> t -> t

  val sqrt_f : t -> t
 end

module QuatOps =
 functor (F:FLOAT_OPS) ->
 struct
  type coq_Quat = { qw : F.t; qx : F.t; qy : F.t; qz : F.t }

  (** val qw : coq_Quat -> F.t **)

  let qw q =
    q.qw

  (** val qx : coq_Quat -> F.t **)

  let qx q =
    q.qx

  (** val qy : coq_Quat -> F.t **)

  let qy q =
    q.qy

  (** val qz : coq_Quat -> F.t **)

  let qz q =
    q.qz

  type coq_Vec3 = { vx : F.t; vy : F.t; vz : F.t }

  (** val vx : coq_Vec3 -> F.t **)

  let vx v =
    v.vx

  (** val vy : coq_Vec3 -> F.t **)

  let vy v =
    v.vy

  (** val vz : coq_Vec3 -> F.t **)

  let vz v =
    v.vz

  (** val quat_zero : coq_Quat **)

  let quat_zero =
    { qw = F.zero; qx = F.zero; qy = F.zero; qz = F.zero }

  (** val quat_one : coq_Quat **)

  let quat_one =
    { qw = F.one; qx = F.zero; qy = F.zero; qz = F.zero }

  (** val quat_add : coq_Quat -> coq_Quat -> coq_Quat **)

  let quat_add p q =
    { qw = (F.add p.qw q.qw); qx = (F.add p.qx q.qx); qy = (F.add p.qy q.qy);
      qz = (F.add p.qz q.qz) }

  (** val quat_scale : F.t -> coq_Quat -> coq_Quat **)

  let quat_scale a q =
    { qw = (F.mul a q.qw); qx = (F.mul a q.qx); qy = (F.mul a q.qy); qz =
      (F.mul a q.qz) }

  (** val quat_neg : coq_Quat -> coq_Quat **)

  let quat_neg q =
    { qw = (F.opp q.qw); qx = (F.opp q.qx); qy = (F.opp q.qy); qz =
      (F.opp q.qz) }

  (** val qi : coq_Quat **)

  let qi =
    { qw = F.zero; qx = F.one; qy = F.zero; qz = F.zero }

  (** val qj : coq_Quat **)

  let qj =
    { qw = F.zero; qx = F.zero; qy = F.one; qz = F.zero }

  (** val qk : coq_Quat **)

  let qk =
    { qw = F.zero; qx = F.zero; qy = F.zero; qz = F.one }

  (** val quat_imag_basis : nat -> coq_Quat **)

  let quat_imag_basis = function
  | O -> quat_zero
  | S n ->
    (match n with
     | O -> qi
     | S n0 ->
       (match n0 with
        | O -> qj
        | S n1 -> (match n1 with
                   | O -> qk
                   | S _ -> quat_zero)))

  (** val quat_near_identity : nat -> F.t -> coq_Quat **)

  let quat_near_identity k eps =
    quat_add quat_one (quat_scale eps (quat_imag_basis k))

  (** val quat_mul : coq_Quat -> coq_Quat -> coq_Quat **)

  let quat_mul p q =
    { qw =
      (F.sub
        (F.sub (F.sub (F.mul p.qw q.qw) (F.mul p.qx q.qx)) (F.mul p.qy q.qy))
        (F.mul p.qz q.qz));
      qx =
      (F.add
        (F.add (F.add (F.mul p.qw q.qx) (F.mul p.qx q.qw)) (F.mul p.qy q.qz))
        (F.opp (F.mul p.qz q.qy)));
      qy =
      (F.add
        (F.add (F.sub (F.mul p.qw q.qy) (F.mul p.qx q.qz)) (F.mul p.qy q.qw))
        (F.mul p.qz q.qx));
      qz =
      (F.add
        (F.add (F.add (F.mul p.qw q.qz) (F.mul p.qx q.qy))
          (F.opp (F.mul p.qy q.qx)))
        (F.mul p.qz q.qw)) }

  (** val quat_conj : coq_Quat -> coq_Quat **)

  let quat_conj q =
    { qw = q.qw; qx = (F.opp q.qx); qy = (F.opp q.qy); qz = (F.opp q.qz) }

  (** val embed_vec : coq_Vec3 -> coq_Quat **)

  let embed_vec v =
    { qw = F.zero; qx = v.vx; qy = v.vy; qz = v.vz }

  (** val extract_vec : coq_Quat -> coq_Vec3 **)

  let extract_vec q =
    { vx = q.qx; vy = q.qy; vz = q.qz }

  (** val quat_rotate : coq_Quat -> coq_Vec3 -> coq_Vec3 **)

  let quat_rotate q v =
    extract_vec (quat_mul (quat_mul q (embed_vec v)) (quat_conj q))

  (** val quat_norm_sq : coq_Quat -> F.t **)

  let quat_norm_sq q =
    F.add
      (F.add (F.add (F.mul q.qw q.qw) (F.mul q.qx q.qx)) (F.mul q.qy q.qy))
      (F.mul q.qz q.qz)

  (** val quat_coord : coq_Quat -> nat -> F.t **)

  let quat_coord q = function
  | O -> q.qw
  | S n ->
    (match n with
     | O -> q.qx
     | S n0 ->
       (match n0 with
        | O -> q.qy
        | S n1 -> (match n1 with
                   | O -> q.qz
                   | S _ -> F.zero)))

  (** val quat_identity_entry : nat -> nat -> F.t **)

  let quat_identity_entry i j =
    if Nat.eqb i j then F.one else F.zero

  (** val quat_hamilton_basis_entry : nat -> nat -> nat -> F.t **)

  let quat_hamilton_basis_entry k i j =
    match k with
    | O ->
      (match i with
       | O -> (match j with
               | O -> F.one
               | S _ -> F.zero)
       | S n ->
         (match n with
          | O ->
            (match j with
             | O -> F.zero
             | S n0 -> (match n0 with
                        | O -> F.one
                        | S _ -> F.zero))
          | S n0 ->
            (match n0 with
             | O ->
               (match j with
                | O -> F.zero
                | S n1 ->
                  (match n1 with
                   | O -> F.zero
                   | S n2 -> (match n2 with
                              | O -> F.one
                              | S _ -> F.zero)))
             | S n1 ->
               (match n1 with
                | O ->
                  (match j with
                   | O -> F.zero
                   | S n2 ->
                     (match n2 with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 -> (match n4 with
                                    | O -> F.one
                                    | S _ -> F.zero))))
                | S _ -> F.zero))))
    | S n ->
      (match n with
       | O ->
         (match i with
          | O ->
            (match j with
             | O -> F.zero
             | S n0 -> (match n0 with
                        | O -> F.opp F.one
                        | S _ -> F.zero))
          | S n0 ->
            (match n0 with
             | O -> (match j with
                     | O -> F.one
                     | S _ -> F.zero)
             | S n1 ->
               (match n1 with
                | O ->
                  (match j with
                   | O -> F.zero
                   | S n2 ->
                     (match n2 with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 ->
                           (match n4 with
                            | O -> F.opp F.one
                            | S _ -> F.zero))))
                | S n2 ->
                  (match n2 with
                   | O ->
                     (match j with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 -> (match n4 with
                                    | O -> F.one
                                    | S _ -> F.zero)))
                   | S _ -> F.zero))))
       | S n0 ->
         (match n0 with
          | O ->
            (match i with
             | O ->
               (match j with
                | O -> F.zero
                | S n1 ->
                  (match n1 with
                   | O -> F.zero
                   | S n2 -> (match n2 with
                              | O -> F.opp F.one
                              | S _ -> F.zero)))
             | S n1 ->
               (match n1 with
                | O ->
                  (match j with
                   | O -> F.zero
                   | S n2 ->
                     (match n2 with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 -> (match n4 with
                                    | O -> F.one
                                    | S _ -> F.zero))))
                | S n2 ->
                  (match n2 with
                   | O -> (match j with
                           | O -> F.one
                           | S _ -> F.zero)
                   | S n3 ->
                     (match n3 with
                      | O ->
                        (match j with
                         | O -> F.zero
                         | S n4 ->
                           (match n4 with
                            | O -> F.opp F.one
                            | S _ -> F.zero))
                      | S _ -> F.zero))))
          | S n1 ->
            (match n1 with
             | O ->
               (match i with
                | O ->
                  (match j with
                   | O -> F.zero
                   | S n2 ->
                     (match n2 with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 ->
                           (match n4 with
                            | O -> F.opp F.one
                            | S _ -> F.zero))))
                | S n2 ->
                  (match n2 with
                   | O ->
                     (match j with
                      | O -> F.zero
                      | S n3 ->
                        (match n3 with
                         | O -> F.zero
                         | S n4 ->
                           (match n4 with
                            | O -> F.opp F.one
                            | S _ -> F.zero)))
                   | S n3 ->
                     (match n3 with
                      | O ->
                        (match j with
                         | O -> F.zero
                         | S n4 -> (match n4 with
                                    | O -> F.one
                                    | S _ -> F.zero))
                      | S n4 ->
                        (match n4 with
                         | O -> (match j with
                                 | O -> F.one
                                 | S _ -> F.zero)
                         | S _ -> F.zero))))
             | S _ -> F.zero)))

  (** val quat_hamilton_entry : coq_Quat -> nat -> nat -> F.t **)

  let quat_hamilton_entry x i j =
    F.add
      (F.add
        (F.add (F.mul x.qw (quat_hamilton_basis_entry O i j))
          (F.mul x.qx (quat_hamilton_basis_entry (S O) i j)))
        (F.mul x.qy (quat_hamilton_basis_entry (S (S O)) i j)))
      (F.mul x.qz (quat_hamilton_basis_entry (S (S (S O))) i j))

  (** val quat_hamilton_apply : coq_Quat -> coq_Quat -> coq_Quat **)

  let quat_hamilton_apply x xi =
    { qw =
      (F.add
        (F.add
          (F.add (F.mul (quat_hamilton_entry x O O) (quat_coord xi O))
            (F.mul (quat_hamilton_entry x O (S O)) (quat_coord xi (S O))))
          (F.mul (quat_hamilton_entry x O (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_hamilton_entry x O (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qx =
      (F.add
        (F.add
          (F.add (F.mul (quat_hamilton_entry x (S O) O) (quat_coord xi O))
            (F.mul (quat_hamilton_entry x (S O) (S O)) (quat_coord xi (S O))))
          (F.mul (quat_hamilton_entry x (S O) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_hamilton_entry x (S O) (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qy =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_hamilton_entry x (S (S O)) O) (quat_coord xi O))
            (F.mul (quat_hamilton_entry x (S (S O)) (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_hamilton_entry x (S (S O)) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_hamilton_entry x (S (S O)) (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qz =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_hamilton_entry x (S (S (S O))) O) (quat_coord xi O))
            (F.mul (quat_hamilton_entry x (S (S (S O))) (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_hamilton_entry x (S (S (S O))) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_hamilton_entry x (S (S (S O))) (S (S (S O))))
          (quat_coord xi (S (S (S O)))))) }

  (** val quat_matrix_transpose : (nat -> nat -> F.t) -> nat -> nat -> F.t **)

  let quat_matrix_transpose m i j =
    m j i

  (** val quat_matrix_mul_entry :
      (nat -> nat -> F.t) -> (nat -> nat -> F.t) -> nat -> nat -> F.t **)

  let quat_matrix_mul_entry m n i j =
    F.add
      (F.add (F.add (F.mul (m i O) (n O j)) (F.mul (m i (S O)) (n (S O) j)))
        (F.mul (m i (S (S O))) (n (S (S O)) j)))
      (F.mul (m i (S (S (S O)))) (n (S (S (S O))) j))

  (** val quat_hamilton_gram_entry : coq_Quat -> nat -> nat -> F.t **)

  let quat_hamilton_gram_entry x i j =
    quat_matrix_mul_entry (quat_matrix_transpose (quat_hamilton_entry x))
      (quat_hamilton_entry x) i j

  (** val quat_param_mul : F.t -> F.t -> coq_Quat -> coq_Quat -> coq_Quat **)

  let quat_param_mul c2 c3 p q =
    { qw =
      (F.sub
        (F.add (F.add (F.mul p.qw q.qw) (F.mul (F.mul c2 p.qx) q.qx))
          (F.mul (F.mul c3 p.qy) q.qy))
        (F.mul (F.mul (F.mul c2 c3) p.qz) q.qz));
      qx =
      (F.add
        (F.sub (F.add (F.mul p.qw q.qx) (F.mul p.qx q.qw))
          (F.mul (F.mul c3 p.qy) q.qz))
        (F.mul (F.mul c3 p.qz) q.qy));
      qy =
      (F.sub
        (F.add (F.add (F.mul p.qw q.qy) (F.mul p.qy q.qw))
          (F.mul (F.mul c2 p.qx) q.qz))
        (F.mul (F.mul c2 p.qz) q.qx));
      qz =
      (F.sub
        (F.add (F.add (F.mul p.qw q.qz) (F.mul p.qz q.qw)) (F.mul p.qx q.qy))
        (F.mul p.qy q.qx)) }

  (** val quat_param_norm : F.t -> F.t -> coq_Quat -> F.t **)

  let quat_param_norm c2 c3 q =
    F.add
      (F.sub (F.sub (F.mul q.qw q.qw) (F.mul c2 (F.mul q.qx q.qx)))
        (F.mul c3 (F.mul q.qy q.qy)))
      (F.mul (F.mul c2 c3) (F.mul q.qz q.qz))

  (** val quat_param_form_entry : F.t -> F.t -> nat -> nat -> F.t **)

  let quat_param_form_entry c2 c3 i j =
    match i with
    | O -> (match j with
            | O -> F.one
            | S _ -> F.zero)
    | S n ->
      (match n with
       | O ->
         (match j with
          | O -> F.zero
          | S n0 -> (match n0 with
                     | O -> F.opp c2
                     | S _ -> F.zero))
       | S n0 ->
         (match n0 with
          | O ->
            (match j with
             | O -> F.zero
             | S n1 ->
               (match n1 with
                | O -> F.zero
                | S n2 -> (match n2 with
                           | O -> F.opp c3
                           | S _ -> F.zero)))
          | S n1 ->
            (match n1 with
             | O ->
               (match j with
                | O -> F.zero
                | S n2 ->
                  (match n2 with
                   | O -> F.zero
                   | S n3 ->
                     (match n3 with
                      | O -> F.zero
                      | S n4 ->
                        (match n4 with
                         | O -> F.mul c2 c3
                         | S _ -> F.zero))))
             | S _ -> F.zero)))

  (** val quat_param_matrix_entry :
      F.t -> F.t -> coq_Quat -> nat -> nat -> F.t **)

  let quat_param_matrix_entry c2 c3 x i j =
    match i with
    | O ->
      (match j with
       | O -> x.qw
       | S n ->
         (match n with
          | O -> F.mul c2 x.qx
          | S n0 ->
            (match n0 with
             | O -> F.mul c3 x.qy
             | S n1 ->
               (match n1 with
                | O -> F.mul (F.mul (F.opp c2) c3) x.qz
                | S _ -> F.zero))))
    | S n ->
      (match n with
       | O ->
         (match j with
          | O -> x.qx
          | S n0 ->
            (match n0 with
             | O -> x.qw
             | S n1 ->
               (match n1 with
                | O -> F.mul c3 x.qz
                | S n2 ->
                  (match n2 with
                   | O -> F.mul (F.opp c3) x.qy
                   | S _ -> F.zero))))
       | S n0 ->
         (match n0 with
          | O ->
            (match j with
             | O -> x.qy
             | S n1 ->
               (match n1 with
                | O -> F.mul (F.opp c2) x.qz
                | S n2 ->
                  (match n2 with
                   | O -> x.qw
                   | S n3 ->
                     (match n3 with
                      | O -> F.mul c2 x.qx
                      | S _ -> F.zero))))
          | S n1 ->
            (match n1 with
             | O ->
               (match j with
                | O -> x.qz
                | S n2 ->
                  (match n2 with
                   | O -> F.opp x.qy
                   | S n3 ->
                     (match n3 with
                      | O -> x.qx
                      | S n4 -> (match n4 with
                                 | O -> x.qw
                                 | S _ -> F.zero))))
             | S _ -> F.zero)))

  (** val quat_param_matrix_apply :
      F.t -> F.t -> coq_Quat -> coq_Quat -> coq_Quat **)

  let quat_param_matrix_apply c2 c3 x xi =
    { qw =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_param_matrix_entry c2 c3 x O O) (quat_coord xi O))
            (F.mul (quat_param_matrix_entry c2 c3 x O (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_param_matrix_entry c2 c3 x O (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_param_matrix_entry c2 c3 x O (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qx =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_param_matrix_entry c2 c3 x (S O) O)
              (quat_coord xi O))
            (F.mul (quat_param_matrix_entry c2 c3 x (S O) (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_param_matrix_entry c2 c3 x (S O) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_param_matrix_entry c2 c3 x (S O) (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qy =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_param_matrix_entry c2 c3 x (S (S O)) O)
              (quat_coord xi O))
            (F.mul (quat_param_matrix_entry c2 c3 x (S (S O)) (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_param_matrix_entry c2 c3 x (S (S O)) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_param_matrix_entry c2 c3 x (S (S O)) (S (S (S O))))
          (quat_coord xi (S (S (S O))))));
      qz =
      (F.add
        (F.add
          (F.add
            (F.mul (quat_param_matrix_entry c2 c3 x (S (S (S O))) O)
              (quat_coord xi O))
            (F.mul (quat_param_matrix_entry c2 c3 x (S (S (S O))) (S O))
              (quat_coord xi (S O))))
          (F.mul (quat_param_matrix_entry c2 c3 x (S (S (S O))) (S (S O)))
            (quat_coord xi (S (S O)))))
        (F.mul (quat_param_matrix_entry c2 c3 x (S (S (S O))) (S (S (S O))))
          (quat_coord xi (S (S (S O)))))) }

  (** val quat_param_gram_entry :
      F.t -> F.t -> coq_Quat -> nat -> nat -> F.t **)

  let quat_param_gram_entry c2 c3 x i j =
    F.add
      (F.add
        (F.add
          (F.mul
            (F.mul (quat_param_form_entry c2 c3 O O)
              (quat_param_matrix_entry c2 c3 x O i))
            (quat_param_matrix_entry c2 c3 x O j))
          (F.mul
            (F.mul (quat_param_form_entry c2 c3 (S O) (S O))
              (quat_param_matrix_entry c2 c3 x (S O) i))
            (quat_param_matrix_entry c2 c3 x (S O) j)))
        (F.mul
          (F.mul (quat_param_form_entry c2 c3 (S (S O)) (S (S O)))
            (quat_param_matrix_entry c2 c3 x (S (S O)) i))
          (quat_param_matrix_entry c2 c3 x (S (S O)) j)))
      (F.mul
        (F.mul (quat_param_form_entry c2 c3 (S (S (S O))) (S (S (S O))))
          (quat_param_matrix_entry c2 c3 x (S (S (S O))) i))
        (quat_param_matrix_entry c2 c3 x (S (S (S O))) j))

  (** val quat_param_eq6_delta_coord :
      F.t -> F.t -> nat -> coq_Quat -> nat -> F.t **)

  let quat_param_eq6_delta_coord c2 c3 k x i =
    match k with
    | O -> F.zero
    | S n ->
      (match n with
       | O ->
         (match i with
          | O -> F.mul c2 (quat_coord x (S O))
          | S n0 ->
            (match n0 with
             | O -> quat_coord x O
             | S n1 ->
               (match n1 with
                | O -> F.mul (F.opp c2) (quat_coord x (S (S (S O))))
                | S n2 ->
                  (match n2 with
                   | O -> F.opp (quat_coord x (S (S O)))
                   | S _ -> F.zero))))
       | S n0 ->
         (match n0 with
          | O ->
            (match i with
             | O -> F.mul c3 (quat_coord x (S (S O)))
             | S n1 ->
               (match n1 with
                | O -> F.mul c3 (quat_coord x (S (S (S O))))
                | S n2 ->
                  (match n2 with
                   | O -> quat_coord x O
                   | S n3 ->
                     (match n3 with
                      | O -> quat_coord x (S O)
                      | S _ -> F.zero))))
          | S n1 ->
            (match n1 with
             | O ->
               (match i with
                | O ->
                  F.mul (F.mul (F.opp c2) c3) (quat_coord x (S (S (S O))))
                | S n2 ->
                  (match n2 with
                   | O -> F.mul (F.opp c3) (quat_coord x (S (S O)))
                   | S n3 ->
                     (match n3 with
                      | O -> F.mul c2 (quat_coord x (S O))
                      | S n4 ->
                        (match n4 with
                         | O -> quat_coord x O
                         | S _ -> F.zero))))
             | S _ -> F.zero)))

  (** val quat_param_metric_weight : F.t -> F.t -> nat -> F.t **)

  let quat_param_metric_weight c2 c3 = function
  | O -> F.one
  | S n ->
    (match n with
     | O -> F.opp c2
     | S n0 ->
       (match n0 with
        | O -> F.opp c3
        | S n1 -> (match n1 with
                   | O -> F.mul c2 c3
                   | S _ -> F.zero)))

  (** val quat_param_eq7_linear_form :
      F.t -> F.t -> nat -> coq_Quat -> F.t **)

  let quat_param_eq7_linear_form c2 c3 k x =
    F.add
      (F.add
        (F.add
          (F.mul (F.mul (quat_param_metric_weight c2 c3 O) (quat_coord x O))
            (quat_param_eq6_delta_coord c2 c3 k x O))
          (F.mul
            (F.mul (quat_param_metric_weight c2 c3 (S O))
              (quat_coord x (S O)))
            (quat_param_eq6_delta_coord c2 c3 k x (S O))))
        (F.mul
          (F.mul (quat_param_metric_weight c2 c3 (S (S O)))
            (quat_coord x (S (S O))))
          (quat_param_eq6_delta_coord c2 c3 k x (S (S O)))))
      (F.mul
        (F.mul (quat_param_metric_weight c2 c3 (S (S (S O))))
          (quat_coord x (S (S (S O)))))
        (quat_param_eq6_delta_coord c2 c3 k x (S (S (S O)))))

  (** val quat_param_near_identity_quadratic_factor :
      F.t -> F.t -> nat -> F.t **)

  let quat_param_near_identity_quadratic_factor c2 c3 = function
  | O -> F.zero
  | S n ->
    (match n with
     | O -> F.opp c2
     | S n0 ->
       (match n0 with
        | O -> F.opp c3
        | S n1 -> (match n1 with
                   | O -> F.mul c2 c3
                   | S _ -> F.zero)))
 end
