
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
 end
