
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

module QuatOps :
 functor (F:FLOAT_OPS) ->
 sig
  type coq_Quat = { qw : F.t; qx : F.t; qy : F.t; qz : F.t }

  val qw : coq_Quat -> F.t

  val qx : coq_Quat -> F.t

  val qy : coq_Quat -> F.t

  val qz : coq_Quat -> F.t

  type coq_Vec3 = { vx : F.t; vy : F.t; vz : F.t }

  val vx : coq_Vec3 -> F.t

  val vy : coq_Vec3 -> F.t

  val vz : coq_Vec3 -> F.t

  val quat_mul : coq_Quat -> coq_Quat -> coq_Quat

  val quat_conj : coq_Quat -> coq_Quat

  val embed_vec : coq_Vec3 -> coq_Quat

  val extract_vec : coq_Quat -> coq_Vec3

  val quat_rotate : coq_Quat -> coq_Vec3 -> coq_Vec3

  val quat_norm_sq : coq_Quat -> F.t
 end
