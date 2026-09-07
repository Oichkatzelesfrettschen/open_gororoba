
type nat =
| O
| S of nat

module Nat :
 sig
  val eqb : nat -> nat -> bool
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

  val quat_zero : coq_Quat

  val quat_one : coq_Quat

  val quat_add : coq_Quat -> coq_Quat -> coq_Quat

  val quat_scale : F.t -> coq_Quat -> coq_Quat

  val quat_neg : coq_Quat -> coq_Quat

  val qi : coq_Quat

  val qj : coq_Quat

  val qk : coq_Quat

  val quat_imag_basis : nat -> coq_Quat

  val quat_near_identity : nat -> F.t -> coq_Quat

  val quat_mul : coq_Quat -> coq_Quat -> coq_Quat

  val quat_conj : coq_Quat -> coq_Quat

  val embed_vec : coq_Vec3 -> coq_Quat

  val extract_vec : coq_Quat -> coq_Vec3

  val quat_rotate : coq_Quat -> coq_Vec3 -> coq_Vec3

  val quat_norm_sq : coq_Quat -> F.t

  val quat_coord : coq_Quat -> nat -> F.t

  val quat_identity_entry : nat -> nat -> F.t

  val quat_hamilton_basis_entry : nat -> nat -> nat -> F.t

  val quat_hamilton_entry : coq_Quat -> nat -> nat -> F.t

  val quat_hamilton_apply : coq_Quat -> coq_Quat -> coq_Quat

  val quat_matrix_transpose : (nat -> nat -> F.t) -> nat -> nat -> F.t

  val quat_matrix_mul_entry :
    (nat -> nat -> F.t) -> (nat -> nat -> F.t) -> nat -> nat -> F.t

  val quat_hamilton_gram_entry : coq_Quat -> nat -> nat -> F.t

  val quat_param_mul : F.t -> F.t -> coq_Quat -> coq_Quat -> coq_Quat

  val quat_param_norm : F.t -> F.t -> coq_Quat -> F.t

  val quat_param_form_entry : F.t -> F.t -> nat -> nat -> F.t

  val quat_param_matrix_entry : F.t -> F.t -> coq_Quat -> nat -> nat -> F.t

  val quat_param_matrix_apply : F.t -> F.t -> coq_Quat -> coq_Quat -> coq_Quat

  val quat_param_gram_entry : F.t -> F.t -> coq_Quat -> nat -> nat -> F.t

  val quat_param_eq6_delta_coord : F.t -> F.t -> nat -> coq_Quat -> nat -> F.t

  val quat_param_metric_weight : F.t -> F.t -> nat -> F.t

  val quat_param_eq7_linear_form : F.t -> F.t -> nat -> coq_Quat -> F.t

  val quat_param_near_identity_quadratic_factor : F.t -> F.t -> nat -> F.t
 end
