(** * C-030: The sedenion associator is nonzero.

    The octonion associator [e1, e2, e4] <> 0 (proved in OctonionNorm).
    Since any octonion triple embeds into the sedenion lo-half,
    the sedenion associator is also nonzero.

    This is a structural obstruction: no associative action principle
    can be formulated directly on the sedenions without modification. *)

From OpenGororoba Require Import Prelude CayleyDicksonAlgebra Sedenion OctonionNorm.

(** Embed the octonion witness into sedenion lo-half. *)
Definition sed_e1 : CDSed := mkSed (oct_e 1) oct_zero.
Definition sed_e2 : CDSed := mkSed (oct_e 2) oct_zero.
Definition sed_e4 : CDSed := mkSed (oct_e 4) oct_zero.

(** Sedenion associator via the CD product. *)
Definition sed_assoc (a b c : CDSed) : CDSed :=
  sed_sub (sed_mul (sed_mul a b) c) (sed_mul a (sed_mul b c)).

(** The sedenion associator is nonzero for the embedded octonion triple.
    Strategy: extract the qd of oct_hi of sed_lo, which carries the
    same nonzero component as the octonion associator. *)
Theorem C030_sed_assoc_nonzero :
  sed_assoc sed_e1 sed_e2 sed_e4 <> sed_zero.
Proof.
  intro H.
  assert (Hlo := f_equal sed_lo H).
  assert (Hhi := f_equal oct_hi Hlo).
  assert (Hd := f_equal qd Hhi).
  cbv [sed_assoc sed_sub sed_add sed_neg sed_mul sed_e1 sed_e2 sed_e4
       oct_mul oct_conj oct_add oct_neg oct_e oct_zero
       quat_mul quat_add quat_neg quat_conj quat_zero quat_one
       sed_lo sed_hi oct_lo oct_hi oct_zero quat_zero
       qa qb qc qd sed_zero] in Hd.
  lra.
Qed.
