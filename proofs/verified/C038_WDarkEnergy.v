(** * C-038: w = -5/6 is a tuned point in parameter space (refuted).

    C-038 claimed w = -5/6 as a natural prediction. The positive content:
    under the DarkEnergyEOS model, w = -5/6 requires a specific tuning
    of beta * ds = 1/6. This is not generic but a measure-zero constraint.

    Kernel-checked via DarkEnergyEOS.v. *)

From OpenGororoba Require Import Prelude DarkEnergyEOS.

(** w = -5/6 iff beta * ds = 1/6. *)
Theorem C038_w_tuning : forall beta ds : R,
  w_from_ds beta ds = -5/6 <-> beta * ds = 1/6.
Proof.
  intros beta ds. unfold w_from_ds. split; intros H; lra.
Qed.

(** The LCDM limit is w = -1 (beta = 0), far from -5/6. *)
Theorem C038_lcdm_not_five_sixths :
  w_from_ds 0 0 <> -5/6.
Proof.
  unfold w_from_ds. lra.
Qed.
