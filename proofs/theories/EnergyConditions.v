(** * EnergyConditions: classical energy condition definitions and implications.

    Defines WEC, NEC, SEC, DEC as propositions on (rho, p) and proves
    the classical implication chain: DEC => WEC => NEC, SEC => NEC.

    Mirrors: gr_core/src/quantum_inequalities.rs lines 238-269 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(** Weak Energy Condition: rho >= 0 AND rho + p >= 0. *)
Definition WEC (rho p : R) : Prop := rho >= 0 /\ rho + p >= 0.

(** Null Energy Condition: rho + p >= 0. *)
Definition NEC (rho p : R) : Prop := rho + p >= 0.

(** Strong Energy Condition: rho + 3*p >= 0 AND rho + p >= 0. *)
Definition SEC (rho p : R) : Prop := rho + 3 * p >= 0 /\ rho + p >= 0.

(** Dominant Energy Condition: rho >= 0 AND rho >= |p| AND rho + p >= 0. *)
Definition DEC (rho p : R) : Prop := rho >= 0 /\ Rabs p <= rho /\ rho + p >= 0.

(** WEC implies NEC. *)
Theorem WEC_implies_NEC : forall rho p, WEC rho p -> NEC rho p.
Proof. intros rho p [_ H]. exact H. Qed.

(** SEC implies NEC. *)
Theorem SEC_implies_NEC : forall rho p, SEC rho p -> NEC rho p.
Proof. intros rho p [_ H]. exact H. Qed.

(** DEC implies WEC. *)
Theorem DEC_implies_WEC : forall rho p, DEC rho p -> WEC rho p.
Proof. intros rho p [Hrho [_ Hsum]]. split; assumption. Qed.

(** DEC implies NEC (by transitivity). *)
Theorem DEC_implies_NEC : forall rho p, DEC rho p -> NEC rho p.
Proof. intros. apply WEC_implies_NEC. apply DEC_implies_WEC. assumption. Qed.
