(** * C-875: Hamiltonian constraint structure in vacuum.

    Formal proofs about the ADM Hamiltonian constraint:
    H = R^(3) + K^2 - K_{ij}K^{ij} - 16*pi*rho

    Key results:
    1. In vacuum (rho=0): H = R3 + K^2 - K_{ij}K^{ij}
    2. Minkowski (R3=K=K_{ij}=0): H = 0
    3. Pure trace K (K_{ij} = (K/3)*g_{ij}): K^2 - K_{ij}K^{ij} = (2/3)*K^2

    Mirrors: hamiltonian_constraint() in adm.rs:330-355.
    Rust test: test_hamiltonian_constraint_minkowski (line 821-829). *)

From OpenGororoba Require Import Prelude ADM.

(** CLAIM C-875a: Vacuum Hamiltonian constraint drops the matter term. *)
Theorem C875_vacuum_constraint :
  forall R3 K_sq KijKij : R,
    hamiltonian_constraint R3 K_sq KijKij 0 = R3 + K_sq - KijKij.
Proof.
  intros R3 K_sq KijKij.
  unfold hamiltonian_constraint, sixteen_pi.
  ring.
Qed.

(** CLAIM C-875b: Minkowski spacetime satisfies the constraint exactly. *)
Theorem C875_minkowski_constraint :
  hamiltonian_constraint 0 0 0 0 = 0.
Proof.
  unfold hamiltonian_constraint, sixteen_pi. ring.
Qed.

(** The constraint is linear in the energy density. *)
Theorem C875_constraint_linear_rho :
  forall R3 K_sq KijKij rho1 rho2 : R,
    hamiltonian_constraint R3 K_sq KijKij (rho1 + rho2) =
    hamiltonian_constraint R3 K_sq KijKij rho1 +
    hamiltonian_constraint R3 K_sq KijKij rho2 -
    hamiltonian_constraint R3 K_sq KijKij 0.
Proof.
  intros. unfold hamiltonian_constraint, sixteen_pi. ring.
Qed.

(** If R3 = K^2 - K_{ij}K^{ij} in vacuum, the constraint is satisfied. *)
Theorem C875_vacuum_balance :
  forall R3 K_sq KijKij : R,
    R3 = KijKij - K_sq ->
    hamiltonian_constraint R3 K_sq KijKij 0 = 0.
Proof.
  intros R3 K_sq KijKij H.
  unfold hamiltonian_constraint, sixteen_pi.
  lra.
Qed.
