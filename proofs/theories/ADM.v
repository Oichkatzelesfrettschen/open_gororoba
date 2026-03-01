(** * ADM 3+1 decomposition: constraint structure.

    Mirrors the Hamiltonian constraint from adm.rs:330-355:
    H = R^(3) + K^2 - K_{ij}K^{ij} - 16*pi*rho

    We model the constraint as a function of scalar quantities
    (the actual tensor contractions are done in Rust). *)

From OpenGororoba Require Import Prelude.

(** Hamiltonian constraint residual.
    H = R3 + K^2 - K_{ij}K^{ij} - 16*pi*rho

    For valid initial data, H = 0 on each spatial slice. *)
Definition hamiltonian_constraint (R3 K_sq KijKij rho : R) : R :=
  R3 + K_sq - KijKij - sixteen_pi * rho.

(** Painleve-Gullstrand lapse squared.

    In PG coordinates: g_{00} = -(1 - 2M/r), g_{0r} = sqrt(2M/r), g_{rr} = 1.
    alpha^2 = beta^r * beta_r - g_{00}
            = (sqrt(2M/r))^2 + (1 - 2M/r)
            = 2M/r + 1 - 2M/r = 1.

    Mirrors: PainleveGullstrand in adm.rs:527-573, test at line 710-721. *)
Definition pg_lapse_sq (M r : R) : R :=
  (sqrt (2 * M / r)) * (sqrt (2 * M / r)) + (1 - 2 * M / r).

(** PG shift vector: beta^r = sqrt(2M/r). *)
Definition pg_shift_r (M r : R) : R := sqrt (2 * M / r).

(** PG spatial metric: gamma_{rr} = 1 (flat in radial direction). *)
Definition pg_gamma_rr : R := 1.
