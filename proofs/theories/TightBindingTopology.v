(** * TightBindingTopology: Definitions for lattice band topology.

    Defines the mathematical objects underlying the FHS (Fukui-Hatsugai-
    Suzuki) discretised Berry curvature algorithm:
    - U(1) link variable on a BZ plaquette
    - Plaquette Berry phase
    - Chern number as BZ sum of Berry phases
    - Valley Chern number as half-BZ sum
    - Flat band criterion (bandwidth < epsilon)

    These definitions are purely axiomatic (no computation on R);
    proofs using them reason via real analysis (lra, nra, field).

    Mirrors: crates/quantum_core/src/tight_binding.rs
    Claims: C-1233, C-1234, C-1236 *)

From OpenGororoba Require Import Prelude.

Open Scope R_scope.

(* ------------------------------------------------------------------ *)
(** * Berry curvature and time-reversal symmetry.                      *)
(* ------------------------------------------------------------------ *)

(** Berry curvature F(k) at a point in the discretised BZ.
    In the FHS algorithm this is Im(ln(U_12 * U_23 * U_34 * U_41))
    where U_ij are link variables between neighbouring k-points. *)
Definition berry_curvature := R -> R -> R.

(** Time-reversal symmetry (TRS) constraint on Berry curvature:
    F(kx, ky) = -F(-kx, -ky).
    This holds whenever the Hamiltonian satisfies H(k) = H(-k)^*. *)
Definition has_trs (F : berry_curvature) : Prop :=
  forall kx ky, F kx ky = - F (-kx) (-ky).

(** Chern number: integral of Berry curvature over the full BZ.
    Discretised as (1/2pi) * sum_{plaquettes} F(k).
    We model this as a real-valued function of a Berry curvature field.
    The actual summation is performed in Rust; here we axiomatise the
    integral as a linear functional. *)
Parameter chern_integral : berry_curvature -> R.

(** Linearity of the Chern integral (used to split BZ sums). *)
Axiom chern_integral_linear :
  forall (F G : berry_curvature),
    (forall kx ky, F kx ky + G kx ky = 0) ->
    chern_integral F + chern_integral G = 0.

(** Splitting axiom: the full BZ integral equals the sum of two
    half-BZ integrals (valley K and valley K'). *)
Parameter valley_integral_K  : berry_curvature -> R.
Parameter valley_integral_Kp : berry_curvature -> R.

Axiom chern_splits_into_valleys :
  forall F : berry_curvature,
    chern_integral F = valley_integral_K F + valley_integral_Kp F.

(** TRS maps the K valley to K' valley with sign flip. *)
Axiom trs_swaps_valleys :
  forall F : berry_curvature,
    has_trs F ->
    valley_integral_Kp F = - valley_integral_K F.

(* ------------------------------------------------------------------ *)
(** * Flat band definitions.                                           *)
(* ------------------------------------------------------------------ *)

(** Band energy as a function of k. *)
Definition band_dispersion := R -> R -> R.

(** Bandwidth: max - min over the BZ. *)
Parameter bandwidth : band_dispersion -> R.

(** A band is flat when its bandwidth is below a threshold. *)
Definition is_flat_band (E : band_dispersion) (eps : R) : Prop :=
  0 < eps /\ bandwidth E < eps.

(** Group velocity magnitude bound: if the band is sampled on a grid
    with spacing dk, then the discrete gradient |dE/dk| is bounded
    by bandwidth / dk at every k-point. *)
Parameter max_group_velocity : band_dispersion -> R -> R.

Axiom group_velocity_bandwidth_bound :
  forall (E : band_dispersion) (dk : R),
    0 < dk ->
    max_group_velocity E dk <= bandwidth E / dk.
