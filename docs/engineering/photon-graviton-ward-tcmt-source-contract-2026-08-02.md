---
description: source-faithful Ward and photon-graviton TCMT contract for the oracle quarantine
last_verified: 2026-08-02
evidence_class: source-contract
---

# Photon-graviton Ward and TCMT source contract

This contract separates source identities from the current scalarized
implementation. It is the boundary for later tensor Ward and scattering work.
The retained machine-readable inventory is
`data/output/audit/2026-08-02/source_inventory.toml`.

## Source inventory

| Source | Role | Local surface |
| --- | --- | --- |
| Bastianelli and Schubert, gr-qc/0412095 | Worldline one-loop photon-graviton amplitude | `crates/gr_core/src/photon_graviton` |
| Bastianelli et al., 0710.5572 | Worldline polarization structure | `crates/gr_core/src/photon_graviton` |
| Ahmadiniaz et al., 2601.23279 | Three diagrams, tadpole, and Ward source | `crates/gr_core/src/photon_graviton` |
| Ruan and Fan, 0909.3323 | Cylindrical channel TCMT and Fano scattering | `crates/optics_core/src/fano_tcmt.rs` |
| Maksimov et al., 2505.00396 | Symmetry constraints on TCMT parameters | `crates/gr_core/src/photon_graviton_tcmt` |

The registry records the source entries in
`registry/source_lanes/web_references.toml`. The source URLs are
`https://arxiv.org/abs/gr-qc/0412095`,
`https://arxiv.org/abs/0710.5572`,
`https://arxiv.org/abs/2601.23279`,
`https://arxiv.org/abs/0909.3323`, and
`https://arxiv.org/abs/2505.00396`.

## Ward identities

The electromagnetic gauge Ward identity is a per-diagram statement:

```text
k_alpha * Gamma_d^{mn,alpha} = 0
```

The diagram index `d` ranges over irreducible, tadpole, and external-leg
contributions. A valid implementation must retain the tensor contraction
components before reducing them to a scalar norm. A scalar output of exactly
zero does not establish the identity when the contraction was multiplied by a
hardcoded on-shell factor.

The nontrivial off-shell relation is gravitational. The contracted amplitude
must be compared with a declared lower-point right-hand side. The on-shell
relations used by the source treatment are:

```text
Gamma_irr(delta_eps_0) + Gamma_ext(delta_eps_0) = 0
Gamma_tadpole(delta_eps_0) = 0
```

The current `ward.rs` path relabels the irreducible gauge residual as the full
gravitational residual. The current tadpole and external checks return asserted
scalar zeros. These paths are retained as negative controls and are not valid
positive evidence for C-820 or C-822.

The P1 implementation must declare all of the following before numerical
comparison:

| Convention | Required declaration |
| --- | --- |
| Metric | Signature and index raising/lowering rule |
| Continuation | Euclidean to Minkowski prescription, if used |
| Momentum | Components, conservation equation, and incoming/outgoing signs |
| Shell mode | On-shell and off-shell constraints, including `k_squared` |
| Polarizations | Photon, graviton, and diffeomorphism-vector basis |
| Renormalization | Bare or renormalized contribution for each diagram |
| Norm | Absolute component norm, normalized norm, and conditioning scale |

The source contract does not supply missing tensor components by inference from
the scalar implementation. Missing components are an implementation boundary,
not a reason to promote a surrogate.

The intended P1 result records are:

```rust
type ComplexFourVector = SVector<Complex64, 4>;
type ComplexLorentzMatrix = SMatrix<Complex64, 4, 4>;

struct GaugeWardResidual {
    diagram: Diagram,
    shell_mode: ShellMode,
    renormalization_state: RenormalizationState,
    contracted_components: ComplexLorentzMatrix,
    conditioning_scale: f64,
    absolute_norm: f64,
    normalized_norm: f64,
    tolerance: ResidualTolerance,
}

struct GravitationalWardResidual {
    shell_mode: ShellMode,
    renormalization_state: RenormalizationState,
    lhs_components: ComplexFourVector,
    one_photon_rhs_components: ComplexFourVector,
    two_photon_rhs_components: ComplexFourVector,
    lower_point_rhs_components: ComplexFourVector,
    defect_components: ComplexFourVector,
    conditioning_scale: f64,
    absolute_defect: f64,
    normalized_defect: f64,
    tolerance: ResidualTolerance,
}
```

Contract the photon index into the rank-two `ComplexLorentzMatrix` and the
graviton indices into the rank-one `ComplexFourVector`. Retain every matrix or
vector component before computing a Frobenius or Euclidean norm. Compare the
off-shell gravitational contraction with its lower-point right-hand side; do
not require either side to be nonzero. The fixed-rank implementations live in
`crates/gr_core/src/photon_graviton/tensor_types.rs` and are exercised by the
source-owned tensor paths.

## Scattering identities

Ruan-Fan's cylindrical formulation uses a radial reflection coefficient and an
angular-momentum channel coefficient:

```text
R_l = h_l_minus / h_l_plus
S_l = (R_l - 1) / 2
```

The lossless radial-channel condition is `abs(R_l) = 1`. This object is not a
generic multiport scattering matrix. A multiport model requires a declared port
basis, flux normalization, and a complex matrix `S`.

For a flux-normalized multiport model, the conditions are distinct:

| Property | Condition |
| --- | --- |
| Lossless unitarity | `S dagger * S = I` |
| Reciprocity | A port-basis relation declared independently of unitarity |
| Time reversal | A symmetry relation declared independently of loss |
| Passive loss | `S dagger * S <= I` |
| Absorption balance | Incoming flux equals outgoing flux plus explicit internal absorption |

The channel cross-section identities must use an independently defined
extinction observable:

```text
C_sct = factor * sum_l abs(S_l)^2
C_abs = -factor * sum_l (Re(S_l) + abs(S_l)^2)
C_ext = -factor * sum_l Re(S_l)
```

Defining extinction as scattering plus absorption can be a useful bookkeeping
identity, but it is not an optical-theorem verification. The current
`CrossSections::new` path calculates the ratio from the same two inputs and is
therefore a retained tautology control.

## TCMT mapping boundary

The current photon-graviton mapping is a phenomenological hypothesis until a
source-derived port normalization and an independent complex-amplitude fit are
available. The implementation currently:

| Quantity | Current construction | Contract status |
| --- | --- | --- |
| Coupling strength | Scalar amplitude magnitude times two declared scalars, floored at `1e-20` | Underidentified |
| Radiative decay | Twice the absolute tadpole imaginary part, floored at `1e-20` | Port normalization absent |
| External phase | Modulo of a real scalar | Complex phase absent |
| Nonradiative decay | `0.1 * gamma_radiative` | Hardcoded assumption |
| Gravitational decay | `0.01 * gamma_radiative` | Hardcoded assumption |
| Weak-field parameter | Coupling divided by total decay and resonance frequency | Units and zero-decay behavior require review |

C-864 through C-867 are one dependency closure. The model cannot be made
source-faithful by replacing a single scalar complement while the amplitude map,
loss model, and port semantics remain unspecified.

The next experiment must fit declared parameters on a training subset and
compare held-out complex channel amplitudes. It must report lossless unitarity,
reciprocity, time-reversal, passive contractivity, and absorption balance as
separate predicates.

## Promotion boundary

The source-level propositions are not refuted by invalid numerical oracles.
Implementation-conformance claims remain provisional until the tensor and
complex-channel observables are represented independently of the legacy scalar
reductions.
