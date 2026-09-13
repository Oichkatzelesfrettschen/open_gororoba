---
description: Physical-model, calibration, and mechanism-discrimination contract for Casimir and optical experiments
last_verified: 2026-09-12
evidence_class: implementation_and_claim_scope_contract
---

# Casimir, optics, and calibrated discrimination

This contract separates three questions that a scientific workflow must answer:

1. Does the forward model implement its stated mathematics?
2. Do independent measurements constrain the specimen and nuisance state?
3. Do physically admissible alternatives remain distinguishable at a declared effect size?

A positive answer at one layer never supplies a positive answer at another.

## Observation law and typed parameters

The shared statistical object is an observation law

```text
p(y, z | theta, alpha, eta, u, H)
```

where `H` is a physical hypothesis, `theta` is the tested effect amplitude,
`alpha` is material or device state, `eta` contains instrument and nuisance
parameters, `u` is an admissible intervention, `y` is the target record, and
`z` contains independent calibration and witness records.

Material state and effect amplitude are different coordinates. An optical
measurement can establish a change in `alpha`; it does not establish that a
force change arose from a proposed `theta`-dependent mechanism.

Shared projection, response, or Green-function mathematics does not identify
shared physical mechanisms. Lifshitz forces are conventional electromagnetic
physics. A layer-order contrast tests a specified material and geometry model;
nonzero contrast alone does not establish new physics.

## Forward-model contracts

### Planar zero-temperature Lifshitz interaction

For planar media,

```text
E/A = hbar/(4*pi^2) integral dxi integral k_parallel dk_parallel
      sum_p log(1 - Q_p)

P = -hbar/(2*pi^2) integral dxi integral k_parallel*kappa dk_parallel
    sum_p Q_p/(1 - Q_p)

Q_p = r_1p*r_2p*exp(-2*kappa*d)
```

`quantum_core::casimir::lifshitz` evaluates the full nonnegative angular domain
with dimensionless polar coordinates. Ideal-mirror pressure, ideal planar
energy, the pressure-energy derivative, and sphere-plane PFA are independent
tests. Increasing point count cannot repair a truncated angular domain, so the
legacy `k_parallel=(xi/c)*v`, `v<=10` construction is not an accepted oracle.

The sphere-plane PFA conversion is

```text
F_PFA = 2*pi*R*(E/A) = 2*pi*R*integral_d^infinity P(z) dz.
```

For ideal mirrors, `P` scales as `d^-4`, so the pressure-only expression is
`2*pi*R*d*P(d)/3`, not `/2`.

### Finite temperature

The planar free-energy prefactor is `k_B*T/(2*pi)` and the zero Matsubara mode
has half weight. Under the local Drude prescription a continuous conducting
outer cap has `r_TM(0,k)=1` and `r_TE(0,k)=0`. Identical outer caps therefore
cancel the zero-frequency term in their differential channel.

That cancellation is conditional on the local Drude model, cap continuity,
common outer response, and the differential observable. It does not remove
finite-frequency thermal response, film-continuity uncertainty, electrostatic
patch effects, or fabrication differences. The cap both screens buried-layer
contrast and cancels a leading thermal common mode; cap thickness is therefore
a design tradeoff rather than a monotone rule.

### Passive optics and specimen geometry

The real-frequency convention is `exp(-i*omega*t)`:

```text
epsilon(omega) = epsilon_inf
  - omega_p^2/[omega*(omega+i*gamma_D)]
  + sum_j S_j*omega_j^2/[omega_j^2-omega^2-i*gamma_j*omega].
```

Its direct imaginary-axis continuation has positive damping terms in every
denominator. Taking `abs(Im(sqrt(epsilon)))` can hide a branch or sign error;
it cannot restore causality or analytic continuation. Tests therefore inspect
the complex dielectric function before derived positive observables.

An optical state witness must describe the same film thickness, substrate,
interfaces, constitutive model, and state history as the force specimen. A
half-space reflectivity is not a finite-film witness. WO3 and oxygen-deficient
WO3 endpoint tables do not establish reversible switching of one specimen;
composition, ion insertion, trapping, thermal history, morphology, hysteresis,
and electrostatics remain separate measured coordinates.

Constant nonunit `epsilon_inf` values are effective finite-band backgrounds,
not complete all-frequency dielectric models. UV completions are model
sensitivity scenarios unless a measured uncertainty distribution supports
probabilistic language.

`DrudeLorentzParams::at_temperature` applies an explicit coth oscillator
scaling and `gamma(T)=gamma(0)*(1+(T/T_D)^2)` Drude scaling. The second
expression is a quadratic Debye-scaled heuristic, not a Bloch-Gruneisen
transport integral. Its tests establish arithmetic behavior only; physical use
requires specimen-specific temperature-response calibration.

## Mathematical imitation versus physical imitation

The unrestricted residual

```text
r = (I - P_N)t
```

answers how much of target vector `t` lies outside the linear span of nuisance
directions `N`. It permits arbitrary coefficients and tests only the supplied
local tangent model. A tiny residual establishes mathematical imitation, not a
calibrated physical explanation.

For a local Gaussian target and independent calibration model,

```text
y = theta*t + N*eta + epsilon_y
z = K*eta + epsilon_z

J = t^T Sigma_y^-1 t
v = N^T Sigma_y^-1 t
M = N^T Sigma_y^-1 N + K^T Sigma_z^-1 K
I_eff = J - v^T M^-1 v.
```

Implementations solve the whitened augmented least-squares problem rather than
subtracting nearly equal quadratic forms. A Gaussian prior can add precision,
but registry and report text must call it a prior, not calibration data.

If calibration adds precision `p_j` to nuisance coordinate `j`,

```text
d I_eff / d p_j = (e_j^T M^-1 v)^2 >= 0.
```

The derivative ranks calibration work for the declared protocol, covariance,
and assumed widths. It does not establish a universal ordering across devices.

For nonlinear bounded alternatives, use the global profile distance

```text
D(u) = inf || Sigma^-1/2 [mu_1(u,eta_1)-mu_0(u,eta_0)] ||,
       eta_0 in C_0, eta_1 in C_1.
```

The sets `C_i` carry units, positive-thickness constraints, calibration ranges,
state history, and other physical restrictions. A local fit that requires
negative layers or shifts large enough to invalidate its derivatives is a
numerical imitation only.

Minimax design over arbitrary distinct continuous parameter values has zero
infimum. A design must declare a physically meaningful minimum separation such
as `|theta_1-theta_0| >= theta_min` and an acquisition/resource budget.

If the two whitened forward predictions have bounded numerical and model errors
`epsilon_0` and `epsilon_1`, the certified separation is

```text
D_true >= max(0, D_computed - epsilon_0 - epsilon_1).
```

Reported solver precision cannot replace constitutive-model accuracy.

## Covariance and resource accounting

A derivative computed from an existing record is a deterministic map:

```text
y_aug = A*y,  A = [I; D],  Sigma_aug = A*Sigma*A^T.
```

The joint covariance includes cross-covariance and is rank deficient. A
pseudoinverse-aware Fisher calculation preserves the original information. A
separately measured gradient can add information only through an independent
physical response and its measured joint noise.

Active/sham and null channels consume acquisition time and carry noise. Compare
protocols at fixed total resource. Exact nuisance identities that survive a
shared derivative operator remain identities after augmentation.

## Mechanism-specific scope

An ordinary classical noncommuting loop can produce an exact orientation-odd
`q*a*b` term. The term isolates a declared interaction sector only after
ordinary noncommuting dynamics enter the alternative set. Symmetry and
noncommutativity alone do not identify a quantum or exotic mechanism.

Quantum memory kernels require a separate admissibility gate. Positive scalar
exponential weights do not imply a completely positive trace-preserving map.
`quantum_core::channel_admissibility` tests the qubit depolarizing Choi
eigenvalue and retains a positive-kernel counterexample with a negative Choi
eigenvalue. A deployable auxiliary-memory approximation needs a CPTP dilation,
valid subordination construction, or explicit channel tests.

For finite-dimensional Markovian metrology, the Hamiltonian-not-in-Lindblad-
span theorem assumes noiseless ancillas and arbitrarily fast, accurate control.
Under those assumptions, a signal Hamiltonian outside the Lindblad span permits
QEC-assisted Heisenberg scaling. This is an operator-space theorem, not an
identity between classical regression and QEC. Kwon et al. (2026) give further
sufficient conditions and finite-time error scaling for autonomous correction
at finite engineered-dissipation ratio; citing the ideal theorem alone does not
validate a finite-rate controller.

## Condition-bound materials

Common names and formulae identify material families, not specimens. The typed
evidence graph is

```text
Material -> MaterialState -> Specimen -> Measurement -> QuantityValue
                                      \-> ModelRun -> DerivedValue
```

Every observed quantity carries a unit, representation and basis, conditions,
applicability range, uncertainty or covariance, evidence basis, method, raw
artifact identity, source locator, source-byte hash, parser version, and
transformation lineage. Direct experiments, fitted experimental parameters,
computed outputs, and inferred proxies remain distinct admission classes.
State-overlapping quantity conditions use explicit keys such as
`temperature_k`, `pressure_pa`, and zero-based `strain_component:<index>`;
graph validation resolves the measurement through its specimen and rejects
values that contradict the typed `MaterialState`. Ambiguous aliases such as
`temperature`, `pressure`, and `strain` are not state bindings.

Missing values use `not_measured`, `below_detection_limit`, `not_applicable`,
`withheld`, or `unknown`. Numeric zero and empty text never encode missingness
in the typed graph. The legacy `MineralMetadata` surface is a catalog
compatibility type; its quarantine conversion maps historical sentinels to
typed absence and cannot satisfy direct-measurement admission.

`get_material_model("gold")` selects one scalar 300 K optical model. The legacy
`get_material("gold")` alias has the same model-selection semantics. Neither API
selects a physical gold specimen. Johnson-Christy, Olmon evaporated gold,
McPeak template-stripped gold, heated films, thickness series, and nanoparticle
records remain separate datasets and specimens.

## Claim and experiment promotion

The promotion chain is:

```text
admissible physical model
  -> verified forward calculation
  -> independent calibration and covariance
  -> bounded mechanism comparison at theta_min
  -> held-out prediction
  -> claim with explicit scope.
```

Each transition records the executed command, code revision, source and output
hashes, covariance, model-error margin, falsifier, and unresolved residuals.
Native Rust conformance does not establish a laboratory effect. Source
admission does not establish implementation conformance. A bounded forecast
under hypothetical widths is not measured apparatus performance.

## Native retained replay

The deterministic native producer writes `summary.toml`,
`native-output-manifest.toml`, and the typed TSVs in
`data/output/audit/casimir-optics-discrimination/`. The manifest binds every
derived output to the producer, model sources, workspace and crate manifests,
resolved dependency lockfile, and pinned Rust toolchain. The retained values
include:

- `-0.4709381888912518 Pa` for the 20 nm Au capped
  `SiO2(50 nm)/Al2O3(50 nm)/Si` stack at a 200 nm gap;
- `2.7242957e-8` relative pressure-energy derivative disagreement for that
  stack;
- exactly zero differential zero-mode pressure for the common-cap pair at the
  tested gaps under the local Drude prescription;
- `0.0065246449` maximum relative contrast change across the declared
  hypothetical UV completions;
- `5.0049074e-8` unrestricted and `0.67018369` one-width bounded nuisance
  residual fractions for the 13-gap, 10 nm cap scenario; and
- `51.485234` baseline effective information, with halving the full
  differential-gap width producing the largest tested information gain at
  `116.5914 percent`.

Those values are model outputs under declared optical parameters and scenario
widths. They are neither measurements nor uncertainty distributions.

The canonical artifact path is `native-output-manifest.toml`.
`derived-output-manifest.toml` remains a historical generation because the
first typed path-repair event admitted its exact bytes before Clippy exposed a
producer-local parameter-shape defect. The second typed repair advances the
canonical path to the lint-clean producer and retains the earlier manifest;
neither append-only event is rewritten.

The source branch `chat-2026-09-10-identifiability-next` at
`0d3278e47d433b57d6423efb85e3ab880c2e136c` contains an executable 96-row
full-factorial commutator schedule. Its exact nuisance residual fraction is
`1.0`; the native 100000-trial replay retains its seed, critical value, false
positive fraction, simulated power, and analytic power. The separately
reported residual fraction `0.9953610602` and associated Monte Carlo values
refer to `design_commutator.csv`. That CSV is absent from the repository, all
searched Git refs, and the retained session paths searched during the audit.
The audit records those reported numbers as an unreplayed observation with
`resource_scope=missing_design_commutator_csv`; it does not synthesize a
schedule to reproduce them.

The declared frontier therefore has 30 rows, 28 closed rows, and two open
rows. Hosted cutoff and quadrature-order refinement closes
`lifshitz-planar-polar-normalization`; `planar_convergence.tsv` retains the
independent cutoff, radial-order, and angular-order observations.
`selected-schedule-independent-monte-carlo` remains open until the exact CSV is
retained and hashed. `declared-frontier-denominator-proof` remains open because
it depends on that mechanism row. The finite-frontier verifier proves this
exact partition and rejects denominator, witness, dependency, and state
mutations; it does not execute every row's scientific evidence.

## Retained sources

The byte-level manifest is
`data/output/audit/casimir-optics-discrimination/source-retrieval-manifest.toml`.
Load-bearing sources include Lifshitz (1956), the Casimir review by Klimchitskaya
et al. (2009), Zhou et al. on HNLS (2017), Breuer and Vacchini on CP memory
kernels (2009), Kwon et al. on finite-rate AutoQEC metrology (2026), and McPeak
et al. on template-stripped plasmonic films (2015). The manifest also retains a
misidentified 2008 tutorial as a non-supporting provenance witness.

The pinned refractiveindex.info records are CC0 digitizations. Their underlying
papers remain the experimental authority. A digitized table can establish its
stored values and cited specimen label; it cannot supply method or uncertainty
metadata absent from the retained source.
