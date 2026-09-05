# Magnonic isolated bundles and time reversal

The source-fitted tight-binding model uses real on-site energies, real hopping
amplitudes and integer cell offsets. A sampled Chern sum cannot establish that
a spectral bundle exists over the whole Brillouin torus. Global external-gap
admission supplies that missing premise; internal degeneracies remain compatible
with a bundle invariant.

## Mathematical statement

Let `q` belong to the two-dimensional unit torus. For a finite orbital basis,
define the exact Hermitian matrix

```text
H(q) = D + sum_h [t_h exp(2 pi i n_h.q) E_(a_h,b_h)
               + t_h exp(-2 pi i n_h.q) E_(b_h,a_h)].
```

Assume that `D` and every `t_h` are real and every `n_h` is an integer pair.
Assume that a fixed contiguous set of ordered eigenvalues has a positive gap
from its complement at every `q`. The associated spectral projector `P(q)`
is smooth and periodic, even when eigenvalues inside the set coincide.

Complex conjugation gives `H(-q) = conjugate(H(q))`. Ordered eigenvalues
coincide at `q` and `-q`, so the same spectral set obeys
`P(-q) = conjugate(P(q))`. Differentiation reverses each coordinate derivative.
Consequently,

```text
A(q) = Tr(P(q) [d_1 P(q), d_2 P(q)])
A(-q) = conjugate(A(q)) = -A(q).
```

The last equality follows because the projector and its derivatives are
Hermitian and the commutator is anti-Hermitian; cyclicity of the trace makes
`A` purely imaginary. The map `q -> -q` preserves orientation in dimension two.
Therefore the integral of `A` over the torus vanishes, and

```text
c_1(P) = (1 / (2 pi i)) integral_T2 A(q) dq_1 dq_2 = 0.
```

The argument applies to each globally isolated, time-reversal-preserved bundle.
An individual band touching a neighbor lacks the required rank-one projector
and does not acquire an individual Chern number from a small sampled sum.
A valley patch is a different integration domain; its curvature integral need
not vanish or be quantized.

## Cell-cover admission

For a square cell centered at `q_0` with coordinate half-width `r`,
`|exp(i x) - exp(i y)| <= |x-y|` bounds each hopping contribution. Summing
absolute contributions by row, including both Hermitian directions, gives

```text
S = max_row sum_incident_h |t_h| (|n_h,1| + |n_h,2|)
||H(q) - H(q_0)||_2 <= 2 pi r S.
```

For a Hermitian matrix the maximum absolute row sum bounds its spectral norm.
Weyl's inequality then bounds the reduction of an adjacent eigenvalue gap by
`4 pi r S`. For a uniform `N` by `N` centered torus cover, `r = 1/(2N)`.
The corresponding gap reduction is `2 pi S/N`.

A numerical certificate must additionally enclose the eigenvalues at every
cell center. If center eigenvalues have absolute errors at most `epsilon`, a
center gap `g` has the cellwise lower bound

```text
g_lower = g - 2 epsilon - 4 pi r S.
```

Every arithmetic operation used to obtain a lower bound must preserve its
direction. A floating-point eigensolver's returned values alone supply neither
`epsilon` nor a certificate. Residual bounds must also account for basis
orthogonality and Hamiltonian evaluation error. An unexamined library
trigonometric error assumption leaves the certificate conditional.

For a full square candidate eigenvector matrix `Q` and real diagonal `D`, let
directed enclosures establish

```text
eta >= ||Q* Q - I||_2,        eta < 1,
rho >= ||Q* H Q - D||_2.
```

Set `G = Q* Q`, `B = G^(-1/2)` and `U = Q B`. The matrix `U` is unitary.
The exact identity

```text
U* H U - D = B (Q* H Q - D) B + (B D B - D)
```

and the bounds `||B||^2 <= 1/(1-eta)` and
`||B-I|| (||B||+1) <= eta/(1-eta)` yield

```text
epsilon = (rho + ||D||_2 eta) / (1-eta).
```

Weyl's inequality applies to the sorted diagonal entries of `D`. A partial
Ritz basis does not establish the same full-spectrum statement. The interval
evaluation must enclose the exact represented `Q`, `D` and `H`, and arithmetic
overflow, invalid endpoints or a failed `eta < 1` gate must reject admission.
Directed-next-float arithmetic also requires its stated IEEE rounding and
underflow behavior; flush-to-zero or denormals-are-zero execution invalidates
naive subnormal enclosures. Runtime admission must establish those premises or
use a bounding method that remains valid under the detected mode.

The cell list must cover the complete torus. Adaptive subdivision replaces a
parent cell with all four children; accepting only favorable children leaves
the global premise open. A retained certificate records every terminal cell,
its bounds and bundle boundary indices, including failed admissions.

## Implementation and source boundaries

`quantum_core::tight_binding::TightBindingModel::hamiltonian_at_k` implements
the periodic-cell hopping convention. The independently reconstructed geometry
in `crates/quantum_core/tests/magnonic_source_projection.rs` supplies the
source-fitted nine-orbital models. An exact-dyadic coefficient certificate
applies to the represented model; extending that certificate to rounded source
parameter intervals requires a separate perturbation enclosure.
The model identity includes the final on-site and hopping bit patterns,
including geometry-derived coefficients. A tolerance-based test for a small
imaginary hopping amplitude cannot replace the exact real-hopping premise.

The source Table I and Table II coefficients were fitted upstream. Global
isolation and zero bundle Chern number establish mathematical properties of
those models. Those properties do not establish independent device prediction,
localized defect states, bulk flat-band identification or a physical GHz
calibration. The E-299 receipt retains those separate source and observational
boundaries in `data/output/audit/magnonic-source-projection/adjudication.toml`.
Admission of the source Table I or Table II models also leaves the historical
C-1688 `delta_eps_s=0.05/0.10` rank-one bands under their original producer and
degeneracy obligations.
