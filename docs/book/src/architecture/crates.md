# Crate Ecosystem

The workspace uses 30+ external crates for numerical computation, data formats,
and hardware acceleration.  Key adopted crates:

## Numerical foundations

| Crate | Version | Purpose |
|-------|---------|---------|
| nalgebra | 0.33 | Linear algebra (pinned due to ode\_solvers constraint) |
| rustfft + realfft | 6.4.1 + 3.5.0 | FFT (NOT hand-rolled) |
| gauss-quad | 0.2.4 | Gauss-Legendre quadrature |
| statrs | 0.18 | Statistical distributions (Normal.inverse\_cdf) |
| ellip | 1.0.4 | Carlson symmetric + Legendre elliptic integrals |

## Data structures and algorithms

| Crate | Version | Purpose |
|-------|---------|---------|
| kodama | 0.3.0 | Hierarchical clustering (dendrograms) |
| kiddo | 5.2.4 | k-d trees with AVX2 (spatial queries) |
| petgraph | 0.7 | Graph algorithms |
| wide | 0.7 | SIMD (f64x4 in Cayley-Dickson) |

## Domain-specific

| Crate | Version | Purpose |
|-------|---------|---------|
| atlas\_embeddings | 0.1.1 | E8 root systems |
| qua\_ten\_net | 0.2.0 | Tensor networks |
| fitsrs + votable | 0.4.1 + 0.7.0 | Astronomical data formats |
| satkit | 0.9.3 | Satellite toolkit, coordinate frames |
| cudarc | 0.19.1 | CUDA via dynamic loading (NVRTC) |

## Hand-rolled code that stays

Some algorithms remain hand-rolled for justified reasons:

- **RK4 steppers** (6x): domain-specific structs, 15 lines each, non-stiff
- **Gram-Schmidt QR** (3x): Mezzadri Haar measure trick needs R diagonal
- **Frechet distance**: no suitable crate for `&[f64] -> &[f64]`
- **Bootstrap CI**: tightly integrated with claims framework
- **Hurst exponent**: `hurst` crate is GPL-3.0 (license conflict)
- **Cosmological distances**: domain-specific (Macquart DM)

## Known constraints

- **nalgebra 0.33 vs 0.34**: num-dual 0.13.2 requires nalgebra 0.34, but
  workspace is pinned to 0.33.  Autodiff deferred until ode\_solvers updates.
- **HDF5 2.0.0**: The original `hdf5` 0.8.1 crate panics on HDF5 2.0.
  We use `hdf5-metno` 0.12 (MET Norway fork), which supports HDF5 1.8.4
  through 2.0.x.  The `hdf5-export` feature requires `libhdf5` at link time.
