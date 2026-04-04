//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Warp Ring N^3: Physics, Numerics, And What The Data Means
//!
//! This document is the repo-scoped, implementation-aligned interpretation for the
//! 3D "warp ring" lane that produces artifacts like:
//!
//! * `warp_ring_32_...`
//! * `warp_ring_128_...`
//! * `warp_ring_256_...`
//!
//! where `N^3` is the cubic grid resolution.
//!
//! It is written to follow the ANSI-safe UTF-8 policy and to track what the Rust + CUDA code in this
//! repo actually does.
//!
//! ## 1. What `N^3` Represents
//!
//! `N^3` means a 3D periodic lattice with:
//!
//! * `nx = ny = nz = N`
//! * `n_cells = N * N * N`
//!
//! Every time step updates a state defined on all `n_cells`.
//!
//! In this repo the primary 3D fluid backend is a D3Q19 Lattice Boltzmann Method
//! (LBM) solver:
//!
//! * CPU: `lbm_3d::solver::LbmSolver3D` (FP64)
//! * GPU: `lbm_3d_cuda::LbmSolver3DCuda` (FP32 or BF16 storage)
//!
//! The "warp ring" lane additionally applies an `E7SpectralFilter` forcing step
//! periodically (every 10 steps in the current implementation) which uses FFTs in
//! frequency space and a mask derived from the E7 root system.
//!
//! ## 2. Core LBM Update (D3Q19, BGK)
//!
//! LBM evolves particle distribution functions `f_i(x, t)` for discrete velocities
//! `c_i` (19 directions in D3Q19).
//!
//! At each lattice cell `x` and time step `t`, the solver performs (conceptually):
//!
//! 1. Compute macroscopic density and velocity:
//!
//! >  rho(x) = sum_i f_i(x)
//! >  u(x)   = (1/rho(x)) * sum_i c_i * f_i(x)
//!
//! 2. Compute equilibrium distributions `f_i_eq(rho, u)` (low-Mach expansion):
//!
//! >  f_i_eq = w_i * rho * [ 1
//! >                         + 3 (c_i . u)
//! >                         + 4.5 (c_i . u)^2
//! >                         - 1.5 (u . u) ]
//!
//! 3. Relax toward equilibrium with BGK relaxation time `tau`:
//!
//! >  f_i_post = f_i - (1/tau) * (f_i - f_i_eq)
//!
//! 4. Apply forcing (Guo-style form), then stream distributions along `c_i` to
//! >  neighbor cells.
//!
//! In lattice units, the kinematic viscosity is:
//!
//! nu = (tau - 0.5) / 3
//!
//! So stability and physical regime selection depend strongly on ensuring `tau > 0.5`
//! (and typically not too close to 0.5 for noisy/turbulent runs).
//!
//! ### 2.1 Physically Derived Low-Level Forcing (Kolmogorov Flow)
//!
//! The low-level body forcing used by the warp benchmark lane is now tied to the
//! steady incompressible Navier-Stokes balance for Kolmogorov flow (sinusoidal
//! body forcing in a periodic box), not a hand-tuned constant.
//!
//! Reference model equations (see:
//! `https://arxiv.org/abs/1308.3356`,
//! `https://arxiv.org/abs/2101.05176`,
//! `https://arxiv.org/abs/2105.00730`):
//!
//! d_t u + (u . grad)u = -(1/rho) grad p + nu laplacian(u) + f
//! f_x(y) = F0 * sin(k y)
//! u_x(y) = U0 * sin(k y)
//!
//! For this unidirectional sinusoidal ansatz, the nonlinear term cancels and
//! steady balance gives:
//!
//! F0 = nu * k^2 * U0
//!
//! with forcing wave number:
//!
//! k = 2*pi*m / Ny
//!
//! where `m` is the forcing mode and `Ny` is the y-direction lattice size.
//!
//! To avoid arbitrary amplitude selection, `U0` is computed from a target forcing-
//! scale Reynolds number and a low-Mach cap:
//!
//! Re_target = U0 / (nu * k)
//! U0 = min(Re_target * nu * k, Ma_max * c_s)
//!
//! then:
//!
//! F0 = nu * k^2 * U0
//!
//! In lattice units (D3Q19 BGK):
//!
//! c_s^2 = 1/3
//! nu = c_s^2 * (tau - 0.5)
//!
//! The default operational values in `warp_runner` are controlled by env vars:
//!
//! * `GOROROBA_KOLMO_MODE_Y` (default `1`)
//! * `GOROROBA_KOLMO_RE_TARGET` (default `64.0`)
//! * `GOROROBA_KOLMO_MAX_MACH` (default `0.08`)
//! * `GOROROBA_KOLMO_RHO0` (default `1.0`)
//!
//! This means the force amplitude is derived from physically interpretable controls
//! (`nu`, `k`, `Re_target`, `Ma_max`) and emitted into timing reports for audit.
//!
//! ### 2.2 Source-Grounded LBM Forcing Link (Not Hand-Wavy)
//!
//! This lane's forcing model is anchored to three established pieces:
//!
//! 1. Kolmogorov-flow forcing shape:
//! >  sinusoidal body-force-driven shear flow (`f_x ~ sin(k y)`), widely used as a
//! >  canonical turbulence model problem in periodic domains.
//! >  (e.g. `https://arxiv.org/abs/1308.3356`, `https://arxiv.org/abs/2101.05176`)
//!
//! 2. LBM forcing discretization:
//! >  the Guo forcing form with the characteristic BGK prefactor
//! >  `(1 - 1/(2*tau))` used in this repo's CUDA kernels is from:
//! >  Guo, Zheng, Shi (2002), Phys. Rev. E 65, 046308.
//! >  (`https://doi.org/10.1103/PhysRevE.65.046308`)
//!
//! 3. Low-Mach validity regime:
//! >  weakly compressible LBM recovers Navier-Stokes in the low-Mach limit with
//! >  truncation error scaling in Mach number, so we enforce a Mach cap.
//! >  (`https://doi.org/10.1023/B:JOSS.0000015179.12689.e4`)
//!
//! In other words: the code-level force field
//!
//! f_x(y) = F0 * sin(2*pi*m*y/Ny),  F0 = nu * k^2 * U0
//!
//! is not arbitrary. It is the steady Kolmogorov balance parameterized in lattice
//! units and then discretized with a standard LBM forcing scheme.
//!
//! ### 2.3 Quantitative Consequence Of The Mach Cap
//!
//! When the cap is active (`U0 = Ma_max * c_s`), and `m` is fixed:
//!
//! k = 2*pi*m/Ny
//! Re_effective = U0 / (nu*k)  ~  Ny
//! F0 = nu*k^2*U0              ~  Ny^(-2)
//!
//! So for the same `tau`, `m`, and `Ma_max`, moving from `128^3` to `256^3` gives:
//!
//! * `Re_effective` about 2x larger
//! * `F0` about 4x smaller
//!
//! This scaling is expected and physically meaningful (not a bug): larger domains
//! at fixed mode index push the forcing to larger length scales.
//!
//! ### 2.4 BF16 Quantization Limit On Low-Mach Forcing Increments
//!
//! The forcing model above is physically grounded, but BF16 storage can still hide
//! that forcing if per-step updates are too small at the distribution-function
//! scale.
//!
//! For D3Q19, a representative equilibrium magnitude is:
//!
//! f_eq ~ w_axis * rho0,   w_axis = 1/18
//!
//! BF16 has 7 mantissa bits, so one-ULP scale near that magnitude is:
//!
//! ulp_bf16(f_eq) ~ f_eq * 2^(-7)
//!
//! The Guo forcing term contributes a per-step distribution increment with scale:
//!
//! Delta f ~ (1 - 1/(2*tau)) * 3 * w_axis * F0
//!
//! So a useful non-dimensional detectability ratio is:
//!
//! R_bf16 = Delta f / ulp_bf16(f_eq)
//!
//! If `R_bf16 << 1`, forcing is physically present but can be rounded away at BF16
//! distribution precision, making velocity-derived traces look static. The timing
//! reports now emit `bf16_delta_to_ulp_ratio` to expose this regime directly.
//!
//! ## 3. The E7 Spectral Filter (What It Does Numerically)
//!
//! The E7 spectral filter is a periodic forcing / damping operation applied to the
//! velocity field `u(x)`:
//!
//! 1. Take one component at a time: `u_x`, `u_y`, `u_z`.
//! 2. Convert the real field to a complex field (imag=0).
//! 3. Compute 3D FFT to frequency space: `u_hat = FFT(u)`.
//! 4. Apply a mask in frequency space:
//!
//! >  if mask[k] < threshold:
//! >      u_hat[k] *= damping
//!
//! >  The mask is constructed from projected E7 roots and then normalized.
//! >  The damping factor is adapted from enstrophy via:
//!
//! >    alpha = tanh(enstrophy / enstrophy_crit)
//! >    damping = 1 - (1 - base_damping) * alpha
//!
//! 5. Compute inverse FFT back to real space.
//!
//! Important normalization detail:
//!
//! cuFFT (and most FFT libraries) compute *unnormalized* transforms:
//!
//! IFFT(FFT(u)) = (N^3) * u
//!
//! So to recover the original scale after inverse FFT we multiply by:
//!
//! scale = 1 / (N^3)
//!
//! In this repo that normalization is applied after the inverse FFT when converting
//! complex -> real.
//!
//! ## 4. What The Traces Mean (rho_mean, enstrophy, MLUPS)
//!
//! ### 4.1 rho_mean
//!
//! `rho_mean` is the mean density:
//!
//! rho_mean(t) = (1 / N^3) * sum_x rho(x, t)
//!
//! In an ideal periodic LBM with consistent forcing, mass conservation keeps
//! `rho_mean` near its initialization value (commonly ~1.0).
//!
//! The acceptance gates treat the `rho_mean` trace as a primary health signal:
//!
//! * NaN/Inf means the simulation diverged numerically.
//! * Large drift or large standard deviation often indicates instability, too small
//! `tau`, too aggressive forcing, or bugs in reductions / conversions.
//!
//! ### 4.2 Enstrophy
//!
//! Enstrophy is a measure of vorticity magnitude. In continuum fluid dynamics:
//!
//! enstrophy ~ integral |curl u|^2 dV
//!
//! In this repo it is approximated with finite differences on the lattice and then
//! summed/averaged.
//!
//! In the E7 forcing implementation, enstrophy feeds back into the damping
//! coefficient to avoid blowup and to adaptively suppress high-frequency structure.
//!
//! ### 4.3 algebra_norm
//!
//! `algebra_norm` is intended as a coupling-strength diagnostic from optional
//! algebraic/imbalance layers. When those layers are absent, this repo now falls
//! back to a physically meaningful proxy:
//!
//! 1. velocity RMS from the fluid field when available
//! 2. otherwise the Kolmogorov target velocity scale `U0`
//!
//! This avoids all-zero placeholder traces and preserves interpretability in pure
//! fluid lanes.
//!
//! ### 4.4 MLUPS
//!
//! MLUPS means "million lattice updates per second":
//!
//! MLUPS = (steps_per_sec * N^3) / 1e6
//!
//! This is a throughput metric for the whole time-step loop. It is dominated by
//! memory bandwidth and kernel efficiency on the GPU at larger `N`.
//!
//! ## 5. Scaling With N^3 (Why 256^3 Is Harder Than 128^3)
//!
//! For a fixed time step count:
//!
//! * State size (memory) scales as O(N^3).
//! * Core per-step work scales as O(N^3).
//!
//! However, performance is not perfectly linear with `N^3` because:
//!
//! * Cache behavior changes (CPU) and working set grows.
//! * GPU kernels become increasingly memory-bandwidth bound.
//! * FFT and reduction costs can scale with additional constants and synchronization
//! points (especially if host reads are done in the loop).
//!
//! As a practical rule:
//!
//! * 128^3 often fits comfortably in GPU caches / bandwidth regimes.
//! * 256^3 stresses memory traffic and any extra passes (reductions, FFT staging)
//! become more visible.
//!
//! ## 6. Why Naive GPU Timers Lie (And What We Do Instead)
//!
//! CUDA kernel launches are asynchronous: the CPU call returns after enqueuing work.
//!
//! If you measure step time like:
//!
//! t0 = now()
//! state.step()
//! dt = now() - t0
//!
//! then `dt` is often just "launch overhead", not the actual GPU execution time.
//! This produces unrealistically small p50 timing values and a confusing long tail
//! only when the code *happens* to synchronize (for example: reductions with device
//! to host transfers at trace points).
//!
//! To make timing truthful, the warp runner supports:
//!
//! * Stream sync timing (synchronize after each step).
//! * CUDA event timing (record events on the solver stream and compute elapsed time).
//!
//! Both approaches ensure the reported timings correspond to actual device work.
//!
//! ## 7. BF16 vs FP32 (What "BF16" Means Here)
//!
//! In the GPU solver:
//!
//! * FP32 mode stores primary state in FP32.
//! * BF16 mode stores primary state in BF16, but performs most math in FP32.
//!
//! BF16 reduces memory bandwidth pressure but introduces quantization noise.
//! The acceptance gates (rho_mean finite + drift bounds) exist to ensure BF16
//! does not silently destabilize the physics lane.
//!
//! ## 8. Practical Interpretation Of Artifacts
//!
//! When you see artifacts like:
//!
//! * `warp_ring_128_BF16_300s.h5`
//! * `timing_128_BF16_300s.toml`
//! * `gpu_telemetry.csv`
//!
//! Interpret them as:
//!
//! * `*.h5`: the trace-level physics outputs (time series) and contract metadata.
//! * `timing_*.toml`: performance and stability summary for the run (including
//! step timing histograms).
//! * telemetry: external measurement of GPU operating state (temperature, clocks,
//! power, utilization) to help explain throughput changes and throttling.
//!
//! ## 9. What We Optimize (And What We Gate)
//!
//! Optimization targets:
//!
//! * Increase MLUPS at 128^3 and 256^3, especially in BF16 mode.
//! * Reduce sync points and unnecessary memory traffic.
//! * Keep acceptance gates strict so faster code does not corrupt the physics lane.
//!
//! Gates:
//!
//! * HDF5 numeric gate: no NaN/Inf anywhere in numeric datasets.
//! * Simulation trace gate: required trace datasets exist and are finite.
//! * Rho trace gate: rho_mean drift and variance within thresholds.
//!
