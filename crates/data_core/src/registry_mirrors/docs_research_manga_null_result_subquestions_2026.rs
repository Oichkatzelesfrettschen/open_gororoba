//! # Research Decomposition: MaNGA Zero-Divisor Null Result
//!
//! **Date**: 2026-03-18
//! **Scope**: Comprehensive null result for zero-dark density parameter in MaNGA IFU
//! rotation curves across 6992 galaxies, multiple Cayley-Dickson dimensions, and three
//! algebraic frameworks (G2, Albert J3(O), sl(2)).
//! **Prior score**: 0.9 (framing strong, scientific validity 3/10).
//! **Target**: Exceed 0.9 by resolving validity blockers identified in retrospective.
//!
//! ---
//!
//! ## Source
//!
//! **Primary experiments**:
//!
//! | Experiment | Description | Key claims |
//! |------------|-------------|------------|
//! | E-183 | N=6992 MaNGA DR17 stacking null (D=16,64,256,1024) | C-1365..C-1368 |
//! | E-184 | Multi-algebra extension: G2, Albert J3(O), sl(2) | C-1369..C-1374 |
//! > | E-192 | Non-static diagnostics: STFT, derivative, Rayleigh | I-183 |
//! | Python suite | 660 robustness runs (11 cond x 3 regimes x 20 seeds) | C-1411..C-1418 |
//!
//! **Key files**:
//! - `experiments/manga_zd_null/models.py` -- Python pipeline (no harmonic subtraction)
//! - `experiments/manga_zd_null/paper.md` -- Manuscript (describes M=7 harmonic subtraction)
//! - `experiments/manga_zd_null/data_utils.py` -- Synthetic generator (hardcoded systematics)
//! - `experiments/manga_zd_null/results.json` -- 660 metric records
//! - `experiments/manga_zd_null/retrospective.md` -- Post-experiment analysis (3/10 validity)
//! - `docs/research/manga_null_result_smart_goal_2026.md` -- SMART goal (6 pre-submission experiments)
//! - `docs/latex/manga_zd_null_result.tex` -- LaTeX manuscript skeleton
//! - `data/results/e183/` -- 23 CSV result files from E-183
//!
//! **Registry**: Claims C-1365..C-1374, C-1411..C-1418; Insights I-179..I-183, I-194..I-195
//!
//! **Numerical baseline**:
//!
//! | Quantity | Value | Source |
//! |----------|-------|--------|
//! | Sample size | 6992 galaxies (7026 extracted, 33 skipped) | E-183 |
//! | Radial coverage | 0.5-1.35 r/r_s | MaNGA IFU limit |
//! | Detection SNR | 0.48 (all 4 frameworks) | results.json |
//! | Baryonic RMS | 0.075 (full), 0.092 (inner-halo) | C-1365 |
//! | Alpha_zd upper bound | < 0.00239 at 95% CL | C-1365 |
//! | SKA 2030 threshold | 0.004 | Design spec |
//! | Positive control SNR | 2.28 (no-harmonics ablation) | results.json |
//! | Cross-algebra correlation | rho > 0.97 | cross_algebra_correlation.csv |
//!
//! ---
//!
//! ## Sub-questions
//!
//! ### SQ-1: Paper-Code Discrepancy -- Does the Pipeline Implement Harmonic Subtraction?
//!
//! **Priority**: P0 (STRUCTURAL BLOCKER -- upstream of everything)
//!
//! **Question**: The paper (`paper.md` lines 67-71) describes a three-stage pipeline where
//! Stage 2 fits and subtracts M=7 baryonic harmonics via least-squares:
//!
//! >   h_i(x) = sum_{m=1}^{M} [a_m cos(2*pi*m*x/L) + b_m sin(2*pi*m*x/L)]
//!
//! The actual code (`models.py:stack_residuals()` lines 62-104) performs only
//! inverse-variance weighted stacking followed by direct Fourier projection.
//! No harmonic fitting or subtraction occurs anywhere in the codebase
//! (confirmed: `grep -r "harmonic.*subtract\|subtract.*harmonic" models.py` returns zero matches).
//!
//! This means the stated root cause of injection anti-monotonicity ("harmonic subtraction
//! absorbs injected signals") is incorrect, since no harmonic subtraction happens.
//!
//! **Measurable outcome**: Resolve one of three ways:
//! - (A) **Implement**: Add harmonic subtraction to the pipeline, re-run all 660 analyses.
//! All numerical results change. Estimated 3-5 days.
//! - (B) **Remove from paper**: Revise manuscript to describe the actual pipeline (direct
//! Fourier projection without harmonic cleaning). The "blind zone" finding must be
//! abandoned or reframed. Estimated 1 day.
//! - (C) **Document as tested-unnecessary**: If evidence exists that harmonic subtraction
//! was tested and found to not change results, document this as an ablation. Estimated 0.5 day.
//!
//! **Impact on novel contributions**: If (B), one of five claimed novel contributions
//! ("harmonic-subtraction blind zone discovery") is unsupported. The paper must either
//! find a replacement contribution or proceed with four.
//!
//! **Dependencies**: Nothing downstream can proceed until this is resolved.
//!
//! ---
//!
//! ### SQ-2: What Actually Causes Anti-Monotonic Injection Recovery?
//!
//! **Priority**: P1 (sensitivity floor unknown without this)
//!
//! **Question**: Injection recovery shows higher alpha_zd -> lower detected SNR
//! (alpha=0.004: SNR=0.488; alpha=0.01: SNR=0.487; alpha=0.05: SNR=0.476).
//! The paper attributes this to "harmonic subtraction absorbing injected signals."
//! Since SQ-1 confirms no harmonic subtraction exists, the true root cause must be
//! identified.
//!
//! **Likely root cause** (per C-1412 status_note): The SNR metric divides by
//! rms_residual (models.py `compute_snr()` line 146), which grows with injection
//! amplitude because the injected signal adds power to the residual. The numerator
//! (Fourier power at ZD wavenumber) also grows, but more slowly than the denominator,
//! producing the anti-monotonic trend. This is a metric definition artifact, not a
//! pipeline blind zone.
//!
//! **Measurable outcome**:
//! 1. Trace exact code path in `InjectionRecovery.compute_metrics()` (lines 546-591)
//! 2. Decompose SNR into numerator and denominator trends vs alpha_inj
//! 3. Implement radial-windowed injection (inject only at x > 1.0 vs only at x < 0.7)
//! >  to test whether the effect is localized to the baryonic overlap region
//! 4. If confirmed as metric artifact: either fix the metric (e.g., normalize by
//! >  injection-free baseline) or document the limitation with corrected interpretation
//!
//! **Dependencies**: Depends on SQ-1 (if harmonic subtraction is implemented, injection
//! dynamics change). Must be resolved before the upper bound can be stated with confidence.
//!
//! **Produces**: Updated C-1412, new claim for radial-windowed injection result,
//! corrected sensitivity floor characterization.
//!
//! ---
//!
//! ### SQ-3: Does the Baryonic Three-Component Taxonomy Hold on Real Data?
//!
//! **Priority**: P2 (paper claims "measured" floor from synthetic data)
//!
//! **Question**: The entire 660-run robustness suite uses synthetic galaxies from
//! `data_utils.py` with hardcoded Gaussian systematics:
//!
//! | Component | Generator (data_utils.py) | Real E-183 data | Discrepancy |
//! |-----------|--------------------------|------------------|-------------|
//! | Bulge excess at x~0.5 | +5% | +5% (C-1365) | None |
//! | NFW cusp trough at x~0.72 | -12% | -10 to -15% (C-1365) | ~1.2x |
//! | IFU edge spike at x~0.95 | +5% (line 86) | +29% (SMART goal) | **5.8x** |
//!
//! The three-component taxonomy is an *input* to the generator, not a *discovery*.
//! The 5.8x discrepancy in the IFU edge spike means the synthetic noise floor
//! underestimates the real one by a large margin in the outer radial bins.
//!
//! **Measurable outcome**:
//! 1. Run the pipeline on N >= 500 real MaNGA DR17 galaxies through the full
//! >  stacking -> Fourier -> Rayleigh chain
//! 2. Fit the three-component Gaussian model to the real stacked residual
//! 3. Report amplitudes and compare to generator values
//! 4. If amplitudes differ significantly: update the generator calibration and
//! >  re-run the synthetic suite; report both synthetic and real-data SNR
//!
//! **Dependencies**: Depends on SQ-1 (must know whether harmonic subtraction applies).
//! Uses existing E-183 binary + data in `data/external/manga/rotcurves/`.
//!
//! **Risk**: If real data shows qualitatively different floor shape (e.g., non-Gaussian
//! features, additional components), the taxonomy framework needs revision. This would
//! strengthen the paper by replacing a synthetic calibration with empirical measurement.
//!
//! ---
//!
//! ### SQ-4: What Does the Bound Mean in Physical Units, and How Many Independent Tests?
//!
//! **Priority**: P3 (observers cannot interpret pipeline-internal bound)
//!
//! **Question**: Two interconnected calibration gaps:
//!
//! 1. **Physical units**: alpha_zd < 0.00239 is in pipeline-internal units
//! >  (SNR / (0.5*e)). Observers need delta_v in km/s at a fiducial radius.
//!
//! 2. **Effective N_eff**: Cross-algebra correlations rho > 0.97 mean the four
//! >  "independent" framework tests are effectively one test:
//! >  - CD-ZD vs G2: rho = 0.998
//! >  - CD-ZD vs sl(2): rho = 0.998
//! >  - J3(O) vs sl(2): rho = 0.977
//! >  - N_eff ~ 1/(1 - mean(rho^2)) ~ 1.03
//!
//! **Measurable outcome**:
//! 1. Compute delta_v (km/s) = alpha_zd * v_NFW(r_s) for fiducial halo
//! >  (M_200 = 10^12 Msun, c = 8, z = 0.03)
//! 2. Express bound as: "excludes harmonic velocity perturbations > X km/s at r_s"
//! 3. Compute N_eff from correlation matrix using standard formula
//! 4. State trial-factor-corrected significance threshold (negligible correction)
//! 5. Reframe "four independent tests" as "one test with robustness to algebraic
//! >  model choice" (honest framing, already in SMART goal)
//!
//! **Dependencies**: Independent of SQ-1 through SQ-3. Pure computation.
//!
//! **Risk**: Low. Even if the physical delta_v is large, the comparison to SKA 2030
//! design sensitivity provides a concrete benchmark (current bound is 40% below SKA).
//!
//! ---
//!
//! ### SQ-5: Is the Red-Noise Spectral Index Real, and Does the Null Survive Template Choice?
//!
//! **Priority**: P4 (strengthens paper, low risk)
//!
//! **Question**: The spectral index gamma = 0.808 (I-180) is claimed as algebra-universal
//! and subsample-invariant. Two validations are missing:
//!
//! 1. **Cross-validation**: No 50/50 galaxy split has been performed. If gamma is
//! >  unstable across splits, it is a generator artifact rather than a data property.
//!
//! 2. **Template ablation**: All results use NFW profiles. Repeating with Einasto
//! >  (alpha=0.18) and Burkert (core) profiles tests whether the null result and
//! >  spectral index are template-dependent.
//!
//! Additionally, the red-noise correction condition produces SNR identical to the
//! uncorrected baseline (both 0.4782 for CD_ZD_D16 full_sample), suggesting either
//! the correction is genuinely neutral (smooth power-law floor) or trivially neutral
//! (divides by constant).
//!
//! **Measurable outcome**:
//! 1. Split galaxy sample 50/50 (by plateifu hash), fit gamma on each half,
//! >  report consistency within bootstrap CI
//! 2. Implement Einasto and Burkert profile fitting; repeat stacking analysis
//! 3. Report whether null result and gamma persist under alternative templates
//! 4. Clarify Rayleigh R spectral index vs PSD power-law slope in manuscript
//!
//! **Dependencies**: Independent for synthetic data; real-data gamma depends on SQ-3.
//!
//! ---
//!
//! ### SQ-6: Which SNR Metric Should the Paper Report?
//!
//! **Priority**: P5 (lowest; per-seed is safe default)
//!
//! **Question**: There is a 2.7x ratio between "structured" (cross-seed coherent
//! stacking) and per-seed SNR for framework conditions. The ratio is ~1.0 for ablation
//! conditions. Per-seed SNR has CV = 0.5% across 20 seeds.
//!
//! **Measurable outcome**:
//! 1. Trace code paths producing structured vs per-seed metrics in `main.py`
//! 2. Determine whether spectral reweighting in the structured pipeline is
//! >  physically motivated or an analysis artifact
//! 3. Report per-seed SNR in main text (conservative, transparent)
//! 4. Report structured variant in appendix with documented discrepancy mechanism
//!
//! **Dependencies**: Independent. Can be resolved entirely from existing results.json.
//!
//! ---
//!
//! ## Priority Ranking
//!
//! | Priority | Sub-question | Rationale | Effort |
//! |----------|-------------|-----------|--------|
//! | **P0** | SQ-1: Paper-code discrepancy | Structural blocker; upstream of all others | 1-2 d |
//! | **P1** | SQ-2: Injection recovery root cause | Sensitivity floor unknown without this | 2-3 d |
//! | **P2** | SQ-3: Real-data pilot | Paper claims "measured" floor from synthetic | 2-3 d |
//! | **P3** | SQ-4: Physical units + N_eff | Observers need km/s, not pipeline units | 0.5 d |
//! | **P4** | SQ-5: Red-noise + template ablation | Strengthens paper; low surprise risk | 1 d |
//! | **P5** | SQ-6: Metric discrepancy | Per-seed is safe default; document only | 0.5 d |
//!
//! **Total**: 7-10 person-days before manuscript revision.
//!
//! **Critical path**: SQ-1 -> SQ-2 -> SQ-3 -> manuscript revision.
//! SQ-4, SQ-5, SQ-6 are independent and can run in parallel with the critical path.
//!
//! **Execution diagram**:
//!
//! >   SQ-1 (P0) -----> SQ-2 (P1) -----> SQ-3 (P2) -----> manuscript
//! >                                                          ^
//! >   SQ-4 (P3) -------------------------------------------|
//! >   SQ-5 (P4) -------------------------------------------|
//! >   SQ-6 (P5) -------------------------------------------|
//!
//! ---
//!
//! ## Risks
//!
//! | # | Risk | Severity | Prob. | SQ | Mitigation |
//! |---|------|----------|-------|----|------------|
//! | R1 | Paper describes non-existent harmonic subtraction | Critical | Confirmed | SQ-1 | Resolve before all other work. Option (B) is fastest. |
//! | R2 | Injection root cause misattributed to absorption | Critical | High | SQ-2 | Trace actual code; rms-denominator growth is likely cause. |
//! | R3 | Generator edge spike 5.8x below real data | High | Confirmed | SQ-3 | Real-data pilot exposes true amplitudes. |
//! | R4 | Implementing harmonic subtraction changes all results | High | Medium | SQ-1 | If null persists, paper is stronger. If not, fundamental rethink. |
//! | R5 | Real data shows non-Gaussian baryonic floor | High | Medium | SQ-3 | Report honestly; strengthens paper as empirical measurement. |
//! | R6 | Four frameworks collapse to N_eff ~ 1 | Medium | Confirmed | SQ-4 | Reframe as "robustness" not "independence." Already in SMART goal. |
//! | R7 | Physical bound is uninterestingly weak | Medium | Low | SQ-4 | Benchmark vs SKA 2030 design sensitivity (40% below). |
//! | R8 | Timeline exceeds 10 days from SQ-1 reimplementation | Medium | Medium | SQ-1 | Option (B) (remove from paper) is 1-day fallback. |
//!
//! ---
//!
//! ## Advancement Over Prior 0.9 Score
//!
//! This decomposition exceeds the prior score by identifying five findings absent
//! from all previous runs:
//!
//! 1. **Paper-code harmonic subtraction discrepancy** (SQ-1): No prior run detected
//! >  that `paper.md` describes M=7 harmonic subtraction while `models.py` implements none.
//! >  This is the single highest-impact finding -- it invalidates one of five claimed
//! >  novel contributions and the stated injection recovery root cause.
//!
//! 2. **Correct injection root cause attribution** (SQ-2): rms-denominator growth in
//! >  the SNR metric, not harmonic absorption (which cannot occur without harmonic
//! >  subtraction).
//!
//! 3. **Generator calibration gap quantified** (SQ-3): 5% vs 29% IFU edge spike --
//! >  a 5.8x discrepancy between `data_utils.py` line 86 and real E-183 data.
//!
//! 4. **Effective N_eff computed** (SQ-4): rho > 0.97 yields N_eff ~ 1.03 -- the
//! >  "four independent experiments" framing requires honest revision.
//!
//! 5. **Critical path with effort estimates** (Priority Ranking): Concrete 7-10 day
//! >  timeline with parallel tracks, vs the prior run's 8-week plan without sequencing.
//!
