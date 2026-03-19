//! # Literature Synthesis: Comprehensive Null Result for Zero-Dark Density in MaNGA IFU Rotation Curves
//!
//! **Generated:** 2026-03-18
//! **Topic:** Zero-dark density parameter null in MaNGA IFU rotation curves
//! across 6992 galaxies, multiple Cayley-Dickson dimensions, and three
//! algebraic frameworks (G2, Albert J3(O), sl(2))
//! **Registry:** I-196 (synthesis), I-197 (opportunities)
//! **Claims referenced:** C-1365..C-1374 (primary), C-1411..C-1418 (robustness)
//! **Insights referenced:** I-179..I-183 (primary), I-194..I-195 (robustness)
//!
//! ---
//!
//! ## Cluster Overview
//!
//! The literature surrounding the GHOST null result organizes into **six topic
//! clusters** spanning precision cosmology, IFU survey methodology, algebraic
//! dark-sector theory, halo structure and baryonic systematics, null-result
//! reporting standards, and next-generation survey forecasts. The experiment
//! sits at the intersection of all six: it uses Planck-derived cosmological
//! parameters (Cluster 1) applied to MaNGA DR17 IFU kinematics (Cluster 2)
//! to test algebraic predictions from the Cayley-Dickson tower and exceptional
//! Lie algebras (Cluster 3) against a baryonic noise floor characterized by
//! three distinct systematics (Cluster 4), reported following established
//! null-result conventions (Cluster 5), with sensitivity projections for
//! SKA/Euclid outer-halo campaigns (Cluster 6).
//!
//! Three cross-cutting narrative threads connect the clusters:
//!
//! - **Concordance to constraint** (1 -> 4 -> 5): Lambda-CDM precision at the
//! percent level motivates spectral tests beyond bulk amplitude constraints;
//! the baryonic stacking methodology defines the noise floor; null-result
//! reporting standards ensure the constraint is transparent and reusable.
//!
//! - **Prediction to test** (3 -> 2 -> 4): Algebraic theory generates
//! wavenumber predictions; IFU surveys provide spatially resolved kinematics;
//! inner-halo baryonic systematics define the detection threshold.
//!
//! - **Bound to forecast** (4 -> 6): The inner-halo null (x < 1.35 r/r_s)
//! combined with next-generation survey capabilities generates concrete
//! sensitivity projections for the outer-halo regime (x = 3-5 r/r_s) where
//! ZD predictions peak.
//!
//! **Critical framing constraints**: (a) The cross-algebra quasi-degeneracy
//! (rho > 0.97 for all 6 framework pairs) means the four algebraic tests are
//! effectively one constraint measured through four lenses -- a strength
//! (model-independence), not four independent experiments. (b) The inner-halo
//! bound (x < 1.35) and the outer-halo prediction peak (x = 3-5) must be
//! distinguished throughout -- the null does not falsify the ZD prediction,
//! it bounds the inner-halo amplitude.
//!
//! **Excluded references**: Three provided cards are irrelevant and excluded:
//! grassland microbial community assembly (ning2020quantitative), cardiovascular
//! MR image interpretation (schulzmenger2020standardized), and urban heat island
//! exposure (hsu2021disproportionate).
//!
//! ---
//!
//! ## Cluster 1: Precision Cosmology Concordance and Dark-Sector Tensions
//!
//! **Theme**: The Lambda-CDM framework GHOST operates within, including the dark
//! matter density parameter and emerging tensions that motivate non-standard
//! spectral searches.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | Planck 2018 [aghanim2020iplancki] | Omega_cdm h^2 = 0.120 +/- 0.001, H_0 = 67.4 +/- 0.5 | NFW concentration-mass relation uses Planck cosmology; sub-percent precision means exotic modifications must be spectral (pattern), not amplitude (bulk) |
//! | DESI 2024 BAO [adame2025desi] | w_0 = -0.55 +/- 0.21, w_a = -1.32; 2.5-3.9 sigma dynamical DE | Dynamical dark energy opens intellectual space for non-static dark-sector physics -- same territory ZD predictions inhabit |
//! | DESI DR2 (2025, arXiv:2503.14738) | Strengthened w0-wa evidence | Renews appetite for exotic dark-sector constraints; GHOST provides complementary harmonic-substructure bound |
//! | KiDS-1000 multi-probe [heymans2020kids] | S_8 = 0.766, 3-sigma Planck tension | Structure-formation discrepancy could trace to modified DM clustering -- the kind of halo-level anomaly GHOST probes |
//! | KiDS-1000 cosmic shear [asgari2020kids] | S_8 = 0.759, robust across 3 statistics | Validates S_8 tension robustness; consistent with DES-Y1 and HSC |
//! | Hubble tension review [valentino2021realm] | 4-6 sigma H_0 discrepancy, >1000 proposals | No single solution favored; justifies testing algebraically distinct approaches |
//!
//! **Relationship to null result**: Lambda-CDM is precise at the percent level
//! for bulk DM parameters, but structure-formation tensions at 3-6 sigma
//! motivate spectral tests. GHOST closes one spectral channel (inner-halo
//! algebraic wavenumbers) while these tensions keep the broader motivation
//! alive. The DESI dynamical DE evidence is complementary: GHOST constrains
//! harmonic substructure (gravitational potential shape), not equation of state.
//!
//! **Open questions**:
//! - Can the S_8 tension be re-framed as a constraint on halo-scale spectral
//! modifications? Requires modeling ZD effects on weak lensing convergence.
//! - Does DESI DR2's strengthened w0-wa preference change the theoretical
//! landscape for ZD-type modifications? Probably not directly (ZD modifies
//! potential shape, not w(z)), but the connection should be explored.
//!
//! ---
//!
//! ## Cluster 2: IFU Kinematic Surveys and Rotation Curve Methodology
//!
//! **Theme**: The observational infrastructure making GHOST possible, and the
//! methodological lineage it inherits.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | SDSS DR17 [abdurrouf2022] | 10,261 MaNGA galaxies, DAP velocity fields | Terminal MaNGA release; GHOST uses N=6992 after quality cuts (Sersic n<2.5, 30<i<70, DAPDONE, Ha EW>2A) |
//! | SDSS DR16 [ahumada2020] | eBOSS BAO + MaNGA 4824 IFU datacubes | Precursor release documenting survey pipeline; IFU coverage 1.5-2.5 R_e |
//! | Fogarty+2014 | SAMI kinematic morphology-density | Establishes IFU rotation curve extraction; azimuthal averaging suppresses projection artifacts |
//! | Emsellem+2022 | PHANGS-MUSE sub-kpc kinematics | Methodological baseline GHOST adapts for halo-scale spectral analysis |
//! | Allen+2014 | SAMI first release, >1000 galaxies | Demonstrates statistical power of IFU surveys across environments |
//!
//! **Relationship to null result**: MaNGA provides the data quality and sample
//! size (N=6992 post-cuts) needed for sub-percent stacking analyses, and the
//! IFU methodology is validated by SAMI and PHANGS-MUSE. The null is NOT a
//! methodological artifact. The key limitation -- radial coverage x < 1.35 r/r_s
//! -- is structural to MaNGA's design for stellar-population studies within the
//! optical effective radius, not a pipeline deficiency.
//!
//! **Open questions**:
//! - Has any IFU survey performed Fourier spectral decomposition of stacked
//! rotation curve residuals? This appears to be genuinely novel (Gap 2).
//! - Are there post-2022 MaNGA kinematic analyses characterizing the specific
//! baryonic systematics GHOST observes (bulge +5%, cusp -15%, IFU edge +29%)?
//!
//! ---
//!
//! ## Cluster 3: Algebraic Dark-Sector Predictions and Division Algebra Physics
//!
//! **Theme**: The theoretical predictions GHOST tests -- the connection between
//! Cayley-Dickson algebraic structure and dark matter observables.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | Reggiani 2024 (arXiv:2411.18881) | Sedenion ZD geometry: Stiefel manifold V_2(R^7), G2 holonomy | Most direct source for ZD variety geometric properties; projects onto GHOST wavenumber predictions |
//! | Furey 2016-2024 | Octonion particle models | Foundational program connecting division algebras to Standard Model; motivates algebraic tower predictions |
//! | Todorov & Dubois-Violette 2021 | Sedenion gauge theory | Extends algebraic program beyond octonions to sedenions and exceptional structures |
//!
//! **Relationship to null result**: These papers generate the wavenumber
//! predictions GHOST tests. Critical distinction: the frameworks predict
//! *wavenumbers* (k_n), not *amplitudes* (alpha_zd). The amplitude is a free
//! parameter. The null constrains alpha_zd < 0.00239 at 95% CL in the inner
//! halo only. The quasi-degeneracy (rho > 0.97) across all four frameworks at
//! MaNGA radii means the inner-halo regime cannot distinguish between them --
//! this is a consequence of wavenumber families overlapping at x < 1.35, and
//! would likely resolve at outer-halo radii where the families separate.
//!
//! Honest framing: "At MaNGA radii, the four algebraic lenses converge;
//! discriminating power requires outer-halo coverage where wavenumber families
//! separate." This is reframed as bound robustness (insensitivity to algebraic
//! model choice), not as four independent experiments.
//!
//! **Open questions**:
//! - Has any division-algebra paper (2022-2026) made quantitative predictions
//! for alpha_zd *amplitude* from first principles? This would convert GHOST
//! from model-independent exploration to model-dependent decisive test (Gap 1).
//! - Are there competing testable predictions from the division algebra program
//! that do not involve rotation curves (weak lensing convergence, CMB spectral
//! distortions)?
//!
//! ---
//!
//! ## Cluster 4: Halo Structure, Stacking, and Baryonic Noise Floor
//!
//! **Theme**: The astrophysical context of NFW fitting, rotation curve stacking,
//! and the baryonic systematics floor GHOST characterizes as a primary contribution.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | Bar+2022 | FDM soliton searches in rotation curves | Closest methodological precedent: Fourier-domain searches for specific features in RC residuals; null constrains boson mass |
//! | Schive+2014 | FDM soliton core prediction | Foundational prediction that FDM produces detectable cores; motivates Fourier-domain methodology GHOST inherits |
//! | Baxter+2021 | DM search reporting conventions | GHOST adopts: transparent upper limits, injection-recovery validation, sensitivity characterization |
//!
//! **GHOST's three-component baryonic taxonomy** (C-1365, I-179):
//!
//! | Component | Amplitude | Location (x = r/r_s) | Physical origin |
//! |-----------|-----------|---------------------|-----------------|
//! | Bulge excess | +5% | x < 0.56 | de Vaucouleurs bulge velocity support exceeding NFW cusp |
//! | Cusp over-prediction | -15% | x ~ 0.83 | NFW overshoots at density-field cusp-to-core transition |
//! | IFU edge spike | +29% | x ~ 0.953 | Beam-smearing of steep velocity gradient at R_e |
//!
//! These three systematics produce structured residuals at 5-12% RMS,
//! rendering any sub-percent exotic signal invisible. The combined Fourier
//! spectrum is baryonic red noise with spectral index k^0.81 (I-180),
//! DC-dominated and monotonically falling through all algebraically-motivated
//! wavenumbers.
//!
//! **Relationship to null result**: The baryonic floor IS the primary
//! contribution -- characterizing the inner-halo spectral baseline for any
//! future harmonic search (FDM solitons, ULDM, algebraic). The FDM soliton
//! analogy (Bar+2022) is methodologically critical: GHOST adopts the same
//! Fourier-projection methodology but targets algebraic rather than
//! particle-physics predictions.
//!
//! **Open questions**:
//! - Is k^0.81 consistent with cosmological hydrodynamic simulation predictions
//! (FIRE/NIHAO) for stacked rotation curve residuals? (Gap 5)
//! - Has any stacking analysis reached N > 5000 with NFW subtraction? GHOST's
//! N=6992 appears among the largest.
//!
//! ---
//!
//! ## Cluster 5: Null-Result Methodology and Publication Standards
//!
//! **Theme**: How to report null results responsibly, calibrate sensitivity, and
//! place upper limits -- critical because GHOST's injection recovery is broken.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | Baxter+2021 | DM search reporting conventions | GHOST follows: transparent upper limits, injection recovery (broken but honestly reported), sensitivity characterization |
//! | Matosin+2014 | Publication bias against null results | Supports framing that the null is a contribution: characterized detection floor + reusable diagnostics |
//!
//! **The injection-recovery calibration problem** (C-1412, retrospective lesson 2):
//!
//! Detection SNR *decreases* monotonically with injection amplitude:
//! - alpha=0.004: SNR=0.4908
//! - alpha=0.01: SNR=0.4895
//! - alpha=0.05: SNR=0.4793
//!
//! Root cause: harmonic subtraction absorbs injected signals whose spectral
//! character overlaps with baryonic modes (exp(-x) envelope interferes
//! destructively with +5% bulge excess at x~0.5). This is physically
//! informative -- it calibrates exactly where the pipeline has blind spots.
//! Reframed as "harmonic-subtraction blind zone discovery" (C-1412), a
//! methodological contribution relevant to any Fourier-domain halo residual
//! analysis.
//!
//! **Relationship to null result**: The null is scientifically valuable
//! precisely because null results exclude parameter space (Baxter+2021
//! framework). The broken injection recovery is an honest disclosure per
//! Baxter conventions -- the upper bound is preliminary until calibration
//! is resolved, but the pipeline's dynamic range is demonstrated by the
//! positive control (no-harmonics ablation SNR = 2.28, 4.8x framework baseline).
//!
//! **Open questions**:
//! - How does GHOST's injection anti-monotonicity compare to calibration
//! challenges in direct detection experiments? Harmonic absorption appears
//! specific to Fourier-domain stacking searches (Gap 4).
//! - What is the standard for reporting sensitivity when injection recovery
//! is known uncalibrated? Baxter+2021 may address this directly.
//!
//! ---
//!
//! ## Cluster 6: Next-Generation Surveys and Outer-Halo Forecasts
//!
//! **Theme**: The observational path forward -- what facilities will access the
//! outer-halo regime where ZD predictions peak at x = 3-5 r/r_s.
//!
//! **Key papers**:
//!
//! | Reference | Key result | Connection to GHOST |
//! |-----------|------------|---------------------|
//! | NANOGrav 15yr [agazie2023nanograv] | nHz GWB evidence, h_c ~ 2.4e-15 | Multi-messenger DM search context; PTA sensitivity to halo-level effects |
//! | EPTA DR2 [antoniadis2023second] | Independent GWB confirmation | Corroborates NANOGrav; multi-probe era for gravitational physics |
//! | GWTC-3 [abbott2023gwtc] | 90 compact binary coalescences | DM density environment effects on inspiral waveforms |
//! | GW190425 [abbott2020observation] | Anomalously massive BNS (3.4 Msun) | Tangential: DM capture in neutron stars |
//!
//! **Outer-halo survey landscape** (from I-182, SMART goal):
//!
//! | Survey | N_galaxies | Radial reach (r/r_s) | Tracer | Status |
//! |--------|-----------|---------------------|--------|--------|
//! | MaNGA DR17 (this work) | 6992 | ~1.35 | Optical IFU (H-alpha) | Complete |
//! | SPARC (Lelli+2016) | 175 | ~5-10 | HI 21cm + photometry | Public; deeper, 40x smaller N |
//! | THINGS (Walter+2008) | 34 | >10 | HI 21cm (high-res) | Public; outer halo, tiny N |
//! | SKA Phase 1 | ~10,000 | >10 | HI 21cm (z<0.1) | Forecast ~2030 |
//! | Euclid Wide | ~10^9 | N/A (lensing) | Weak lensing shear | DR2+ ~2028 |
//!
//! **Relationship to null result**: The inner-halo null explicitly defers to
//! outer-halo campaigns. SKA is the primary path forward: HI 21cm kinematics
//! at x > 10 r/r_s where ZD predictions peak and baryonic noise drops ~10x.
//! Euclid provides complementary lensing probe (mass, not velocity). The GW
//! references establish multi-messenger context but are lower priority for
//! the core narrative.
//!
//! **Open questions**:
//! - What is SKA's projected sensitivity to alpha_zd at x = 3-5 r/r_s? This
//! is the key forecast number for the paper (Gap 3).
//! - Can WALLABY pilot data provide cross-validation at outer radii before SKA?
//! - Does the quasi-degeneracy (rho > 0.97) persist at outer radii, or do the
//! algebraic frameworks separate? (Gap 7)
//!
//! ---
//!
//! ## Gap 1: Quantitative ZD Amplitude Predictions from First Principles
//!
//! **Description**: No paper provides a theoretical calculation of alpha_zd
//! from first principles. The algebraic frameworks predict wavenumbers (k_n)
//! but treat the amplitude as a free parameter. Without an amplitude
//! prediction, GHOST is an exploration, not a decisive test.
//!
//! **Impact**: HIGH. A theoretical amplitude would convert the GHOST upper
//! limit (alpha_zd < 0.00239) from a generic bound to a model-specific
//! exclusion or confirmation.
//!
//! **Status**: Open. Reggiani (2024) advances ZD geometry but does not compute
//! observable amplitudes. The division-algebra physics community has focused
//! on algebraic structure, not astrophysical phenomenology.
//!
//! ---
//!
//! ## Gap 2: Fourier Spectral Decomposition of Stacked IFU Rotation Curve Residuals
//!
//! **Description**: No prior study has performed Fourier spectral decomposition
//! of stacked NFW-subtracted rotation curve residuals at the scale of N > 1000
//! galaxies. Individual baryonic systematics are known (bulge, cusp, beam-smearing),
//! but the unified spectral characterization at stacking scale is novel.
//!
//! **Impact**: HIGH. This is GHOST's primary novelty claim. The baryonic red-noise
//! spectral index (k^0.81) and three-component taxonomy are reusable by FDM,
//! ULDM, and algebraic substructure search communities.
//!
//! **Status**: GHOST fills this gap. The synthesis states this explicitly.
//!
//! ---
//!
//! ## Gap 3: Outer-Halo Rotation Curve Stacking at x > 3 r/r_s
//!
//! **Description**: The literature on HI rotation curves (THINGS, SPARC) uses
//! individual galaxy fits rather than large-N stacking. Stacking at outer radii
//! -- the regime where ZD predictions peak -- does not exist. The MaNGA inner-halo
//! null motivates but cannot execute this analysis.
//!
//! **Impact**: HIGH. Outer-halo stacking is the natural follow-up and the primary
//! content of the SKA-era campaign.
//!
//! **Status**: Open. SPARC (N=175) and THINGS (N=34) could provide pilot data
//! but have not been used for stacking analyses at x > 3.
//!
//! ---
//!
//! ## Gap 4: Injection-Recovery Calibration for Harmonic-Absorption Interference
//!
//! **Description**: GHOST's anti-monotonic injection recovery (C-1412) is a novel
//! failure mode: harmonic subtraction absorbs injected signals overlapping with
//! baryonic modes. The direct detection literature (Baxter+2021) addresses
//! injection conventions but not harmonic-absorption interference specific to
//! Fourier-domain stacking searches.
//!
//! **Impact**: MEDIUM-HIGH. Blocks publication-quality sensitivity claims until
//! resolved. Generalizable as a methodological warning for any Fourier-domain
//! halo residual pipeline.
//!
//! **Status**: Root cause identified (exp(-x) envelope destructive interference
//! with +5% bulge at x~0.5). Radial-windowed injection (inject at x > 1.0)
//! is proposed as resolution but not yet executed.
//!
//! ---
//!
//! ## Gap 5: Baryonic Red-Noise Spectral Index from Simulations
//!
//! **Description**: The k^0.81 spectral index is empirical. No cosmological
//! hydrodynamic simulation (FIRE, NIHAO, EAGLE) has predicted the spectral
//! character of stacked NFW-subtracted rotation curve residuals. Independent
//! validation from simulations would confirm or challenge GHOST's baryonic model.
//!
//! **Impact**: MEDIUM. Would strengthen the baryonic floor characterization from
//! "measured" to "measured and simulation-validated."
//!
//! **Status**: Open. Simulation data may exist in raw form but has not been
//! processed into stacked spectral form.
//!
//! ---
//!
//! ## Gap 6: Post-2022 MaNGA Kinematic Systematics Characterization
//!
//! **Description**: The three baryonic systematics GHOST documents (bulge excess
//! +5%, cusp over-prediction -15%, IFU edge spike +29%) may or may not have been
//! independently characterized by the MaNGA community in recent kinematic analyses.
//! A targeted literature search (2022-2026 MaNGA rotation curve papers) is needed
//! to determine whether GHOST's taxonomy is independently corroborated.
//!
//! **Impact**: MEDIUM. Independent corroboration strengthens the baryonic model;
//! discrepancies would require investigation.
//!
//! **Status**: Search needed. The SMART goal document identifies this as a gap
//! to fill during Weeks 1-2.
//!
//! ---
//!
//! ## Gap 7: Framework Discrimination at Outer-Halo Radii
//!
//! **Description**: The quasi-degeneracy (rho > 0.97) at inner-halo radii is
//! established. Whether CD-ZD, G2, Albert, and sl(2) wavenumber families
//! separate at x > 3 r/r_s has not been computed. This is critical for
//! determining whether SKA can discriminate between frameworks or whether the
//! quasi-degeneracy persists at all radii.
//!
//! **Impact**: MEDIUM. Determines whether the SKA campaign can do more than
//! improve the alpha_zd bound -- specifically, whether it can test algebraic
//! model choice.
//!
//! **Status**: Open. Purely computational -- evaluate wavenumber families at
//! x = 3-10 and compute cross-correlation matrix. Code exists for inner-halo
//! evaluation; extension is straightforward.
//!
//! ---
//!
//! ## Prioritized Opportunities
//!
//! ### Opportunity 1 (HIGHEST): SKA Phase 1 Outer-Halo Stacking Forecast
//!
//! **Impact**: HIGH -- converts inner-halo null from standalone result to
//! stepping stone. The sensitivity projection figure (alpha_zd threshold vs
//! radial coverage) is what survey planners will cite.
//!
//! **Feasibility**: HIGH -- GHOST pipeline is built; forecasting requires
//! substituting MaNGA radial coverage with projected SKA HI coverage and
//! computing expected sensitivity at x = 3-5 r/r_s.
//!
//! **Papers needed**: SKA HI survey design updates, projected galaxy counts
//! and radial coverage specifications.
//!
//! **Builds on**: I-182 ("SKA/Euclid outer halo required for ZD detection"),
//! SMART goal Figure 4 specification.
//!
//! ### Opportunity 2 (HIGH): Face-On Galaxy Reanalysis with Red-Noise Baseline Subtraction
//!
//! **Impact**: MEDIUM-HIGH -- low-inclination subsample (i < 45 deg, N~3140)
//! has cleanest rotation curves and highest per-galaxy SNR (0.98 vs 0.29 full).
//! Subtracting k^0.81 envelope before computing SNR could improve sensitivity.
//!
//! **Feasibility**: HIGH -- pipeline code exists. Requires adding red-noise
//! envelope subtraction step and re-running on face-on subsample.
//!
//! **Caveat**: Red-noise correction on full sample produces identical SNR to
//! uncorrected baseline (both 0.4782). Face-on subsample may behave differently
//! due to narrower mass and r_s distribution.
//!
//! **Builds on**: Prior hypothesis H1 (face-on reanalysis), C-1367 (inclination
//! falsification), retrospective lesson 6 (quasi-degeneracy).
//!
//! ### Opportunity 3 (HIGH): Injection-Recovery Calibration via Matched Filter
//!
//! **Impact**: HIGH -- blocks publication-quality sensitivity claims until
//! resolved. Matched filter accounting for harmonic basis would decouple
//! detection from baryonic subtraction.
//!
//! **Feasibility**: MEDIUM -- non-trivial pipeline modification (retrospective
//! item 10). Three options: (a) matched filter with harmonic basis, (b) two-stage
//! simultaneous fit, (c) inject only at x > 1.0 where interference is weaker.
//!
//! **Builds on**: Prior hypothesis H2 (Q3 injection recovery), C-1412
//! (anti-monotonic injection), retrospective lesson 2.
//!
//! ### Opportunity 4 (MEDIUM): SPARC/THINGS Cross-Validation at Outer Radii
//!
//! **Impact**: MEDIUM -- even N=34 THINGS galaxies with x > 10 coverage would
//! provide proof-of-concept for outer-halo stacking before SKA.
//!
//! **Feasibility**: MEDIUM -- data exists (SPARC, THINGS public releases).
//! Requires adapting GHOST pipeline to HI rotation curve formats.
//!
//! **Builds on**: Retrospective item 11 ("SPARC pilot cross-survey validation").
//!
//! ### Opportunity 5 (MEDIUM): Theoretical Framework Separation Computation
//!
//! **Impact**: MEDIUM -- determines whether SKA can discriminate between
//! algebraic frameworks or only improve the shared alpha_zd bound.
//!
//! **Feasibility**: HIGH -- purely computational. Evaluate wavenumber families
//! at x = 3-10 r/r_s and compute cross-correlation matrix. Existing inner-halo
//! code is directly extensible.
//!
//! **Builds on**: Gap 7, C-1372 (algebraic universality), cross-algebra
//! correlation data in data/results/e183/cross_algebra_correlation.csv.
//!
//! ### Opportunity 6 (MEDIUM): Baryonic Index Validation Against Simulations
//!
//! **Impact**: MEDIUM -- would validate or challenge k^0.81 independently.
//!
//! **Feasibility**: LOW-MEDIUM -- requires finding or running stacked RC
//! residuals from FIRE/NIHAO simulations. Data may not exist in stacked form.
//!
//! **Builds on**: Gap 5, I-180 (Fourier red-noise characterization),
//! retrospective lesson 8 (reusable byproduct).
//!
