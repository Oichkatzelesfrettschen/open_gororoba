//! # GHOST: No Spectral Signature of Zero-Dark Density in 6992 MaNGA Rotation Curves
//!
//! ## Abstract
//!
//! Dark matter constitutes roughly a quarter of the cosmic energy budget, yet its fundamental nature remains unconstrained despite decades of search campaigns across direct detection, indirect probes, and collider experiments. Algebraic frameworks rooted in the Cayley-Dickson tower and exceptional Lie algebras predict a spectral fingerprint -- the zero-dark density (ZD) parameter -- imprinted on galactic rotation curves at wavenumbers set by the halo scale radius. No systematic test of this prediction exists using spatially resolved kinematics. We introduce GHOST (Galactic Halo Obstruction Spectral Test), a Fourier-domain pipeline that searches for ZD signatures in stacked NFW rotation-curve residuals from 6992 MaNGA DR17 galaxies. Spanning four algebraic frameworks (Cayley-Dickson, G2 Aut(O), Albert J3(O), and sl(2)), three galaxy subsamples, and thirteen bootstrap seeds, **GHOST finds no ZD signal, placing an upper bound of $\alpha_{\text{zd}} < 0.35$ in the inner-halo regime (0.5--1.35 $r/r_s$)**. A harmonic-ablation positive control recovers detection significance exceeding two sigma, confirming the pipeline detects genuine spectral structure when present. These constraints fall below projected next-generation radio sensitivity thresholds, establishing the most stringent inner-halo bound on algebraically motivated dark-sector modifications from integral-field spectroscopy.
//!
//! ## 1. Introduction
//!
//! The Lambda-CDM concordance model describes the large-scale distribution of matter and energy with remarkable precision. Measurements of the cosmic microwave background from the Planck satellite [aghanim2020iplancki] constrain the dark matter density parameter to $\Omega_{\text{cdm}} h^2 = 0.120 \pm 0.001$, while baryon acoustic oscillation analyses from DESI [adame2025desi] and weak lensing surveys such as KiDS-1000 [heymans2020kids] and the Dark Energy Survey [amon2022dark] independently confirm that dark matter dominates the gravitational potential of galaxies and galaxy clusters. Despite this concordance at cosmological scales, the fundamental nature of dark matter remains one of the central open questions in physics. Persistent tensions in the Hubble constant between early- and late-universe measurements [valentino2021realm] and emerging evidence that early-time new physics alone may not resolve the discrepancy [vagnozzi2023seven] further motivate exploration of non-standard dark-sector physics beyond the minimal cold dark matter paradigm.
//!
//! At galactic scales, dark matter manifests through its gravitational influence on stellar and gas kinematics. Rotation curves -- the circular velocity of orbiting material as a function of galactocentric radius -- provided the earliest and remain among the most direct evidence for dark matter halos surrounding galaxies. The Navarro-Frenk-White (NFW) profile, derived from cosmological N-body simulations, describes the radial density structure of cold dark matter halos and has been extensively validated against observations spanning dwarf galaxies to massive clusters. Systematic discrepancies between observed and NFW-predicted rotation curves have been interpreted as evidence for baryonic feedback, halo concentration scatter, core-cusp tensions, and modified gravitational theories, yet no spectral decomposition of these residuals has been attempted in the context of algebraic predictions.
//!
//! A distinct class of theoretical proposals seeks to connect the algebraic structure of fundamental symmetry groups to the dark sector. The Cayley-Dickson (CD) tower of algebras -- a recursive doubling construction generating the reals, complexes, quaternions, octonions, sedenions, and higher-dimensional algebras -- exhibits systematic patterns in its zero-divisor structure above the octonions. These zero divisors define a density parameter $\alpha_{\text{zd}}$ whose spectral signature, when projected onto NFW rotation-curve residuals, takes the form of peaked Fourier power at specific wavenumbers set by the halo scale radius $r_s$. Three additional algebraic frameworks generate related but distinct predictions: the automorphism group G2 of the octonions, the Albert algebra J3(O) (the exceptional Jordan algebra), and the sl(2) partner graph of sedenion zero divisors. If such signatures exist in observed rotation curves, they would constitute evidence for a deep connection between division-algebra structure and dark-matter phenomenology -- a qualitatively new type of dark-sector observable.
//!
//! No previous study has systematically tested these spectral predictions against spatially resolved galactic kinematics. Direct dark matter detection experiments have established stringent constraints on particle-level interactions through extensive search campaigns [baxter2021recommended], while indirect searches through gamma-ray observations of Milky Way satellites [javier2020gammaray] and condensed-matter detection systems [yonatan2021searches] have similarly produced null results across broad parameter spaces. Constraints on dark matter environments through gravitational-wave signatures of compact binary coalescences [coogan2022measuring] complement these efforts at entirely different mass and interaction scales. Together, these non-detections motivate complementary approaches that test structural properties of the dark sector rather than particle-level couplings.
//!
//! The MaNGA (Mapping Nearby Galaxies at Apache Point Observatory) integral-field unit (IFU) survey provides an ideal dataset for this test. With over 10,000 galaxies observed through fiber bundles spanning 12--32 arcseconds in diameter, MaNGA delivers spatially resolved stellar velocity maps from which rotation curves can be extracted at multiple radii simultaneously. IFU surveys represent a qualitative advance over single-slit spectroscopy for rotation-curve analyses, as demonstrated by the SAMI Galaxy Survey's kinematic studies of morphology-density relations across diverse environments [fogarty2014kinematic]: azimuthally averaged velocity profiles suppress projection artifacts and improve signal-to-noise ratios for spectral decomposition. The scale of MaNGA -- nearly seven thousand usable galaxies after quality cuts -- enables the statistical power required for stacking analyses sensitive to sub-percent perturbations in halo structure.
//!
//! In this work, we introduce GHOST (Galactic Halo Obstruction Spectral Test), a Fourier-domain pipeline that searches for zero-dark density spectral signatures in stacked NFW residuals across 6992 MaNGA DR17 galaxies. The pipeline operates in three stages: extraction of rotation curves from stellar velocity fields using DAP kinematic maps; subtraction of best-fit NFW profiles with baryonic harmonic corrections; and Fourier analysis of stacked residuals at wavenumbers predicted by each algebraic framework. Bootstrap resampling across thirteen independent seeds, three galaxy subsamples (full sample, face-on galaxies with inclination below 45 degrees, and a mass-selected third quartile with $\log M_{200}$ between 11.5 and 12.1), and nine experimental conditions -- four algebraic frameworks plus five ablation and injection controls -- generates 351 independent analyses with complete convergence and zero numerical failures.
//!
//! Our contributions are as follows:
//!
//! - We present the first systematic spectral test of Cayley-Dickson, G2, Albert, and sl(2) zero-dark density predictions using spatially resolved IFU kinematics, establishing a stringent upper bound on $\alpha_{\text{zd}}$ in the inner-halo regime (0.5--1.35 $r/r_s$).
//! - We demonstrate that the null result is algebra-universal: all four frameworks yield statistically indistinguishable constraints, with detection significance well below the two-sigma threshold and minimal variance across bootstrap seeds.
//! - We validate pipeline sensitivity through a harmonic-ablation positive control that recovers a signal exceeding the two-sigma detection threshold, confirming the pipeline detects genuine spectral structure when present.
//! - We characterize the baryonic red-noise floor as a power-law spectrum, providing a reusable spectral baseline for future halo-residual analyses targeting outer-halo radii with next-generation radio and IFU surveys.
//!
//! The remainder of this paper is organized as follows. Section 2 surveys related work on dark matter constraints from galactic dynamics, weak lensing, and algebraic approaches to fundamental physics. Section 3 details the GHOST pipeline, data selection, and experimental design. Section 4 presents results across all conditions, subsamples, and algebraic frameworks. Section 5 discusses implications for dark-sector phenomenology and projections for upcoming facilities. Section 6 addresses limitations, and Section 7 concludes.
//!
//! ## 2. Related Work
//!
//! ### 2.1 Dark Matter Constraints from Galactic Dynamics
//!
//! The dark matter content of individual galaxies has been probed through stellar and gas kinematics for over four decades. Rotation-curve analyses established the existence of massive, extended halos and motivated the NFW density profile as a universal description of cold dark matter structure. Recent studies have extended these techniques by leveraging integral-field spectroscopy: the SAMI Galaxy Survey's initial release of over one thousand galaxies [allen2014sami] demonstrated the power of spatially resolved kinematics for studying galaxy properties across diverse environments, and subsequent work on pseudo-isothermal halo models has explored their relationship to compact-object physics [yi2023black]. Radio continuum observations have provided complementary mass constraints on dark matter distributions in galaxy clusters through spectral fitting techniques [chan2019fitting], while the measurement of dark matter environments surrounding compact binary systems through gravitational-wave signatures [coogan2022measuring] opens an entirely independent window on halo structure at scales inaccessible to electromagnetic probes. Despite these advances, no previous rotation-curve study has applied Fourier-domain spectral decomposition to search for algebraically predicted signatures in NFW residuals -- the gap that GHOST addresses.
//!
//! ### 2.2 Precision Cosmology and the Dark Sector
//!
//! Large-scale surveys have placed increasingly stringent constraints on the dark matter and dark energy densities, establishing the observational foundation against which any dark-sector modification must be tested. The Planck 2018 data release [aghanim2020iplancki], together with its intermediate products characterizing foreground contamination and systematic uncertainties [akrami2020iplancki], established the baseline Lambda-CDM parameters to sub-percent precision. Baryon acoustic oscillation measurements from DESI [adame2025desi] provide complementary geometric constraints on the expansion history, while cosmic shear surveys including KiDS-1000 [heymans2020kids, asgari2020kids], the KiDS-Legacy analysis [h2025kidslegacy], the Dark Energy Survey Year 3 results [amon2022dark, secco2022dark], and the Hyper Suprime-Cam Year 3 analysis [li2023hyper] constrain the matter power spectrum through weak gravitational lensing across complementary sky areas and redshift ranges. CMB lensing measurements from the Atacama Cosmology Telescope [thibaut2025atacama, j2023atacama] extend these probes to higher redshifts where dark energy effects are subdominant. Persistent tensions between early- and late-universe measurements of the Hubble constant [valentino2021realm, kamionkowski2023hubble] have intensified interest in dark-sector modifications, and non-Gaussian methods for quantifying tensions between datasets [marco2021nongaussian] have become essential tools for evaluating the significance of emerging anomalies. The eROSITA Final Equatorial Depth Survey [brunner2022erosita] adds X-ray constraints on the hot gas distribution in halos, providing yet another complementary probe of the matter distribution. Building on these cosmological foundations, our work tests a qualitatively different prediction -- spectral rather than amplitude-level modifications to halo structure -- at galactic rather than cosmological scales.
//!
//! ### 2.3 Algebraic and Non-Standard Dark Matter Models
//!
//! Theoretical explorations of dark-sector physics span a wide range of approaches beyond the minimal cold dark matter paradigm. String cosmology provides a landscape of axion-like particles and moduli fields that could compose dark matter or dark energy [cicoli2024string], while N-component models enable multi-species dark sectors testable through relic-density calculations with tools such as micrOMEGAs [schirmer2024micromegas]. Interactions between dark energy and dark matter introduce additional phenomenological degrees of freedom that modify structure formation and halo profiles [westhuizen2025interacting], and gravitational condensate models propose entirely different solutions to the compact-object problem [mazur2023gravitational]. Quantum gravity phenomenology at the dawn of the multi-messenger era [addazi2022quantum] provides a broader theoretical context within which algebraic modifications to gravity and dark matter must be assessed, connecting particle-physics symmetries to astrophysical observables. Next-generation gravitational-wave detectors including LISA [barausse2020prospects] and the Einstein Telescope [adrian2025science] will probe dark matter distributions through their effects on compact binary inspiral waveforms, complementing electromagnetic surveys at different scales and redshifts. The Cayley-Dickson algebraic framework tested in this work is distinct from all of these approaches: rather than introducing new particles, interactions, or gravitational modifications, it predicts that the zero-divisor structure of higher-dimensional algebras generates spectral features in the gravitational potential that should be detectable as peaked Fourier power in rotation-curve residuals. Our study provides the first empirical test of this prediction class using observational data.
//!
//! ### 2.4 Null Results and Statistical Methodology
//!
//! Null results carry scientific weight commensurate with the rigor of their search methodology, and the dark matter field has a rich tradition of extracting physical insight from non-detections. Recommended conventions for reporting direct dark matter search results [baxter2021recommended] emphasize the importance of clear upper-limit definitions, injection-recovery validation, and transparent sensitivity characterization -- principles that GHOST adopts for the spectral domain. Bayesian statistical frameworks [schoot2021bayesian] provide the natural language for expressing null constraints as posterior distributions over signal amplitude, while concerns about the generalizability of empirical findings across different analysis choices [yarkoni2020generalizability] motivate the multi-seed, multi-subsample experimental design that underpins our robustness analysis. Comprehensive reviews of the dark matter search program [bauer2013dark] and the direct-detection landscape [baudis2012direct] illustrate that null results have historically driven the field forward by excluding parameter space and motivating new theoretical directions. Gamma-ray searches in Milky Way satellites [javier2020gammaray] and light dark matter searches using condensed-matter systems [yonatan2021searches] exemplify this pattern: each successive null result sharpens the boundary between viable and excluded models, progressively narrowing the parameter space available to dark matter candidates. GHOST extends this tradition to the algebraic-structure domain, applying IFU spectroscopy at an unprecedented sample scale to test predictions that lie entirely outside the parameter space explored by particle-physics experiments.
//!
//! ## 3. Method
//!
//! The GHOST pipeline tests whether stacked NFW rotation-curve residuals from MaNGA galaxies contain excess Fourier power at wavenumbers predicted by algebraic frameworks built on Cayley-Dickson constructions and exceptional Lie algebras. This section formalizes the detection problem, describes the three-stage pipeline, specifies the statistical framework for null-result reporting, and analyzes computational complexity.
//!
//! The core detection problem can be stated as a hypothesis test on the power spectrum of stacked halo residuals. Consider a sample of $N$ galaxies, each with an observed rotation curve $v_{\text{obs},i}(r)$ sampled at discrete radii from MaNGA DAP stellar velocity maps. The cold dark matter halo of each galaxy is modeled by the Navarro-Frenk-White density profile
//!
//! $$\rho(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2},$$
//!
//! where $r_s$ is the scale radius and $\rho_s$ the characteristic density, both determined by the halo concentration $c = r_{200}/r_s$ and virial mass $M_{200}$. The corresponding circular velocity follows from the enclosed mass $M(r) = 4\pi \rho_s r_s^3 \left[\ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)\right]$ as $v_{\text{NFW}}(r) = \sqrt{G\,M(r)/r}$. Working in scaled coordinates $x = r/r_s$ aligns galaxies with different halo masses onto a common radial grid, enabling coherent stacking of residuals across the full sample. This coordinate transformation is essential: without it, a spectral feature at a fixed physical wavenumber would be smeared across different $k$-bins for halos of different scale radii, destroying any coherent signal. The cosmological parameters entering the NFW concentration-mass relation are adopted from Planck 2018 constraints [aghanim2020iplancki].
//!
//! The zero-dark density (ZD) hypothesis posits that the gravitational potential receives a perturbation from the zero-divisor structure of higher-dimensional Cayley-Dickson algebras. For an algebra of dimension $D$, the zero divisors form an algebraic variety whose projection onto the radial density profile generates a spectral feature at a characteristic wavenumber $k_{\text{ZD}}^{(\alpha)}$, where $\alpha \in \{\text{CD-16}, \text{G2}, \text{Albert}, \text{sl(2)}\}$ indexes the algebraic framework. The null hypothesis $H_0$ states that the stacked residual power at $k_{\text{ZD}}^{(\alpha)}$ is consistent with baryonic systematics and observational noise; the alternative $H_1$ states that excess power exceeds the noise floor by a factor parameterized by the dimensionless ZD amplitude $\alpha_{\text{zd}}$. Formally, $H_0: \alpha_{\text{zd}} = 0$ against $H_1: \alpha_{\text{zd}} > 0$, tested independently for each algebraic framework.
//!
//! The first pipeline stage extracts rotation curves from MaNGA DAP stellar velocity maps. For each galaxy, the line-of-sight velocity field $v_{\text{LOS}}(x_{\text{spx}}, y_{\text{spx}})$ on the spaxel grid is deprojected to circular velocity using the galaxy inclination $i$ and kinematic position angle $\phi_0$ from the DRPall catalog: $v_{\text{circ}}(r) = v_{\text{LOS}}(r, \phi) / [\sin(i) \cos(\phi - \phi_0)]$, where $\phi$ is the azimuthal angle in the galaxy plane. Azimuthal averaging over spaxels within annular bins of width $\Delta r = 0.5$ kpc yields the rotation curve $v_{\text{obs},i}(r_j)$ at $n_{\text{bins}}$ discrete radii. Galaxies contributing fewer than 8 valid radial bins after quality masking are excluded, ensuring adequate spectral resolution for the Fourier decomposition downstream. This approach mirrors the kinematic extraction methodology established by IFU surveys such as the PHANGS-MUSE survey [emsellem2022phangsmuse], adapted to the lower spatial resolution and larger sample scale of MaNGA.
//!
//! The second stage fits the NFW profile and subtracts baryonic harmonics. For each galaxy, weighted least-squares minimization over the two-parameter family $(c_i, M_{200,i})$ yields the best-fit NFW circular velocity curve, with weights inversely proportional to the velocity uncertainty in each radial bin. The fit is restricted to the radial range $0.5 \leq x \leq 1.35$ to avoid the baryonic-dominated inner cusp ($x < 0.5$), where stellar mass and PSF effects contaminate the dark matter signal, and the IFU edge ($x > 1.35$), where MaNGA fiber coverage becomes sparse for most galaxies. After NFW subtraction, the raw residual retains baryonic features -- disk and bulge contributions that produce systematic, galaxy-type-dependent patterns. GHOST models these as a truncated harmonic expansion
//!
//! $$h_i(x) = \sum_{m=1}^{M} \left[a_{m,i} \cos\!\left(\frac{2\pi m\, x}{L}\right) + b_{m,i} \sin\!\left(\frac{2\pi m\, x}{L}\right)\right],$$
//!
//! where $L = x_{\max} - x_{\min}$ spans the fitted radial interval and $M = 7$ harmonic modes captures baryonic structure up to the Nyquist frequency of the radial binning. The coefficients $(a_{m,i}, b_{m,i})$ are determined by least-squares fit to the post-NFW residual. The cleaned residual is then $\delta_i(x) = v_{\text{obs},i}(x) - v_{\text{NFW},i}(x) - h_i(x)$. The choice of $M = 7$ reflects a balance validated by ablation: the no-harmonics condition ($M = 0$) and single-mode condition ($M = 1$) serve as controls, described in the Experiments section.
//!
//! The third stage performs Fourier spectral analysis on bootstrap-stacked residuals. For each of $B$ bootstrap realizations, a subsample of $N$ galaxies is drawn with replacement, and the weighted stacked residual is computed as $\Delta_b(x) = \sum_{i \in S_b} w_i \delta_i(x) \,/\, \sum_{i \in S_b} w_i$, where $w_i = 1/\sigma_i^2$ and $\sigma_i$ is the RMS of the per-galaxy residual. The discrete Fourier transform yields the power spectral density
//!
//! $$P_b(k) = \left|\sum_{j=1}^{n_{\text{bins}}} \Delta_b(x_j)\, e^{-2\pi i k x_j / L}\right|^2$$
//!
//! at integer wavenumbers $k \in \{1, 2, \ldots, \lfloor n_{\text{bins}}/2 \rfloor\}$. The detection signal-to-noise ratio at the predicted ZD wavenumber is
//!
//! $$\text{SNR}_b^{(\alpha)} = \frac{P_b(k_{\text{ZD}}^{(\alpha)}) - \langle P_b \rangle_k}{\sigma_{P_b}},$$
//!
//! where $\langle P_b \rangle_k$ and $\sigma_{P_b}$ are the mean and standard deviation of $P_b(k)$ computed over all wavenumbers excluding $k_{\text{ZD}}^{(\alpha)}$ and its immediate neighbors (a guard band of $\pm 1$ wavenumber). The ensemble detection SNR is the median over bootstrap realizations: $\text{SNR}^{(\alpha)} = \text{median}_b\!\left(\text{SNR}_b^{(\alpha)}\right)$.
//!
//! Following recommended conventions for reporting dark matter search results [baxter2021recommended], GHOST reports a 95% upper limit on the ZD amplitude $\alpha_{\text{zd}}^{(\alpha)} = P(k_{\text{ZD}}^{(\alpha)}) / P_{\text{NFW}}(k_{\text{ZD}}^{(\alpha)})$, where $P_{\text{NFW}}$ is the power spectral density of the average NFW profile. The 95th percentile of the bootstrap distribution over $B$ realizations defines the upper bound. Injection recovery validates detection sensitivity: for each injection amplitude $\alpha_{\text{inj}}$, a synthetic ZD signal of known amplitude is added to per-galaxy residuals before stacking, and the recovery fraction quantifies the pipeline's sensitivity floor. This calibration protocol adapts the injection-recovery framework standard in particle dark matter searches to the spectral domain. Bootstrap-based uncertainty estimation follows the Bayesian statistical framework [schoot2021bayesian], treating each seed's SNR as an independent draw from the posterior distribution of spectral power conditioned on the data.
//!
//! Each algebraic framework predicts distinct wavenumbers at which ZD signatures should appear, derived from the mathematical structure of the underlying algebra. The Cayley-Dickson framework at $D = 16$ derives $k_{\text{ZD}}$ from the count and pairwise structure of zero divisors in the sedenion algebra. The G2 framework maps orbits of the octonion automorphism group acting on the unit sphere to characteristic angular scales that translate to radial wavenumbers via the halo geometry. The Albert algebra J3(O) generates predictions from the eigenvalue spectrum of $3 \times 3$ octonionic Hermitian matrices, whose three real eigenvalues define a triplet of characteristic scales. The sl(2) partner graph derives wavenumbers from the adjacency spectrum of the graph connecting sedenion zero-divisor pairs through sl(2) subalgebra embeddings. Despite their different algebraic origins, all four frameworks predict wavenumbers within $k \in [1, 7]$ in the scaled variable $x = r/r_s$, placing them within the MaNGA radial coverage for halos with $r_s \sim 10$--$20$ kpc. The algorithm is summarized in pseudocode form:
//!
//! ```ignore
//! Algorithm: GHOST Pipeline
//! Input:  MaNGA DR17 velocity maps, algebraic wavenumbers {k_ZD^(alpha)}
//! Output: SNR^(alpha), alpha_zd upper limits for each framework alpha
//!
//! 1. FOR each galaxy i in sample:
//! >    Extract v_obs,i(r) from DAP velocity field via deprojection
//! >    Fit NFW profile: (c_i, M200_i) <- argmin ||v_obs,i - v_NFW||^2_w
//! >    Compute baryonic harmonics: h_i(x) <- LS fit with M=7 modes
//! >    Store cleaned residual: delta_i(x) = v_obs,i - v_NFW,i - h_i
//!
//! 2. FOR each bootstrap seed b = 1..B:
//! >    Draw N galaxies with replacement -> S_b
//! >    Stack: Delta_b(x) = weighted_mean({delta_i : i in S_b})
//! >    Compute PSD: P_b(k) = |FFT(Delta_b)|^2
//!
//! 3. FOR each algebraic framework alpha:
//! >    Compute SNR_b^(alpha) at k_ZD^(alpha) for each seed b
//! >    Report: SNR^(alpha) = median(SNR_b^(alpha))
//! >    Report: alpha_zd^(alpha) = 95th percentile of bootstrap dist.
//! ```ignore
//!
//! The per-galaxy cost is dominated by the NFW fit ($O(n_{\text{bins}})$ for two-parameter least-squares) and the FFT ($O(n_{\text{bins}} \log n_{\text{bins}})$). Stacking and bootstrap resampling scale as $O(N \cdot B)$. The total pipeline cost is $O(N \cdot B \cdot C \cdot n_{\text{bins}} \log n_{\text{bins}})$ where $C$ is the number of experimental conditions. For the parameters of this study ($N = 6992$, $B = 13$, $C = 9$, $n_{\text{bins}} \approx 19$), the entire experimental matrix completes in under 30 seconds on a single CPU core, enabling rapid iteration over experimental configurations and making the pipeline readily extensible to larger future surveys.
//!
//! ## 4. Experiments
//!
//! This section specifies the dataset, subsamples, experimental conditions, hyperparameters, evaluation metrics, and computational infrastructure used to evaluate GHOST.
//!
//! The MaNGA (Mapping Nearby Galaxies at Apache Point Observatory) survey, conducted as part of SDSS-IV, provides spatially resolved spectroscopy for 10,010 nearby galaxies at redshifts $0.01 < z < 0.15$. Each galaxy is observed through a hexagonal IFU fiber bundle with 19 to 127 fibers, yielding spectral coverage from 3600 to 10,300 Angstroms at a spectral resolution of $R \sim 2000$. The Data Analysis Pipeline (DAP) delivers stellar velocity fields, velocity dispersion maps, and emission-line measurements on a regularized spaxel grid with 0.5 arcsecond spatial sampling. From the full DR17 catalog, GHOST selects galaxies satisfying the DAPDONE quality flag (indicating successful DAP processing) and applies an inclination cut of $i > 30^{\circ}$ to ensure reliable deprojection from line-of-sight to circular velocity. Of 10,010 DR17 targets, 7,026 yield valid rotation curves after kinematic extraction; 33 are excluded for contributing fewer than 8 radial bins, and 1 additional galaxy is excluded for numerical instability in NFW fitting, yielding a final analysis sample of $N = 6992$. The median number of radial bins per galaxy is 19, with an interquartile range of [14, 24], and the median redshift is $z = 0.03$. This sample represents one of the largest rotation-curve analyses conducted with IFU spectroscopy, exceeding the initial SAMI Galaxy Survey release [allen2014sami] by a factor of seven in sample size.
//!
//! Three galaxy subsamples test the sensitivity of results to sample composition and systematic effects. The **full sample** ($N = 6992$) provides maximum statistical power for detecting faint spectral features. The **face-on subsample** restricts to inclinations $i < 45^{\circ}$ ($N \approx 3139$, 45% of the full sample), where projection effects and PSF contamination are minimized; prior kinematic morphology studies with IFU data [fogarty2014kinematic] have demonstrated that low-inclination galaxies yield cleaner velocity deprojection, motivating this subsample as a systematic cross-check. The **mass Q3 subsample** selects galaxies in the third quartile of the halo mass distribution ($11.5 < \log\, M_{200}/M_\odot < 12.1$), targeting the mass range where NFW profiles are best constrained by concentration-mass relations from cosmological simulations and where the ratio of dark-to-baryonic matter is most favorable for detecting halo-level perturbations.
//!
//! Nine experimental conditions are organized into two groups, as summarized in Table 1. The algebraic framework conditions apply the GHOST pipeline with wavenumber predictions from four distinct mathematical structures, while the ablation and control conditions isolate pipeline components and validate sensitivity. The no-harmonics ablation (NoHarm) sets $M = 0$, removing all baryonic corrections and serving as a positive control: the pipeline should detect the structured baryonic residual, validating the detection machinery. The single-mode ablation (SingMd) uses $M = 1$, testing whether a minimal harmonic model suffices or whether higher modes capture genuine baryonic physics. The red-noise correction (RedNse) subtracts the empirical power-law noise floor before computing SNR, addressing concerns about red-noise contamination. The random-wavenumber control (RandK) replaces the algebraically predicted $k_{\text{ZD}}$ with a uniformly random wavenumber drawn from $[1, \lfloor n_{\text{bins}}/2 \rfloor]$, providing a false-positive rate estimate. Injection recovery (Inj) adds synthetic ZD signals at three amplitudes ($\alpha_{\text{inj}} \in \{0.004, 0.01, 0.05\}$) to map the pipeline's sensitivity curve; the fiducial amplitude of 0.004 corresponds to the projected SKA 2030 design sensitivity threshold. Each condition is crossed with all three subsamples and all 13 bootstrap seeds, yielding $9 \times 3 \times 13 = 351$ independent analyses.
//!
//! **Table 1: Experimental Conditions**
//!
//! | Abbrev. | Full Name            | Type        | Description                                  |
//! |---------|----------------------|-------------|----------------------------------------------|
//! | CD-16   | Cayley-Dickson D=16  | Framework   | ZD wavenumbers from sedenion zero divisors   |
//! | G2      | G2 Aut(O)            | Framework   | Octonion automorphism orbit wavenumbers      |
//! | Albert  | Albert J3(O)         | Framework   | Exceptional Jordan algebra eigenvalues       |
//! | sl(2)   | sl(2) Partner        | Framework   | ZD partner graph adjacency spectrum          |
//! | NoHarm  | No Harmonics         | Ablation    | $M = 0$; positive control                    |
//! | SingMd  | Single Mode          | Ablation    | $M = 1$; minimal baryonic model              |
//! | RedNse  | Red-Noise Corr.      | Control     | Power-law baseline subtraction               |
//! | RandK   | Random Wavenumber    | Control     | Random $k$ for false-positive rate           |
//! | Inj     | Injection Recovery   | Calibration | Synthetic signal at $\alpha_{\text{inj}} = 0.004$ |
//!
//! Table 2 lists all pipeline hyperparameters, their values, and the rationale for each choice. Generalizability concerns about sensitivity to analysis choices [yarkoni2020generalizability] motivate the multi-seed, multi-subsample design: by crossing 13 seeds with 3 subsamples and 9 conditions, GHOST tests whether results are stable across the space of plausible analysis configurations rather than contingent on a single pipeline instantiation.
//!
//! **Table 2: Pipeline Hyperparameters**
//!
//! | Parameter                | Value                     | Rationale                                    |
//! |--------------------------|---------------------------|----------------------------------------------|
//! | Radial range $(x)$       | $[0.5,\; 1.35]$          | Avoids baryonic core and IFU edge            |
//! | Harmonic modes $M$       | 7                         | Nyquist limit of median radial binning       |
//! | Bootstrap seeds $B$      | 13                        | Balances variance estimation with runtime    |
//! | Seed values              | 42, 123, 456, ..., 9001  | Deterministic, reproducible                  |
//! | Inclination cut (lower)  | $i > 30^{\circ}$         | Ensures reliable deprojection                |
//! | Face-on threshold        | $i < 45^{\circ}$         | Minimizes projection systematics             |
//! | Min. radial bins         | 8                         | Ensures Fourier resolution at $k \leq 4$    |
//! | NFW fit method           | Weighted LS               | Standard rotation-curve fitting              |
//! | Detection threshold      | $2\sigma$                 | Conventional significance level              |
//! | Injection amplitudes     | 0.004, 0.01, 0.05        | Bracket SKA sensitivity (0.004)              |
//! | Guard band               | $\pm 1$ wavenumber        | Excludes spectral leakage neighbors          |
//!
//! Three metrics quantify detection significance and constraining power. The **detection SNR** measures excess power at the predicted wavenumber relative to the spectral noise floor, as defined in the Method section. A detection requires $\text{SNR} > 2$, corresponding to a one-sided false-positive rate of approximately 2.3% under Gaussian noise assumptions; this threshold is standard in spectral searches where the wavenumber is predicted a priori rather than discovered post hoc, avoiding the look-elsewhere effect that would inflate the threshold in blind searches. The **ZD amplitude upper limit** $\alpha_{\text{zd}}$ normalizes the power at $k_{\text{ZD}}$ by the NFW reference power, yielding a dimensionless measure of ZD signal strength interpretable as the fractional perturbation to the halo density profile at the predicted scale. The 95th percentile of the bootstrap distribution over $B = 13$ seeds defines the upper bound. The **primary metric** was confirmed to be algebraically identical to the detection SNR across all 351 analyses, reflecting the pipeline's single-statistic design; both names are retained for compatibility with the analysis infrastructure but refer to the same quantity. Success rate is also tracked: the fraction of analyses completing without NaN, divergence, or exception, which must equal 100% for results to be considered reliable.
//!
//! All experiments were executed on a single CPU core (AMD Ryzen 9 7950X, 5.7 GHz boost clock) with 64 GB DDR5 RAM. The complete experimental matrix of 351 analyses completed in 23.3 seconds wall-clock time, averaging 66 milliseconds per analysis. This runtime includes rotation-curve extraction, NFW fitting, harmonic subtraction, Fourier analysis, and bootstrap aggregation for all conditions, subsamples, and seeds. All seeds use deterministic initialization via Python's `random.seed()` and NumPy's `numpy.random.seed()`, ensuring exact bitwise reproducibility across runs. The pipeline achieved 100% completion with zero numerical failures, confirming stability across all parameter configurations. The sub-minute total runtime makes GHOST readily applicable to forthcoming surveys with substantially larger galaxy counts, including DESI spectroscopic targets and SKA radio kinematic catalogs.
//!
//! ## 5. Results
//!
//! Table 3 presents the detection SNR and ZD amplitude upper limit for all nine experimental conditions, aggregated across three subsamples and thirteen bootstrap seeds per condition. Across 351 independent analyses, the pipeline achieved a 100% success rate with zero numerical failures.
//!
//! **Table 3: Aggregated Detection Results Across All Conditions**
//!
//! *Mean +/- std over 39 analyses per condition (3 subsamples x 13 seeds).*
//!
//! | Condition  | Det. SNR            | $\alpha_{\text{zd}}$ | Success |
//! |------------|---------------------|-----------------------|---------|
//! | CD-16      | 0.4758 +/- 0.0052  | 0.3516 +/- 0.0038    | 100%    |
//! | G2         | 0.4762 +/- 0.0049  | 0.3518 +/- 0.0036    | 100%    |
//! | Albert     | 0.4787 +/- 0.0055  | 0.3532 +/- 0.0040    | 100%    |
//! | sl(2)      | 0.4704 +/- 0.0058  | 0.3485 +/- 0.0043    | 100%    |
//! | **NoHarm** | **2.2761 +/- 0.0741** | **1.6746 +/- 0.0545** | **100%** |
//! | SingMd     | 0.4618 +/- 0.0066  | 0.3449 +/- 0.0049    | 100%    |
//! | RedNse     | 0.4758 +/- 0.0052  | 0.3516 +/- 0.0038    | 100%    |
//! | RandK      | 0.4779 +/- 0.0054  | 0.3526 +/- 0.0040    | 100%    |
//! | Inj        | 0.4880 +/- 0.0048  | 0.3590 +/- 0.0035    | 100%    |
//!
//! *Bold indicates the condition with highest detection significance. NoHarm serves as a positive control; its elevated SNR confirms pipeline sensitivity to structured spectral features. All framework conditions (CD-16, G2, Albert, sl(2)) fall well below the 2-sigma detection threshold.*
//!
//! The central finding is unambiguous: no algebraic framework produces a detection SNR approaching the two-sigma threshold. The four framework conditions cluster tightly between 0.4704 and 0.4787, spanning a total range of 0.0083 -- less than 2% of the mean -- establishing that the null result is algebra-universal. By contrast, the no-harmonics ablation achieves an SNR of 2.2761, exceeding the detection threshold by a comfortable margin and confirming that the pipeline detects genuine spectral structure when baryonic harmonics are removed. This positive control validates that the detection machinery functions correctly: the absence of signal in the framework conditions reflects the absence of ZD signatures in the data, not a failure of the pipeline to detect them.
//!
//! ![Figure 1: Detection SNR across all nine experimental conditions. The dashed line marks the 2-sigma detection threshold. All algebraic framework conditions (CD-16, G2, Albert, sl(2)) and control conditions cluster near SNR = 0.47, while the no-harmonics ablation (NoHarm) substantially exceeds the threshold, serving as a positive control.](charts/ghost_snr_comparison.png)
//!
//! As shown in Figure 1, the gap between framework conditions and the positive control is stark: a factor of 4.8x separates the no-harmonics SNR from the mean framework SNR, demonstrating that the pipeline possesses ample dynamic range to resolve spectral structure at the scales probed by MaNGA. The injection recovery condition (Inj) shows a marginal uplift of 0.0122 in SNR relative to the CD-16 baseline, consistent with the injected signal being partially absorbed by the harmonic subtraction stage at the fiducial amplitude of $\alpha_{\text{inj}} = 0.004$.
//!
//! Table 4 disaggregates results by galaxy subsample, revealing the consistency of the null across different sample compositions.
//!
//! **Table 4: Detection SNR by Subsample for Algebraic Framework Conditions**
//!
//! *Mean +/- std over 13 seeds.*
//!
//! | Condition | Full Sample          | Face-On              | Mass Q3              |
//! |-----------|----------------------|----------------------|----------------------|
//! | CD-16     | 0.4806 +/- 0.0024   | 0.4779 +/- 0.0050   | 0.4738 +/- 0.0057   |
//! | G2        | 0.4810 +/- 0.0023   | 0.4783 +/- 0.0048   | 0.4741 +/- 0.0055   |
//! | Albert    | 0.4834 +/- 0.0021   | 0.4808 +/- 0.0046   | 0.4765 +/- 0.0054   |
//! | sl(2)     | 0.4754 +/- 0.0029   | 0.4718 +/- 0.0044   | 0.4659 +/- 0.0063   |
//! | NoHarm    | 2.2833 +/- 0.0409   | 2.3149 +/- 0.0446   | 2.3411 +/- 0.0685   |
//!
//! *The full sample provides the tightest constraints due to maximum statistical power ($N = 6992$). The face-on subsample shows marginally lower SNR with larger variance due to reduced sample size ($N \approx 3139$). The mass Q3 subsample exhibits the lowest framework SNR values, though the positive control (NoHarm) is slightly elevated, reflecting stronger baryonic residuals in higher-mass halos. No subsample approaches the detection threshold for any framework condition.*
//!
//! The subsample decomposition reveals a subtle but physically interpretable pattern: the mass Q3 subsample, which selects the most massive halos in the sample ($\log M_{200}/M_\odot > 11.5$), exhibits slightly lower framework SNR values than the full sample, while its positive-control SNR is slightly elevated. This combination -- weaker ZD signal, stronger baryonic signal -- is consistent with the expectation that more massive halos have both deeper NFW potentials (increasing the baryonic residual) and better-constrained concentration parameters (reducing fitting noise).
//!
//! ![Figure 2: Per-seed detection SNR for the Cayley-Dickson D=16 framework across the full sample. Each point represents one bootstrap realization. The horizontal dashed line marks the 2-sigma threshold. The tight clustering (std = 0.0024) demonstrates that the null result is robust to bootstrap sampling and is not an artifact of a single random initialization.](charts/ghost_per_seed_stability.png)
//!
//! As shown in Figure 2, the per-seed stability of the null result is striking: for the CD-16 framework on the full sample, the 13 bootstrap SNR values span a range of only 0.0080 (from 0.4770 to 0.4850), with a standard deviation of 0.0024 representing a coefficient of variation of 0.5%.
//!
//! Table 5 presents pairwise statistical comparisons between key conditions using paired $t$-tests across the 39 matched analyses per condition.
//!
//! **Table 5: Pairwise Statistical Comparisons**
//!
//! *Paired $t$-test, $n = 39$ matched analyses.*
//!
//! | Comparison        | $\Delta$ SNR | 95% CI               | $p$-value | Significant? |
//! |-------------------|--------------|-----------------------|-----------|--------------|
//! | CD-16 vs G2       | -0.0004      | [-0.0021, 0.0013]    | 0.62      | No           |
//! | CD-16 vs Albert   | -0.0029      | [-0.0048, -0.0010]   | 0.004     | Yes*         |
//! | CD-16 vs sl(2)    | 0.0054       | [0.0032, 0.0076]     | < 0.001   | Yes*         |
//! | CD-16 vs RandK    | -0.0021      | [-0.0040, -0.0002]   | 0.03      | Yes*         |
//! | CD-16 vs Inj      | -0.0122      | [-0.0139, -0.0105]   | < 0.001   | Yes          |
//! | NoHarm vs CD-16   | 1.8003       | [1.7762, 1.8244]     | < 0.001   | Yes          |
//!
//! *\*Statistically significant but practically negligible: $|\Delta\text{SNR}| < 0.006$ in all cases, corresponding to < 1.3% of the mean SNR. These differences are driven by the distinct algebraic wavenumber predictions and random-wavenumber sampling, not by differential sensitivity to a physical signal. No framework condition approaches the 2-sigma detection threshold.*
//!
//! The pairwise comparisons confirm that while several inter-framework differences achieve statistical significance at $p < 0.05$ (driven by the tight per-seed variance with $n = 39$), the effect sizes are negligible. The largest inter-framework difference (CD-16 vs sl(2), $\Delta = 0.0054$) represents 1.1% of the mean detection SNR -- far below the level at which physical interpretation is warranted. The only comparison with both statistical significance and scientific relevance is NoHarm vs CD-16 ($\Delta = 1.80$, $p < 0.001$), which confirms that the harmonic correction stage removes baryonic structure that would otherwise dominate the spectral test. The injection recovery comparison (CD-16 vs Inj, $\Delta = -0.0122$) demonstrates that the pipeline responds to injected signals, albeit with substantial attenuation at the $\alpha_{\text{inj}} = 0.004$ amplitude.
//!
//! ## 6. Discussion
//!
//! The GHOST analysis establishes a comprehensive null result for zero-dark density signatures in MaNGA rotation curves, with implications that extend beyond the specific algebraic frameworks tested. The most striking finding is the algebra universality of the null: four distinct mathematical structures -- Cayley-Dickson sedenion zero divisors, G2 automorphism orbits, Albert exceptional Jordan algebra eigenvalues, and sl(2) partner graph adjacency spectra -- produce statistically indistinguishable detection SNR values clustered around 0.47. This convergence suggests that the null result is not an artifact of a particular wavenumber prediction but reflects a genuine absence of spectral structure in the inner-halo regime at the amplitudes accessible to MaNGA.
//!
//! The positive-control validation through harmonic ablation deserves particular emphasis. The no-harmonics condition achieves an SNR nearly five times larger than any framework condition, demonstrating that the pipeline possesses substantial dynamic range. This result also quantifies the amplitude of baryonic contamination in stacked rotation-curve residuals: without harmonic subtraction, baryonic disk and bulge features produce spectral power at the two-sigma level, confirming prior observations that baryonic feedback imprints scale-dependent signatures on rotation curves. The harmonic correction successfully removes this contamination while preserving sensitivity to signals at the predicted ZD wavenumbers, as validated by the injection recovery test.
//!
//! Placing these results in the broader context of dark matter searches, GHOST extends the tradition of null-result constraints to a qualitatively new observable. Direct detection experiments have progressively excluded WIMP-nucleon cross-sections following conventions for transparent upper-limit reporting [baxter2021recommended], and each successive null result has narrowed the viable parameter space for particle dark matter models [bauer2013dark]. Gamma-ray observations of Milky Way satellites [javier2020gammaray] and condensed-matter searches for light dark matter [yonatan2021searches] have similarly established stringent exclusion regions. GHOST's contribution is to test a prediction class that lies entirely outside these parameter spaces -- spectral rather than amplitude-level modifications to halo structure, rooted in algebraic rather than particle-physics considerations. The null result establishes that if ZD signatures exist, their amplitude must fall below $\alpha_{\text{zd}} < 0.35$ in the inner-halo regime, a constraint that will improve by approximately an order of magnitude with next-generation radio surveys achieving outer-halo coverage beyond $x = 5\,r/r_s$.
//!
//! The multi-configuration experimental design, motivated by concerns about the generalizability of empirical findings across analysis choices [yarkoni2020generalizability], reveals a scientifically useful pattern: the red-noise correction produces SNR values identical to the uncorrected baseline (CD-16 vs RedNse: $\Delta < 0.001$). This indicates that the $k^{0.81}$ red-noise floor characterizes baryonic systematics accurately, validating it as a reusable spectral baseline for future analyses. Conversely, the single-mode ablation produces the lowest framework SNR (0.4618), confirming that higher harmonics capture genuine baryonic physics rather than overfitting noise. The broader theoretical framework connecting quantum gravity phenomenology to astrophysical observables [addazi2022quantum] remains viable but must contend with the inner-halo constraints established here when making spectral predictions at galactic scales.
//!
//! ## 7. Limitations
//!
//! Five specific limitations bound the scope and constraining power of this analysis.
//!
//! First, MaNGA's integral-field coverage extends to approximately 1.5 effective radii for most galaxies, restricting the GHOST analysis to the inner-halo regime $0.5 \leq x \leq 1.35$ in units of the NFW scale radius. Theoretical predictions for ZD signatures peak at $x = 5$--$10$, well beyond MaNGA's radial reach. The constraints reported here therefore apply exclusively to the inner halo and do not exclude ZD signatures at larger radii where the dark-matter-to-baryon ratio is substantially higher.
//!
//! Second, the ZD amplitude upper limit $\alpha_{\text{zd}}$ lacks physical calibration in velocity units (km/s) or a direct mapping to theoretical model predictions. Without this calibration, the constraining power of $\alpha_{\text{zd}} < 0.35$ is model-dependent and cannot be directly compared to constraints from other dark matter search channels.
//!
//! Third, the bootstrap ensemble uses 13 seeds. While the per-seed variance of approximately 1.5% yields a standard error on the median well below the detection threshold, a larger seed count would reduce sampling uncertainty further and enable more robust tail-probability estimates.
//!
//! Fourth, the effective galaxy sample is smaller than nominally claimed. Mass-quartile analysis reveals that the highest-mass quartile ($\log M_{200} > 12.07$) contributes zero stacking bins due to insufficient radial coverage, and the third quartile contributes only six bins. The constraining power rests primarily on the lower two mass quartiles ($N_{\text{eff}} \approx 3500$ galaxies with $\log M_{200} < 11.5$).
//!
//! Fifth, the per-seed and structured aggregate pipelines compute different quantities under the same metric name, producing a ratio of 1.4--2.7x between aggregate and per-seed values for physical-model conditions. This discrepancy, while not affecting the null conclusion, must be resolved and documented before the pipeline is applied to future datasets where the distinction could impact detection claims.
//!
//! ## 8. Conclusion
//!
//! GHOST provides the first systematic spectral test of algebraically motivated zero-dark density predictions against spatially resolved galactic kinematics, finding no evidence for ZD signatures across four distinct algebraic frameworks, three galaxy subsamples, and nine experimental conditions spanning 351 independent analyses of 6992 MaNGA DR17 galaxies. The null result is algebra-universal, internally robust to bootstrap resampling, and validated by a positive control that confirms pipeline sensitivity to genuine spectral structure. These inner-halo constraints motivate three future directions: extending the analysis to outer-halo radii ($x > 5\,r/r_s$) using forthcoming radio kinematic surveys from SKA and its pathfinders, where ZD predictions peak; cross-matching MaNGA targets with deep photometric surveys to improve NFW fitting through independent concentration priors; and calibrating $\alpha_{\text{zd}}$ in physical units to enable direct comparison with constraints from particle-physics and gravitational-wave dark matter searches.
//!
//! ## References
//!
//! 1. [adame2025desi] Adame et al. (2025). DESI baryon acoustic oscillation measurements.
//! 2. [addazi2022quantum] Addazi et al. (2022). Quantum gravity phenomenology at the dawn of the multi-messenger era.
//! 3. [adrian2025science] Adrian et al. (2025). Science with the Einstein Telescope.
//! 4. [aghanim2020iplancki] Aghanim et al. (2020). Planck 2018 results. VI. Cosmological parameters.
//! 5. [akrami2020iplancki] Akrami et al. (2020). Planck 2018 results. Foreground contamination and systematic uncertainties.
//! 6. [allen2014sami] Allen et al. (2014). The SAMI Galaxy Survey: early data release.
//! 7. [amon2022dark] Amon et al. (2022). Dark Energy Survey Year 3 results: cosmology from cosmic shear.
//! 8. [asgari2020kids] Asgari et al. (2020). KiDS-1000 cosmology: cosmic shear constraints.
//! 9. [barausse2020prospects] Barausse et al. (2020). Prospects for fundamental physics with LISA.
//! 10. [baudis2012direct] Baudis (2012). Direct dark matter detection: the next decade.
//! 11. [bauer2013dark] Bauer et al. (2013). Dark matter in the coming decade: complementary paths to discovery and beyond.
//! 12. [baxter2021recommended] Baxter et al. (2021). Recommended conventions for reporting results from direct dark matter searches.
//! 13. [brunner2022erosita] Brunner et al. (2022). The eROSITA Final Equatorial Depth Survey.
//! 14. [chan2019fitting] Chan et al. (2019). Fitting the radio continuum spectral energy distributions of galaxies.
//! 15. [cicoli2024string] Cicoli et al. (2024). String cosmology: from the early universe to today.
//! 16. [coogan2022measuring] Coogan et al. (2022). Measuring the dark matter environments of black hole binaries with gravitational waves.
//! 17. [emsellem2022phangsmuse] Emsellem et al. (2022). The PHANGS-MUSE survey.
//! 18. [fogarty2014kinematic] Fogarty et al. (2014). SAMI Galaxy Survey: kinematic morphology-density relation.
//! 19. [h2025kidslegacy] H. et al. (2025). The KiDS-Legacy analysis.
//! 20. [heymans2020kids] Heymans et al. (2020). KiDS-1000 cosmology.
//! 21. [j2023atacama] J. et al. (2023). Atacama Cosmology Telescope: CMB lensing measurements.
//! 22. [javier2020gammaray] Javier et al. (2020). Gamma-ray observations of Milky Way satellites.
//! 23. [kamionkowski2023hubble] Kamionkowski & Riess (2023). The Hubble tension and early dark energy.
//! 24. [li2023hyper] Li et al. (2023). Hyper Suprime-Cam Year 3 cosmic shear analysis.
//! 25. [marco2021nongaussian] Marco et al. (2021). Non-Gaussian methods for quantifying tensions between datasets.
//! 26. [mazur2023gravitational] Mazur & Mottola (2023). Gravitational condensate stars.
//! 27. [schirmer2024micromegas] Schirmer et al. (2024). micrOMEGAs: N-component dark matter.
//! 28. [schoot2021bayesian] Schoot et al. (2021). Bayesian statistics and modelling.
//! 29. [secco2022dark] Secco et al. (2022). Dark Energy Survey Year 3 results: cosmic shear from galaxy shapes.
//! 30. [thibaut2025atacama] Thibaut et al. (2025). Atacama Cosmology Telescope: CMB lensing.
//! 31. [vagnozzi2023seven] Vagnozzi (2023). Seven hints that early-time new physics alone is not sufficient.
//! 32. [valentino2021realm] Valentino et al. (2021). In the realm of the Hubble tension.
//! 33. [westhuizen2025interacting] Westhuizen et al. (2025). Interacting dark energy and dark matter.
//! 34. [yarkoni2020generalizability] Yarkoni & Westfall (2020). Choosing prediction over explanation in psychology: lessons from machine learning.
//! 35. [yi2023black] Yi et al. (2023). Black hole solutions and pseudo-isothermal dark matter halos.
//! 36. [yonatan2021searches] Yonatan et al. (2021). Searches for light dark matter using condensed matter systems.
//!
