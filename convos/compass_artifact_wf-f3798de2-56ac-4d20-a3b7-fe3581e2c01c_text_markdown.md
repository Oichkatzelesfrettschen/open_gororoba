# Discovery of QGP signatures in oxygen and neon collisions at the LHC

The LHC's first-ever light-ion collisions in July 2025 produced definitive evidence of quark-gluon plasma formation in small nuclear systems. CMS discovered charged-particle suppression in O-O collisions with **RAA = 0.69 ± 0.04** at pT ≈ 6 GeV—a 5σ deviation from unity—while ALICE and ATLAS measured sizable flow harmonics driven by nuclear geometry. These results, collected at **√sNN = 5.36 TeV** (not the originally planned 7 TeV), establish that QGP signatures emerge robustly even in collision systems with just 16+16 nucleons. The experimental program successfully tested α-clustering hypotheses, with Ne-Ne/O-O flow ratios revealing sensitivity to neon's prolate "bowling-pin" deformation versus oxygen's tetrahedral structure.

---

## CMS achieves first observation of jet quenching in oxygen collisions

The CMS collaboration's **HIN-25-008** analysis (arXiv:2510.09864, submitted October 2025) represents the primary discovery paper for this physics program. Using **6.1 nb⁻¹** of O-O data and **1 pb⁻¹** of pp reference at matched energy, CMS measured the nuclear modification factor RAA for charged particles across 3.0 < pT < 103.6 GeV in |η| < 1.

The key measurement shows RAA exhibits a characteristic suppression pattern: beginning near **0.77 at low pT**, dropping to a minimum of **0.69 ± 0.04 at pT ≈ 6 GeV**, then rising smoothly toward unity at high pT (~100 GeV). The minimum represents more than five standard deviations from unity, constituting unambiguous evidence for parton energy loss. Systematic uncertainties are dominated by luminosity determination and nuclear PDF (nPDF) effects, with EPPS21, nCTEQ15HQ, and NNPDF4.0 all tested. Models incorporating coherent energy loss with hydrodynamic evolution describe the data well, while baseline calculations without energy loss (using nPDFs alone) predict RAA values above the measurements for most pT.

The companion **CMS HIN-25-014** analysis extended measurements to Ne-Ne collisions, finding an even deeper suppression of **RAA ≈ 0.6 at pT ≈ 6 GeV** with 7σ significance. Combined with existing Xe-Xe and Pb-Pb data, these results enable the first systematic scan of RAA versus nuclear size (A). CMS reports that plotting RAA against A^(1/3)—proportional to nuclear radius—reveals clearer scaling than against A alone, suggesting direct connection between energy loss and path length through the QGP medium.

| System | √sNN (TeV) | RAA minimum | pT of minimum | Significance |
|--------|------------|-------------|---------------|--------------|
| O-O | 5.36 | 0.69 ± 0.04 | ~6 GeV | >5σ |
| Ne-Ne | 5.36 | ~0.60 | ~6 GeV | >7σ |
| Xe-Xe | 5.44 | ~0.25 | ~6 GeV | established |
| Pb-Pb | 5.02 | ~0.15-0.20 | ~6 GeV | established |

---

## Flow harmonics reveal nuclear structure through collectivity

Three major collaborations published flow measurements within weeks of data collection. **ALICE** (arXiv:2509.06428, September 2025) reported the first evidence of geometry-driven anisotropic flow in O-O and Ne-Ne, with both v2 (elliptic) and v3 (triangular) coefficients showing "sizable values with highly non-trivial centrality dependence." Agreement between data and hydrodynamic model predictions from Giacalone et al. (Phys. Rev. Lett. 135, 012302, 2025) equals or surpasses accuracy achieved in heavy-ion collisions—a remarkable result given the far smaller system sizes.

**ATLAS** (arXiv:2509.05171, Phys. Rev. C) measured vn for n = 2, 3, 4 using both two-particle template-fit methods and four-particle subevent cumulants. The flow coefficients peak around pT ≈ 2 GeV with clear hierarchy v2 > v3 > v4. The critical finding concerns the centrality dependence: v2 in central Ne-Ne collisions significantly exceeds v2 in central O-O collisions, consistent with neon's prolate nuclear deformation generating larger initial-state ellipticity compared to oxygen's more spherical (or tetrahedrally-clustered) configuration.

The **CMS HIN-25-009** paper (arXiv:2510.02580) added crucial insight by revealing an unexpected centrality pattern for triangular flow. While v2 decreases toward central collisions (as in Pb-Pb), **v3 increases toward central collisions—opposite to the Pb-Pb trend**. This behavior indicates fundamentally different initial-state fluctuation dynamics in small versus large systems, likely reflecting the discrete nature of α-clustering in light nuclei.

**LHCb** contributed fixed-target measurements (arXiv:2509.12399) of PbNe and PbAr collisions at √sNN = 70.9 GeV, providing the first experimental confirmation of neon's distinctive "bowling-pin" shape through significantly larger v2 in central PbNe versus PbAr collisions.

---

## Nuclear geometry models connect structure to observables

Theoretical predictions preceding the data relied on sophisticated nuclear structure inputs that have now been validated. The ¹⁶O nucleus exhibits **tetrahedral α-clustering**—four α-particles arranged at vertices of a tetrahedron—producing enhanced triangular eccentricity ε3 that drives v3. The ²⁰Ne nucleus shows **bipyramidal α-clustering** (five α-particles forming a "bowling-pin" shape), generating strong prolate deformation that enhances v2 in central collisions.

Updated Glauber model calculations (arXiv:2507.05853, Loizides) using TGlauberMC v3.3 incorporated nuclear density profiles from Nuclear Lattice Effective Field Theory (NLEFT) and Projected Generator Coordinate Method (PGCM) calculations. Key geometric parameters include:

- **¹⁶O:** Near-spherical overall shape with tetrahedral substructure; radius ~2.7 fm
- **²⁰Ne:** Prolate deformation (bowling-pin); β2 ≈ 0.5-0.7; radius ~3.0 fm
- **Path lengths:** L ≈ 2-3 fm in central O-O versus L ≈ 6-8 fm in central Pb-Pb

The IP-Glasma + MUSIC framework (arXiv:2508.20432) successfully predicted the observed flow patterns by incorporating JIMWLK high-energy evolution of gluon saturation. These predictions demonstrated that Ne-Ne/O-O ratios of vn largely cancel systematic uncertainties, providing clean sensitivity to nuclear structure effects.

---

## Path length dependence follows linear scaling in expanding medium

A critical theoretical development concerns the functional form of energy loss versus path length L. The conventional BDMPS-Z formalism predicts ⟨ΔE⟩ ∝ L² for a static medium, while L³ scaling appears in some radiative scenarios. However, Arleo and Falmagne (arXiv:2411.13258, November 2024) analyzed extensive RHIC and LHC data to extract **β = 1.02 ± 0.09** for the power law ⟨ε⟩ ∝ L^β—demonstrating near-linear scaling.

This near-linear behavior arises naturally for longitudinally expanding (Bjorken) QGP, where the decreasing density as the medium expands compensates for longer path lengths. The DREENA framework (arXiv:2412.17106) further distinguishes between:

- **Parton-level energy loss:** Scales between L^1 and L^2
- **Hadron-level suppression:** Similar to partons
- **Jet RAA (R > 0):** Sublinear scaling (~L^0.6) due to retained in-cone radiation

For small systems like O-O, this scaling hierarchy has important implications: the shorter path lengths (~2-3 fm versus ~6-8 fm in Pb-Pb) combined with roughly linear scaling explain why RAA reaches only ~0.69 rather than the ~0.15-0.20 seen in central Pb-Pb.

---

## Coherence effects and pre-equilibrium dynamics shape small-system quenching

Two theoretical developments proved essential for interpreting the O-O results. First, **coherence effects** (arXiv:2510.17570) increase RAA without proportionally affecting v2, as JEWEL Monte Carlo studies with Trajectum hydrodynamic profiles demonstrate. The critical coherence angle θc ~ 1/√(q̂L³) is larger in small systems, meaning jets remain "unresolved" (acting as single color charges) over a broader angular range.

Second, **pre-equilibrium quenching** (arXiv:2509.19430, Pablos & Takacs) shows that energy loss begins remarkably early—around **0.2 fm/c**—deep in the pre-equilibrium phase before hydrodynamization. Using Bayesian inference coupled to IP-Glasma+JIMWLK+MUSIC+UrQMD, these authors predicted sizable hadron/jet suppression in O-O that exceeds no-quenching baselines, consistent with CMS observations.

The q̂ transport coefficient—governing transverse momentum broadening—maintains similar values (~1-2 GeV²/fm at T = 400 MeV) across system sizes, but the effective q̂L product is substantially smaller in O-O due to the geometric limitation. Combined with shorter QGP lifetimes (τQGP ~ 3-4 fm/c versus ~10 fm/c in Pb-Pb) and lower initial temperatures (T₀ ~ 300-350 MeV versus ~400-500 MeV), these factors explain the observed suppression magnitude.

---

## Strangeness enhancement awaits dedicated O-O analysis

While flow and jet quenching results appeared rapidly, strangeness enhancement measurements specifically for O-O and Ne-Ne remain forthcoming. ALICE's existing pp strangeness program (arXiv:2511.10306, arXiv:2405.14511, arXiv:2405.19890) establishes the baseline phenomenology:

- Strange-to-pion ratios increase smoothly with multiplicity from ~2 to ~2000 charged particles at midrapidity
- Enhancement is more pronounced for higher strangeness content: Ω > Ξ > Λ > K
- Ω yields follow **faster-than-linear multiplicity dependence**
- Statistical hadronization models with correlation over ~3 units of rapidity successfully describe event-by-event fluctuations, while string fragmentation models fail

The chemical freeze-out temperature Tch ≈ **155 MeV** emerges consistently from thermal fits. For O-O, the 0-5% most central events correspond roughly to 50-60% centrality Pb-Pb in multiplicity, with **dNch/dη = 135 ± 3 (syst.)** at midrapidity—well within the regime where canonical suppression and strangeness undersaturation become negligible.

---

## Experimental baselines and reference measurements

The CMS discovery relied on carefully matched pp reference data at √s = 5.36 TeV (1 pb⁻¹ integrated luminosity) collected during the same running period. This energy matching eliminates interpolation uncertainties that plagued earlier light-ion analyses.

Key baseline measurements include ALICE's light neutral-meson production in pp at √s = 13 TeV (arXiv:2411.09560), covering π⁰ from 0.2-200 GeV/c and η from 0.4-60 GeV/c. For multiplicity normalization, Pb-Pb reference values at √sNN = 5.02 TeV give **dNch/dη = 1943 ± 54** in 0-5% centrality within |η| < 0.5.

Nuclear PDF uncertainties remain the dominant systematic limitation for precision energy-loss measurements. Gebhard et al. (JHEP 04, 2025, 034; arXiv:2410.22405) demonstrated that deviations from RAA = 1 occur even WITHOUT quenching due to cold nuclear matter effects, and that semi-inclusive jet-triggered hadron measurements (IAA) offer kinematic windows where nPDF uncertainties largely cancel.

---

## Conclusion

The 2025 LHC light-ion campaign achieved its primary physics goals within months of data collection. The discovery of charged-particle suppression (RAA ≈ 0.69 in O-O, ≈ 0.60 in Ne-Ne) and sizable anisotropic flow provides the clearest evidence to date that QGP formation occurs in systems far smaller than traditionally assumed. Three insights emerge as particularly novel: (1) flow harmonics directly encode nuclear α-clustering geometry, enabling high-energy colliders to probe nuclear structure; (2) the opposite centrality dependence of v3 in small versus large systems reveals qualitatively different fluctuation dynamics; (3) near-linear path length scaling of energy loss, combined with pre-equilibrium quenching onset at ~0.2 fm, explains the observed RAA magnitude without requiring new physics beyond the standard jet quenching paradigm. Outstanding questions—including dedicated strangeness measurements and precision q̂ extraction for O-O—will be addressed as full analyses mature through 2026.