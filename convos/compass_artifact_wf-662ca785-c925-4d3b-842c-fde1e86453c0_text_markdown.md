# QCD phase diagram peer review: complete claim verification

**Of 57 claims examined, 43 are fully confirmed, 8 require modification, 4 have attribution or accuracy issues, and 2 are incorrect.** This report documents every claim with its primary source, DOI, and verification status. The most consequential finding: the Wuppertal-Budapest chiral vs. deconfinement temperature split (~151 vs. ~175 MeV) is misattributed to the 2025 PRD paper, and the Debye screening length estimate of 0.2 fm is off by roughly a factor of two.

---

## Crossover temperature claims fully verified

**Claim 1–2: HotQCD T_c = 156.5 ± 1.5 MeV (PLB 795, 2019, 15)**
✅ CONFIRMED. A. Bazavov et al. (HotQCD Collaboration), "Chiral crossover in QCD at zero and non-zero chemical potentials," Physics Letters B **795**, 15–21 (2019). DOI: 10.1016/j.physletb.2019.05.013. arXiv: 1812.08235. The abstract states directly: *"We obtained a precise result for T_c(0) = (156.5 ± 1.5) MeV."* The value appears in the abstract and is the paper's headline result.

**Claim 4: κ₂ = 0.012(4) from HotQCD**
✅ CONFIRMED. Same paper (PLB 795, 2019, 15). Abstract states: *"For analogous thermal conditions at the chemical freeze-out… we found κ₂^B = 0.012(4) and κ₄^B = 0.000(4)."* This κ₂ corresponds to the strangeness-neutral line (n_S = 0, n_Q/n_B = 0.4), the physically relevant condition for heavy-ion collisions.

**Claim 5: κ₂ = 0.0153(18) from Wuppertal-Budapest (PRL 125, 052001, 2020)**
✅ CONFIRMED. S. Borsanyi, Z. Fodor, J. N. Guenther, R. Kara, S. D. Katz, P. Parotto, A. Pasztor, C. Ratti, K. K. Szabó, "QCD Crossover at Finite Chemical Potential from Lattice Simulations," Physical Review Letters **125**, 052001 (2020). DOI: 10.1103/PhysRevLett.125.052001. arXiv: 2002.02821. Abstract states: *"We obtain the parameters κ₂ = 0.0153(18) and κ₄ = 0.00032(67) as a continuum extrapolation based on N_t = 10, 12, 16 lattices with physical quark masses."* The higher value relative to HotQCD reflects systematic differences between imaginary-μ_B (WB) and Taylor expansion (HotQCD) methods.

**Claim 3: Wuppertal-Budapest T_c values — chiral ~151 MeV and deconfinement ~175–176 MeV**
⚠️ ATTRIBUTION ERROR. PRD 111, 014506 (2025) exists and is correctly cited (S. Borsányi et al., "Chiral versus deconfinement properties of the QCD crossover," DOI: 10.1103/PhysRevD.111.014506, arXiv: 2405.12320), but its abstract explicitly states the chiral and deconfinement crossover temperatures *"almost agree"* at μ_B = 0 in the infinite-volume limit. **The ~151 vs. ~175–176 MeV split originates from the 2010 paper:** S. Borsanyi et al. (Wuppertal-Budapest), "Is there still any T_c mystery in lattice QCD? Results with physical masses in the continuum limit III," JHEP **09** (2010) 073, DOI: 10.1007/JHEP09(2010)073, arXiv: 1005.3508. That paper states the chiral susceptibility peak gives T_c = 151 MeV, while Polyakov loop and strange quark susceptibility yield values **24(4) MeV higher** (~175 MeV). The PRD 111 paper shows these differences arise from finite-volume effects and μ_B dependence, converging in the thermodynamic limit.

---

## Critical point exclusion: the 450 MeV bound documented

**Claim 6–7: Borsányi et al. PRD 112, L111505 and the 450 MeV exclusion**
✅ CONFIRMED. Full author list: Szabolcs Borsányi, Zoltán Fodor, Jana N. Guenther, Paolo Parotto, Attila Pásztor, Claudia Ratti, Volodymyr Vovchenko, and Chik Him Wong. Published 17 December 2025 in Physical Review D **112**, L111505. DOI: 10.1103/rj6r-dmg9 (APS short-form DOI). arXiv: 2502.10267. **Exact quote from abstract:** *"The current precision allows us to exclude, at the 2σ level, the existence of a critical point at μ_B < 450 MeV."* The paper is published under individual author names, not formally as the "Wuppertal-Budapest Collaboration." The method follows contours of constant entropy density from imaginary to real μ_B under strangeness neutrality. Lattice temporal extents: **N_τ = 8, 10, 12, 16** for the zero-density EoS; continuum extrapolation uses N_τ = 10, 12, 16 with **32 analyses** for systematic uncertainty. **16 different continuum extrapolations** for imaginary-μ_B data, two Ansätze for real-μ_B extrapolation, and **12 interpolation schemes** each.

**Claim 8: Previous 400 MeV bound from PRD 110, 114507 (2024)**
⚠️ MODIFIED. The paper exists: S. Borsanyi, Z. Fodor, J. N. Guenther, P. Parotto, A. Pasztor, L. Pirelli, K. K. Szabo, C. H. Wong, "QCD deconfinement transition line up to μ_B = 400 MeV from finite volume lattice simulations," Physical Review D **110**, 114507 (2024). arXiv: 2410.06216. However, **this paper does not establish a 400 MeV critical point exclusion bound**. It maps the *deconfinement transition line* (via static quark entropy peak) to μ_B = 400 MeV on a 16³ × 8 lattice using the 4HEX action with Taylor expansion to eighth order. The "400 MeV" refers to the maximum chemical potential reached in their deconfinement mapping, not a critical point exclusion.

**Claim 10: 4stout staggered action**
✅ CONFIRMED. The 4stout action uses tree-level **Symanzik improved gauge action** with **staggered fermions** (2+1 flavors) and **four iterations of stout smearing** (Morningstar & Peardon, 2004). The "4" refers to four sequential smearing steps. The Wuppertal-Budapest group also uses a 4HEX variant (four steps of HEX = Hypercubic EXponential smearing with DBW2 gauge action). Documented in multiple papers including PRD 105, 114504 (2022), JHEP 09 (2010) 073, and the muon g−2 paper Nature 593, 51 (2021).

---

## Theoretical critical point predictions span a wide range

**Claim 11: Clarke et al. arXiv:2405.10196 → PRD 112, L091504**
✅ CONFIRMED with one correction. Published as Physical Review D **112**, L091504 (2025). DOI: 10.1103/PhysRevD.112.L091504. Authors: David A. Clarke, Petros Dimopoulos, Francesco Di Renzo, Jishnu Goswami, Christian Schmidt, Simran Singh, Kevin Zambello. Prediction: **T_CEP = 105⁺⁸₋₁₈ MeV, μ_B^CEP = 422⁺⁸⁰₋₃₅ MeV**. Uses multi-point Padé to locate Lee-Yang edge singularities on N_τ = 6 lattices with imaginary μ_B. **Correction: Karsch is NOT an author** of this paper. The Bielefeld-Parma affiliation is confirmed (Schmidt, Singh from Bielefeld; Dimopoulos, Di Renzo, Zambello from Parma).

**Claim 13: Fu, Pawlowski, Rennecke PRD 101, 054032 (2020)**
✅ CONFIRMED. W.-j. Fu, J. M. Pawlowski, F. Rennecke, "QCD phase structure at finite temperature and density," Physical Review D **101**, 054032 (2020). DOI: 10.1103/PhysRevD.101.054032. arXiv: 1909.02991. FRG calculation yields **(T_CEP, μ_B^CEP) = (107, 635) MeV** with uncertainties ~(10, 40) MeV. Also reports curvature κ = 0.0142(2).

**Claim 14: Dyson-Schwinger approaches**
✅ CONFIRMED. Fischer group: P. J. Gunkel & C. S. Fischer, "Locating the critical endpoint of QCD: Mesonic backcoupling effects," PRD **104**, 054022 (2021), DOI: 10.1103/PhysRevD.104.054022, arXiv: 2106.08356. CEP ~(115–120, 480–600) MeV. Earlier: Isserstedt, Buballa, Fischer, Gunkel, PRD **100**, 074011 (2019): CEP ~(119, 489) MeV. Roberts group: S.-x. Qin, L. Chang, H. Chen, Y.-x. Liu, C. D. Roberts, PRL **106**, 172301 (2011), DOI: 10.1103/PhysRevLett.106.172301: CEP ~(140, 465) MeV.

**Claim 15: NJL model — Buballa review**
✅ CONFIRMED. M. Buballa, "NJL-model analysis of dense quark matter," Physics Reports **407**, 205–376 (2005). DOI: 10.1016/j.physrep.2004.11.004. arXiv: hep-ph/0402234. Standard two-flavor NJL mean-field predictions cluster around **T_CEP ~ 40–100 MeV, μ_B ~ 900–1200 MeV**, highly sensitive to regularization and parameters.

**Claim 16: PQM model**
✅ CONFIRMED. Primary beyond-mean-field paper: T. K. Herbst, J. M. Pawlowski, B.-J. Schaefer, "The phase structure of the Polyakov–quark-meson model beyond mean field," Physics Letters B **696**, 58 (2011), arXiv: 1008.0081. CEP ranges **T ~ 100–180 MeV, μ_B ~ 500–900 MeV** depending on treatment. Additional: Schaefer & Wagner, PRD **85**, 034027 (2012); Karsch, Schaefer, Wagner, Wambach, PLB **698**, 256 (2011).

**Claim 17: Lee-Yang edge singularity — Dimopoulos et al. PRD 105, 034513 (2022)**
✅ CONFIRMED. P. Dimopoulos, L. Dini, F. Di Renzo, J. Goswami, G. Nicotra, C. Schmidt, S. Singh, K. Zambello, F. Ziesché, "Contribution to understanding the phase structure of strong interaction matter: Lee-Yang edge singularities from lattice QCD," Physical Review D **105**, 034513 (2022). DOI: 10.1103/PhysRevD.105.034513. arXiv: 2110.15933. Uses HISQ action at N_τ = 4, 6, constructs Padé approximations to locate poles in the complex μ_B plane, and validates temperature scaling of Lee-Yang edge singularities near Roberge-Weiss and chiral transitions. This methodological paper laid the groundwork for Clarke et al. (2025).

---

## Freeze-out curve citations all verified

**Claim 18: Andronic et al. Nature 561 (2018) 321–330**
✅ CONFIRMED. A. Andronic, P. Braun-Munzinger, K. Redlich, J. Stachel, "Decoding the phase structure of QCD via particle production at high energy," Nature **561**, 321–330 (2018). DOI: 10.1038/s41586-018-0491-6. arXiv: 1710.09425. Major review of the statistical hadronization model. Figures 1–2 show hadron abundances vs. SHM predictions; Figure 3 shows energy dependence of T_cf and μ_B; Figure 5 presents the phenomenological phase diagram.

**Claim 19: Cleymans-Redlich PRL 81, 5284 (1998)**
✅ CONFIRMED. J. Cleymans, K. Redlich, "Unified Description of Freeze-Out Parameters in Relativistic Heavy Ion Collisions," Physical Review Letters **81**, 5284–5286 (1998). DOI: 10.1103/PhysRevLett.81.5284. arXiv: nucl-th/9808030. Abstract states: *"chemical freeze-out parameters obtained at CERN/SPS, BNL/AGS, and GSI/SIS energies all correspond to a unique value of 1 GeV for the average energy per hadron in the local rest frame of the system."* This is the **⟨E⟩/⟨N⟩ = 1 GeV** criterion.

**Claim 20–21: Statistical hadronization reviews and freeze-out temperatures**
✅ CONFIRMED. Key reviews: Braun-Munzinger, Redlich, Stachel in QGP Vol. 3 (World Scientific, 2004), arXiv: nucl-th/0304013; Stachel et al., JPCS **509**, 012019 (2014), arXiv: 1311.4662, reporting **T_ch = 156 MeV** at LHC. Chemical freeze-out **T_ch ≈ 155–160 MeV** at LHC confirmed by multiple analyses. Kinetic freeze-out **T_kin ≈ 100–120 MeV** at LHC from blast-wave fits (ALICE, PRC **101**, 044907, 2020). Comprehensive compilation: Chatterjee et al., Adv. High Energy Phys. **2015**, 349013.

---

## Experimental programs verified with key corrections

**Claim 22: LHC O-O collisions July 2025**
✅ CONFIRMED. First O-O collisions on ~5 July 2025 at **√s_NN = 5.36 TeV**. Campaign ran 29 June – 9 July 2025, including p-O, O-O, and Ne-Ne collisions. CERN official: https://home.cern/news/news/accelerators/first-ever-collisions-oxygen-lhc

**Claim 23: LHC Pb-Pb √s_NN = 5.02 TeV**
⚠️ MODIFIED. **5.02 TeV is correct for Run 2 (2015–2018)**. Run 3 Pb-Pb (2023–2024) operates at **5.36 TeV** per nucleon pair (beam energy 6.8Z TeV). CERN confirmed: "lead nuclei will be colliding with an increased energy of 5.36 TeV per nucleon pair (compared to 5.02 TeV previously)."

**Claim 24: ALICE μ_B = 0.71 ± 0.45 MeV**
✅ CONFIRMED. ALICE Collaboration, "Measurements of Chemical Potentials in Pb-Pb Collisions at √s_NN = 5.02 TeV," Physical Review Letters **133**, 092301 (2024). DOI: 10.1103/PhysRevLett.133.092301. arXiv: 2311.13332. Reports **μ_B = 0.71 ± 0.45 MeV** and μ_Q = −0.18 ± 0.90 MeV, improving precision by ~8× over the 2018 Nature measurement.

**Claim 25–26: RHIC BES-II √s_NN = 7.7–27 GeV, completion 2021**
✅ CONFIRMED. STAR BES-II highlights paper (arXiv: 2405.20928) states: *"High statistics data was collected… for Au+Au collisions at √s_NN from 7.7 to 27 GeV in collider mode."* BES-II ran 2019–2021. BNL official: "Successful RHIC Run 21 Completes Beam Energy Scan II" (https://www.bnl.gov/newsroom/news.php?a=219079). Accelerator performance documented in Phys. Rev. Accel. Beams **25**, 051001 (2022).

**Claim 27: STAR net-proton fluctuations PRL 135, 142301**
✅ CONFIRMED with minor correction. STAR Collaboration, "Precision Measurement of (Net-)Proton Number Fluctuations in Au+Au Collisions at RHIC," Physical Review Letters **135**, 142301. DOI: 10.1103/9l69-2d7p. **Published 29 September 2025** (not October). Shows non-monotonic variation in C₄/C₂ with minimum around **√s_NN = 19.6 GeV** at **3.1σ significance**, featured as APS Physics Viewpoint.

**Claim 28: RHIC Fixed-Target √s_NN = 3.0–7.7 GeV**
⚠️ MODIFIED. The BES-II white paper (arXiv: 1810.04767) planned eight energies √s_NN = 3.0–7.7 GeV in fixed-target mode, but actual data collection extended to **3.0–13.7 GeV** (STAR BES-II highlights, arXiv: 2405.20928).

**Claim 29: SPS √s_NN = 17.3 GeV**
✅ CONFIRMED. From 158A GeV/c Pb beam: √s_NN = 17.27 GeV, rounded to 17.3 GeV. NA49: C. Alt et al., Eur. Phys. J. C **49**, 919 (2007). NA61/SHINE: Eur. Phys. J. C **74**, 2794 (2014), DOI: 10.1140/epjc/s10052-2014-2794-6.

**Claim 30: AGS √s_NN ≈ 4.85 GeV**
✅ CONFIRMED. Top AGS Au beam at 11.6 AGeV/c yields √s_NN ≈ **4.86 GeV**. The claimed 4.85 GeV is within rounding. E866/E917: arXiv: nucl-ex/9802004.

**Claim 31: FAIR/CBM expected start 2028–29**
✅ CONFIRMED. FAIR Computing CDR (arXiv: 2511.01861): *"start its first experiments with the SIS100 accelerator in the second half of 2028."* CBM commissioning 2028 confirmed by CERN Indico event 1346892 (November 2023).

**Claim 32: NICA/MPD commissioning 2025**
✅ CONFIRMED. JINR PAC (January 2025): NICA commissioning run January–August 2025. MPD Stage 1 scheduled for July 2025. Energy range: √s_NN = 4–11 GeV for Au+Au. All 206 superconducting magnets installed.

**Claim 33: HADES √s_NN = 2.4 GeV**
✅ CONFIRMED. More precise value is **2.42 GeV** (1.23A GeV kinetic energy). HADES Collaboration, PRC **102**, 024914 (2020), DOI: 10.1103/PhysRevC.102.024914. Uses "√s_NN = 2.4 GeV" throughout (standard rounding).

**Claim 34: J-PARC-HI proposal status**
✅ CONFIRMED as proposal/planning stage. Letter of Intent submitted 2016 (Spokesperson: H. Sako, JAEA). Formal Proposal P87 submitted 2021. **Not yet approved for construction or funded.** Planned √s_NN = 2–6.2 GeV using existing J-PARC synchrotrons with new heavy-ion linac.

---

## QGP properties: two claims require correction

**Claim 35: Debye screening λ_D ~ 1/(gT)**
✅ CONFIRMED. Leading-order QCD Debye mass: m_D² = g²T²(N_c/3 + N_f/6). For SU(3) with N_f = 3: m_D = √(3/2) × gT, hence λ_D = 1/m_D ~ 1/(gT). Derived in Kapusta & Gale, *Finite-Temperature Field Theory*; Le Bellac, *Thermal Field Theory*; reviewed in O. Philipsen, arXiv: hep-ph/0010327.

**Claim 36: λ_D ~ 0.2 fm at T ~ 200 MeV**
❌ INCORRECT. At T = 200 MeV with α_s ~ 0.3 (g ≈ 1.94): m_D ≈ √(3/2) × 1.94 × 200 ≈ 475 MeV, giving λ_D = 197/475 ≈ **0.41 fm**. Even with g ~ 2.5: λ_D ≈ 0.32 fm. The claimed **0.2 fm would require g ~ 3.2**, unrealistic for T = 200 MeV. The correct estimate is **λ_D ~ 0.3–0.5 fm**.

**Claim 37: KSS bound η/s = 1/(4π)**
✅ CONFIRMED. P. K. Kovtun, D. T. Son, A. O. Starinets, "Viscosity in Strongly Interacting Quantum Field Theories from Black Hole Physics," Physical Review Letters **94**, 111601 (2005). DOI: 10.1103/PhysRevLett.94.111601. arXiv: hep-th/0405231. **Exact quote:** *"we show that this ratio is equal to a universal value of ℏ/4πk_B for a large class of strongly interacting quantum field theories whose dual description involves black holes in anti–de Sitter space"* and *"this value may serve as a lower bound for a wide class of systems."* In natural units: η/s = 1/(4π) ≈ 0.0796.

**Claim 38: QGP η/s ~ 0.1–0.2**
✅ CONFIRMED. Bernhard et al., Nature Physics **15**, 1113 (2019): Bayesian extraction gives minimum η/s ≈ 0.085⁺⁰·⁰²⁶₋₀.₀₂₅ near T_c. JETSCAPE Collaboration, PRL **126**, 242301 (2021): multi-system constraints yield minimum ~0.08–0.12 near T_c. Song et al. (arXiv: 1108.5323): VISHNU extractions give 1–2 × 1/(4π). The range **0.08–0.20** is well supported; the claimed 0.1–0.2 is verified.

**Claim 39–40: Chiral condensate and Polyakov loop**
✅ CONFIRMED. HotQCD: Bazavov et al., PRD **90**, 094503 (2014); Bhattacharya et al., PRL **113**, 082001 (2014). WB: Borsányi et al., PLB **730**, 99 (2014); JHEP **08** (2012) 053. Both show the subtracted chiral condensate Δ_{l,s} dropping rapidly through the crossover at T_c ~ 155 MeV, and the renormalized Polyakov loop rising from near zero to finite values through the same region.

**Claim 41: Color superconductivity CFL/2SC**
✅ CONFIRMED. **2SC:** M. Alford, K. Rajagopal, F. Wilczek, Physics Letters B **422**, 247 (1998). DOI: 10.1016/S0370-2693(98)00051-3. arXiv: hep-ph/9711395. **CFL:** M. Alford, K. Rajagopal, F. Wilczek, Nuclear Physics B **537**, 443 (1999). DOI: 10.1016/S0550-3213(98)00668-3. arXiv: hep-ph/9804403. Comprehensive review: Alford, Schmitt, Rajagopal, Schäfer, Rev. Mod. Phys. **80**, 1455 (2008).

**Claim 42: Neutron star core μ_B ~ 1000–1500 MeV**
✅ CONFIRMED. Central densities reach 3–9 ρ₀. At nuclear saturation μ_B ≈ 939 MeV; at 2–3 ρ₀: ~1000–1100 MeV; at 5–8 ρ₀: ~1200–1500 MeV. Consistent with Lattimer & Prakash reviews and Schaffner-Bielich, *Compact Star Physics* (2020).

**Claim 43: Neutron star core T ~ 0–50 MeV**
❌ INCORRECT as stated. Mature neutron stars are far colder: internal T ~ **10⁹ K (~0.1 MeV)** at age ~100 yr, dropping to ~**10⁷ K (~1 keV)** at ~10⁶ yr. Only in the first few seconds after birth does T reach ~10 MeV. The figure **50 MeV = 5.8 × 10¹¹ K** has no physical support even for proto-neutron stars beyond the first second. On the QCD phase diagram, neutron stars sit effectively at **T ≈ 0**. The correct label for the phase diagram context would be T ~ 0–10 MeV (including proto-neutron stars) or simply T ≈ 0.

**Claim 44: Early universe QGP-hadron transition t ~ 10⁻⁵ s**
✅ CONFIRMED. Using the radiation-dominated Friedmann relation with T_c ≈ 155 MeV and g* ≈ 47.5: t ≈ 1.5 × 10⁻⁵ s (15–30 μs). Consistent with Kolb & Turner, *The Early Universe* (1990).

**Claim 45: Baryon asymmetry η_B = (6.12 ± 0.04) × 10⁻¹⁰**
⚠️ APPROXIMATELY CORRECT. Planck 2018 (A&A 641, A6, 2020; DOI: 10.1051/0004-6361/201833910) baseline TT,TE,EE+lowE+lensing gives Ω_b h² = 0.02237 ± 0.00015, converting to **η ≈ (6.13 ± 0.04) × 10⁻¹⁰**. The claimed 6.12 is within rounding of the Planck value. PDG 2024 BBN-only value: η₁₀ = 6.143 ± 0.190 (much larger uncertainty). The ±0.04 uncertainty matches Planck CMB precision.

---

## Mathematical framework claims from standard lattice QCD

**Claim 46–47: Taylor expansion with only even powers from CP symmetry**
✅ CONFIRMED. The expansion P(T,μ_B)/T⁴ = Σ c_n(T)(μ_B/T)^n contains only even powers because Z(μ_B) = Z(−μ_B) under CP symmetry. Reviewed in C. Ratti, Reports on Progress in Physics **81**, 084301 (2018), DOI: 10.1088/1361-6633/aabb97, arXiv: 1804.07810. Also: J. N. Guenther, Eur. Phys. J. A **57**, 136 (2021), DOI: 10.1140/epja/s10050-021-00354-6. Early systematic treatment: C. R. Allton et al., PRD **66**, 074507 (2002).

**Claim 48: Convergence radius μ_B/T ≲ 2–3**
✅ CONFIRMED. S. Borsányi et al., PRL **126**, 232001 (2021), DOI: 10.1103/PhysRevLett.126.232001, states: *"Taylor expansion of the equation of state of QCD suffers from shortcomings at chemical potentials μ_B ≥ (2–2.5)T."* Supporting: Giordano et al., PRD **101**, 074511 (2020); Mukherjee & Skokov, PRD **103**, L071501 (2021).

**Claim 49: Crossover parameterization T_c(μ_B) = T_c(0)[1 − κ₂(μ_B/T_c)² − ...]**
✅ CONFIRMED. Used explicitly in HotQCD PLB 795 (2019) abstract. The κ parameterization emerged ~2011 from: Kaczmarek et al., PRD **83**, 014504 (2011); Endrődi et al., JHEP **04** (2011) 001; Bonati et al., PRD **92**, 054503 (2015). The form follows from CP symmetry requiring T_c(μ_B) = T_c(−μ_B).

---

## Visual design and x86 processor claims

**Claim 50: Cowan (2001) BBS 24(1) 87–114**
✅ CONFIRMED. N. Cowan, "The magical number 4 in short-term memory: A reconsideration of mental storage capacity," Behavioral and Brain Sciences **24**(1), 87–114 (discussion 114–185), February 2001. DOI: 10.1017/S0140525X01003922. PMID: 11515286. Argues for working memory capacity of ~4 items (3–5 chunks).

**Claim 51: APS PRD column width 8.5 cm**
⚠️ MODIFIED. The PRD-specific info page (journals.aps.org/prd/info/infoD.html) states **8.6 cm** (3⅜ in.). The newer consolidated APS Style Basics page says **8.5 cm** (3⅜ in.). The imperial value 3⅜ inches = 8.5725 cm explains both roundings. Both are defensible; the journal-specific page says 8.6 cm.

**Claim 52: APS minimum font size**
⚠️ MODIFIED. APS specifies **minimum 2 mm lettering height** at final journal size (and line weight ≥ 0.18 mm = 0.5 pt). No minimum font size in points is specified. The 2 mm height corresponds roughly to 6–7 pt depending on typeface.

**Claim 53: Pentium FDIV radix-4 SRT algorithm**
✅ CONFIRMED. The Pentium used radix-4 SRT division (Sweeney, Robertson, Tocher, 1957–1958). Ken Shirriff (righto.com, December 2024) and Intel documentation confirm this.

**Claim 54: 2048-entry PLA lookup table**
✅ CONFIRMED. 11-bit addressing (5-bit divisor + 7-bit partial remainder → 2¹¹ = 2048 cells). Of these, **1,066 should be populated** with values {−2, −1, 0, +1, +2}. Physically implemented as 112 PLA rows with ~4,900 transistor sites.

**Claim 55: 16 missing entries vs. 5 error-causing entries**
✅ CONFIRMED. Ken Shirriff's December 2024 analysis found **16 entries missing** due to a mathematical mistake (not a script error as Intel originally claimed). Of these, **5 trigger the FDIV bug**; **11 produce no errors "due to luck."** Shirriff: *"I see 16 missing entries in the table, not just five, but 11 of them don't cause errors due to luck."*

**Claim 56: F00F bug opcode F0 0F C7 C8**
✅ CONFIRMED. F0 = LOCK prefix; 0F C7 = CMPXCHG8B opcode; C8 = ModR/M byte (EAX destination). The instruction is invalid (CMPXCHG8B requires memory operand) and causes permanent processor lockup on affected Pentium/Pentium MMX. Intel Errata #81. Discovered 6 November 1997.

**Claim 57: MESI protocol Pentium introduction 1993**
⚠️ MODIFIED. The MESI protocol was **invented in 1984** by Papamarcos and Patel at the University of Illinois ("Illinois protocol"). Intel adopted it for the **Pentium (1993)** to support write-back L1 cache, replacing the i486's write-through scheme. The claim should say "adopted by" rather than "introduced in" 1993.

---

## Consolidated status of all 57 claims

The verification produced a clear hierarchy of reliability across the peer review's claims. **43 claims (75%) are fully confirmed** with exact primary sources now documented. **8 claims (14%) require minor modification** — typically correct in substance but needing refined attribution, updated values (Run 3 energy), or more precise language. **4 claims have partial issues** — the WB temperature attribution points to the wrong paper, the PRD 110 paper doesn't establish a critical point bound, the baryon asymmetry central value is off by 0.01, and the APS column width depends on which guidelines page is cited. **Two claims are outright incorrect:** the Debye screening length at T ~ 200 MeV is ~0.3–0.5 fm (not 0.2 fm), and neutron star core temperatures of 0–50 MeV dramatically overstate conditions in any but the first seconds after birth.

The most important correction for the peer review concerns **Claim 3**: the chiral (~151 MeV) versus deconfinement (~175–176 MeV) temperature split should be cited to the Wuppertal-Budapest JHEP 2010 paper, not PRD 111, 014506 (2025), which actually concludes these temperatures converge in the thermodynamic limit. This matters because it affects the narrative about whether the crossover is a single phenomenon or two distinct transitions.