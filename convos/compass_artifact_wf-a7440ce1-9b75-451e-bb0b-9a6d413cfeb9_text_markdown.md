# QCD phase diagram verification: 57 claims fact-checked against primary sources

This comprehensive verification confirms **most claims are accurate** with important corrections needed for Wuppertal-Budapest crossover temperatures, Clarke et al. critical point coordinates, and the Pentium FDIV bug entry count. The December 2025 Borsányi et al. critical point exclusion paper (PRD 112, L111505) is fully verified with all methodological details confirmed. STROBE/CASP checklists are definitively **not applicable** to particle physics—HEP collaborations use internal review standards instead.

---

## Crossover temperature claims require significant corrections

**HotQCD T_c = 156.5 ± 1.5 MeV: VERIFIED ✓**

The exact value appears in Physics Letters B 795 (2019) 15-21 (arXiv:1812.08235). The HotQCD collaboration defines T_c using **chiral observables**—specifically the subtracted chiral condensate Σ(T) and disconnected chiral susceptibility χ_disc(T). The parameterization follows T_c(μ_X) = T_c(0)[1 − κ_2^X(μ_X/T_c(0))² − κ_4^X(μ_X/T_c(0))⁴].

**Wuppertal-Budapest distinct values: NEEDS CORRECTION ⚠️**

The claimed values (~155 MeV chiral vs ~158 MeV deconfinement) significantly understate the actual separation. Primary sources show:
- **Chiral crossover**: T_c ~ 151 MeV (renormalized chiral susceptibility peak)
- **Deconfinement crossover**: T_c ~ 175-176 MeV (Polyakov loop and strange susceptibility)

The **~25 MeV spread** between chiral and deconfinement definitions—not ~3 MeV—is documented in PRD 111, 014506 (2025) [arXiv:2405.12320], titled "Chiral versus deconfinement properties of the QCD crossover."

**Curvature parameter κ_2: COLLABORATION-DEPENDENT**

| Collaboration | κ_2 Value | Source |
|--------------|-----------|--------|
| HotQCD | 0.012(4) | PLB 795 (2019) 15 |
| Wuppertal-Budapest | 0.0153(18) | PRL 125, 052001 (2020) |

The claimed 0.012(2) matches HotQCD but differs from the more precise Wuppertal-Budapest value by ~2σ.

---

## December 2025 critical point exclusion paper fully verified

**Borsányi et al., PRD 112, L111505: ALL CLAIMS VERIFIED ✓**

- **Publication date**: December 17, 2025 ✓
- **arXiv**: 2502.10267 (submitted February 14, 2025) ✓
- **Key result**: "The current precision allows us to exclude, at the **2σ level**, the existence of a critical point at **μ_B < 450 MeV**" ✓

Every methodological detail is confirmed:
- Entropy contour method ✓
- Lattice sizes N_τ = 8, 10, 12, 16 ✓
- 4stout staggered action ✓
- 2+1+1 flavors (u, d, s, c with dynamical charm) ✓
- Imaginary chemical potential with analytic continuation ✓
- Strangeness neutrality condition ✓

This represents the most stringent lattice QCD exclusion limit achieved, superseding their previous 400 MeV bound from PRD 110, 114507 (2024).

---

## Critical point location theories require coordinate updates

**Clarke et al. arXiv:2405.10196: VALUES NEED UPDATING**

The arXiv preprint values (T_CP = 105⁺⁸₋₁₈ MeV, μ_B^CP = 420⁺⁸⁰₋₃₅ MeV) differ from the published PRD 112, L091504 (November 24, 2025) values:
- **Published**: T^CEP = 102⁺¹¹₋₂₃ MeV; μ_B^CEP = 428⁺¹⁶²₋₇₄ MeV
- The uncertainties expanded significantly from preprint to publication
- Collaboration: **Bielefeld-Parma** (not HotQCD)

**Dimopoulos et al. PRD 105, 034513 (2022): CORRECTLY IDENTIFIED AS METHODOLOGY ✓**

This paper presents Lee-Yang edge singularity calculation methodology using Padé approximations—it does not provide specific critical point coordinates.

**FRG consensus from Quark Matter 2025: VERIFIED WITH REFINEMENT**

The claimed (110, 630) MeV is approximately correct. arXiv:2510.11270 by Fabian Rennecke confirms the FRG consensus from Fu, Pawlowski, Rennecke, PRD 101, 054032 (2020):
- **(T_CEP, μ_B^CEP) = (107, 635) MeV**
- Systematic uncertainty: ΔT ≈ 10 MeV, Δμ_B ≈ 40 MeV (not "~10%" as stated)

**Model comparison summary**:
| Method | T_CEP (MeV) | μ_B^CEP (MeV) |
|--------|-------------|---------------|
| FRG | 107-110 | 620-635 |
| Dyson-Schwinger | 105-115 | 580-640 |
| Lattice extrapolation | 100-110 | 400-500 |
| NJL model | 40-120 | 200-400 (highly parameter-dependent) |

---

## Freeze-out curve parameterization verified

**Andronic et al. Nature 561 (2018) 321: VERIFIED ✓**

Title: "Decoding the phase structure of QCD via particle production at high energy"  
Pages: 321-330  
Authors: Andronic, Braun-Munzinger, Redlich, Stachel  
Chemical freeze-out temperature at μ_B = 0: **T_cf = 156.5 ± 1.5 MeV**—coinciding with the lattice QCD crossover temperature.

**Cleymans-Redlich parameterization: VERIFIED ✓**

The original paper is PRL 81, 5284 (1998). The key criterion is:
$$\langle E \rangle / \langle N \rangle \approx 1 \text{ GeV}$$

This empirical observation—that freeze-out occurs when average energy per hadron equals 1 GeV—was fitted to CERN SPS, BNL AGS, and GSI SIS data.

**Chemical vs kinetic freeze-out: VERIFIED ✓**

- **Chemical freeze-out**: inelastic collisions cease; particle ratios fixed; T_ch ~ 156 MeV at LHC
- **Kinetic freeze-out**: elastic collisions cease; spectra fixed; T_kin ~ 100-120 MeV at LHC

The hierarchy T_chemical > T_kinetic is confirmed across all sources.

---

## Experimental programs verified with precision updates

**LHC programs**:
- **O-O collisions**: Completed **July 2025** (first-ever oxygen-oxygen collisions at LHC)
- **Pb-Pb at √s_NN = 5.02 TeV**: ✓ **μ_B = 0.71 ± 0.45 MeV** (ALICE PRL 2024)—the claim of "~1 MeV" is consistent

**RHIC BES-II**:
- **√s_NN range**: 7.7-27 GeV (collider mode) ✓
- **μ_B coverage**: ~100-420 MeV ✓
- **Completion**: 2021 ✓
- **Analysis**: Precision net-proton fluctuation results published PRL 135, 142301 (October 2025)

**RHIC Fixed-Target**:
- **√s_NN = 3.0-7.7 GeV**: ✓
- **μ_B = 420-720 MeV**: ✓ (extends to ~750 MeV at lowest energies)

**Other facilities—all √s_NN values verified**:
| Facility | √s_NN (GeV) | μ_B (MeV) | Status |
|----------|-------------|-----------|--------|
| SPS | 17.3 | ~200-400 | Operating ✓ |
| AGS | ~4.85 | ~550-600 | Historical ✓ |
| FAIR/CBM | 2.7-4.9 | >500 | Expected 2028-29 |
| NICA/MPD | 2.4-11 | ~200-600 | Commissioning 2025 |
| HADES | 2.4 | ~800-830 | Operating ✓ |
| J-PARC-HI | 2-6.2 | High μ_B | Proposed only |

---

## QCD physics properties comprehensively verified

**Debye screening λ_D ~ 0.2 fm: VERIFIED ✓**

At T ~ 200 MeV with QCD coupling g ~ 2-3 (corresponding to α_s ~ 0.3-0.5), the formula λ_D ~ 1/(gT) yields approximately **0.2-0.25 fm**. The claim is confirmed.

**KSS bound η/s ≥ 1/(4π): VERIFIED ✓**

Original paper: Kovtun, Son, Starinets, **PRL 94, 111601 (2005)** [arXiv:hep-th/0405231]

The exact result η/s = ℏ/(4πk_B) ≈ **0.0796** was derived using AdS/CFT correspondence. QGP phenomenology extracts η/s ~ 0.1-0.2, remarkably close to this bound—making QGP among the "most perfect fluids" observed.

**Chiral condensate behavior: VERIFIED WITH NUANCE**

The condensate ⟨ψ̄ψ⟩ is **strongly suppressed** but not exactly zero above T_c. The crossover nature (not true phase transition) at physical quark masses means the chiral symmetry is approximately restored but never perfectly so.

**Color superconductivity: VERIFIED ✓**

Primary reviews by Alford, Rajagopal, Wilczek confirmed:
- PLB 422, 247 (1998) [hep-ph/9711395]
- Nucl. Phys. B537, 443 (1999) [hep-ph/9804403]
- "The Condensed Matter Physics of QCD" [hep-ph/0011333]

Phases include CFL (color-flavor locked) at highest densities and 2SC (two-flavor superconductor) at moderate densities, occurring at μ_B ≳ 300-500 MeV and T ≲ 50 MeV.

**Neutron star cores: VERIFIED ✓**

μ_B ~ 1000-1500 MeV (up to ~1700 MeV in most massive stars), T ~ 0-50 MeV (cold for old stars, hotter for proto-neutron stars and mergers).

**Early universe QGP-hadron transition: VERIFIED ✓**

- Time: t ~ 10⁻⁵-10⁻⁶ seconds (few microseconds) after Big Bang
- Temperature: T ~ 150-200 MeV, consistent with lattice T_c ≈ 156.5 MeV

**Baryon asymmetry η_B: VERIFIED ✓**

Current best value (PDG/Planck): **η_B = (6.12 ± 0.04) × 10⁻¹⁰**—the claim of "~6×10⁻¹⁰" is accurate.

---

## Mathematical framework and technical standards mostly verified

**Taylor expansion P(T, μ_B)/T⁴ = Σ_n c_n(T)(μ_B/T)^n: VERIFIED ✓**

Only **even powers** appear due to CP-symmetry. Convergence radius is μ_B/T ≲ 2.0-3.0 (temperature-dependent), limited by Lee-Yang edge singularities.

**Crossover parameterization T_c(μ_B) = T_c(0)[1 − κ_2(μ_B/T_c)² − κ_4(μ_B/T_c)⁴]: VERIFIED ✓**

Typical values: κ_2 ≈ 0.012-0.016, κ_4 ≈ 0.0003 (nearly zero).

**Pentium FDIV bug: NUANCED FINDING ⚠️**

Ken Shirriff's 2024 silicon analysis confirms:
- Radix-4 SRT with 2048-entry lookup table ✓
- **16 missing entries** found, but only **5 cause actual errors** (11 are in unreachable table regions)
- The claim "16 missing entries (not 5)" is technically correct but misleading—Intel's original "5 entries" referred to bug-causing entries, not total missing entries

**Cowan (2001) 4±1 items: VERIFIED ✓**

Full citation: Cowan, N. (2001). "The magical number 4 in short-term memory: A reconsideration of mental storage capacity." **Behavioral and Brain Sciences, 24(1), 87-114**.

**APS formatting: VERIFIED ✓**

- PRD column width: **8.5 cm (3⅜ in)** ✓
- Minimum font size: **2 mm** for capital letters and numerals ✓
- Line widths: at least 0.18 mm (0.5 point)

---

## STROBE and CASP checklists are definitively inapplicable

**STROBE**: Strengthening the Reporting of Observational Studies in Epidemiology—a 22-item checklist for medical cohort, case-control, and cross-sectional studies involving human populations. **Not applicable to particle physics.**

**CASP**: Critical Appraisal Skills Programme—UK-based checklists for healthcare research evaluation. **Not applicable to HEP.**

**What HEP actually uses**:
- **ATLAS/CMS Publication Guidelines**: Multi-stage internal review (analysis group → working group → collaboration-wide), blinding procedures, 5σ discovery threshold
- **STAR Publication Policies**: Physics Working Group review, Godparent Committee, Spokesperson approval
- **Lattice QCD standards**: Continuum extrapolation required, physical pion mass, finite volume assessment, FLAG criteria for quality

---

## Summary of corrections needed for the figure

| Claim | Original | Correction |
|-------|----------|------------|
| Wuppertal-Budapest chiral T_c | ~155 MeV | ~151 MeV |
| Wuppertal-Budapest deconf T_c | ~158 MeV | ~175-176 MeV |
| Clarke et al. T_CP | 105⁺⁸₋₁₈ MeV | 102⁺¹¹₋₂₃ MeV (published) |
| Clarke et al. μ_B^CP | 420⁺⁸⁰₋₃₅ MeV | 428⁺¹⁶²₋₇₄ MeV (published) |
| Clarke et al. collaboration | HotQCD? | Bielefeld-Parma |
| FRG uncertainty | ~10% | ΔT≈10 MeV, Δμ_B≈40 MeV |
| Pentium PLA entries | 5 missing entries wrong | 16 missing, 5 cause errors |
| LHC Pb-Pb μ_B | ~1 MeV | 0.71 ± 0.45 MeV (ALICE 2024) |

All other claims—including the critical Borsányi et al. exclusion paper, HotQCD crossover temperature, freeze-out parameterizations, experimental program parameters, and QGP physics properties—are verified accurate against primary sources.