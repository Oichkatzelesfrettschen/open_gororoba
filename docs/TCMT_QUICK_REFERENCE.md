# Fano TCMT and Photon-Graviton: Quick Reference
## Synthesis of 15 arXiv Papers (2004-2026)

---

## Paper Summary at a Glance

### TCMT Foundational (Must Read)
1. **0909.3323** - Ruan & Fan (2009) - TEMPORAL COUPLED-MODE THEORY FOR FANO RESONANCE
   - Foundational TCMT framework for small scatterers
   - Key equations: Eqs 8, 14, 21, 23 (Fano line shape, cross-sections)
   - Cylindrical/spherical geometries, angular momentum channels
   - Left for future: Multiple channels, arbitrary geometries

2. **2505.00396** - Maksimov et al. (2025) - GENERIC TCMT CONSTRAINTS FROM SYMMETRY
   - First-principles derivation of TCMT parameter constraints
   - Energy conservation and time-reversal symmetry impose limits
   - Unidirectional guided modes in metasurfaces
   - Complements and extends Ruan-Fan to modern photonic systems

### Worldline QFT (Photon-Graviton)
3. **gr-qc/0412095** - Bastianelli & Schubert (2004) - PHOTON-GRAVITON MIXING (ONE-LOOP)
   - Manifestly covariant worldline formalism
   - Two-parameter integral representation
   - Arbitrary photon energies and EM field strengths
   - Higher-loop formalism not yet developed

4. **0710.5572** - Bastianelli et al. (2007) - ONE-LOOP WORLDLINE STRUCTURE
   - On-shell polarization component analysis
   - Spinor/scalar loop distinctions
   - Limiting case expressions

5. **2601.23279** - Ahmadiniaz et al. (2026) - COMPLETE PHOTON-GRAVITON AMPLITUDE
   - First complete three-diagram calculation (irreducible, tadpole, external)
   - Worldline unification of all contributions
   - Magnetic dichroism: tadpole does NOT affect observable
   - Ward identity validation

### BIC Physics (Bound States in Continuum)
6. **1805.09265** - Bogdanov et al. (2018) - FANO RESONANCES IN HIGH-INDEX DIELECTRICS
   - Friedrich-Wintgen destructive interference suppresses radiation
   - Multipole decomposition analysis
   - Gap: Nonlinear BIC (Kerr) dynamics unexplored

7. **1901.03122** - Abujetas et al. (2019) - FANO TO BIC EVOLUTION IN ROD DIMERS
   - Q-factor divergence approaching BIC limit
   - Terahertz rod dimer metasurfaces
   - Detuning parameter governs BIC transition

8. **2303.12264** - Fan et al. (2023) - HYBRID LATTICE BIC ENGINEERING
   - Lattice engineering reduces scattering loss by 14.6x
   - Fabrication tolerance improvement (practical)
   - Unit cell symmetry control

### Nonlocal Extensions
9. **2307.01186** - Overvig et al. (2023) - SPATIO-TEMPORAL CMT FOR NONLOCAL METASURFACES
   - STCMT framework extends TCMT to spatial dependence
   - Wavefront-shaping resonances
   - Gap: Full dispersion in TCMT not yet achieved

10. **2502.03077** - Ge et al. (2025) - BAND FOLDING FANO GENERATION
    - No symmetry breaking required for Fano
    - Metal-dielectric hybrid THz metasurfaces
    - Band folding creates Fano via guided mode resonance

### Plasmonic & Optical Applications
11. **1105.2503** - Gallinet & Martin (2011) - FANO IN PLASMONIC NANOSTRUCTURES
    - Feshbach formalism for asymmetric resonances
    - Material losses control resonance contrast
    - Extends Ruan-Fan to arbitrary geometries

12. **1704.06477** - Gao et al. (2017) - HIGH-Q FANO RESONANCES VIA MODE COUPLING
    - Core-shell plasmonic with optical gain
    - Quadrupole-dipole coupling generates giant negative optical forces
    - Gain compensation for loss

### Gravitational Extensions
13. **2401.08346** - Anninos et al. (2024) - GRAVITON-PHOTON OSCILLATIONS IN COSMOLOGY
    - FLRW and de Sitter mixing analysis
    - Conformal invariance constraints severely limit oscillations
    - Gap: Strong-field gravity unexplored

14. **1410.4148** - Bjerrum-Bohr et al. (2014) - GRAVITON-PHOTON COMPTON SCATTERING
    - Helicity factorization methods
    - Tree-level amplitudes only
    - Massless limit properties

### Alternative Frameworks
15. **1701.02929** - Kristensen et al. (2017) - CMT VIA FIELD EQUIVALENCE PRINCIPLE
    - Quasinormal mode alternative to TCMT
    - Cavity-waveguide coupling
    - Less intuitive but valid framework

---

## Critical Unsolved Problems (Ranked by Impact + Feasibility)

| Problem | Impact | Feasibility | Sprint Estimate | Why It Matters |
|---------|--------|-------------|-----------------|----------------|
| **Photon-Graviton TCMT** | CRITICAL | HIGH | 2-3 | Bridges QFT and resonant scattering; enables gravitational dichroism predictions |
| **Nonlinear TCMT (Kerr)** | HIGH | MEDIUM | 3-4 | Power-dependent Q-factors; BIC bistability |
| **Loss in Ultra-High-Q** | HIGH | MEDIUM | 3-4 | Material loss limits all real devices |
| **Nonlocal TCMT Dispersion** | MEDIUM | MEDIUM | 2-3 | Frequency-dependent coupling effects |
| **Strong-Field Graviton-Photon** | MEDIUM | VERY HIGH | 6+ | Cosmological GW-EM coupling |
| **Three-Loop Photon-Graviton** | MEDIUM | VERY HIGH | 6+ | Higher-order corrections |
| **Fano-BIC in Lossy Media** | MEDIUM | MEDIUM | 2-3 | Real-world fabrication |

---

## Research Roadmap: Immediate Action Items

### Sprint 56-57 (Next 6-8 weeks)
1. **Photon-Graviton TCMT Reduced Model**
   - Inputs: E-073 amplitude, Ruan-Fan equations
   - Output: C-824..C-827 (4 new claims)
   - Module: `photon_graviton_tcmt.rs`
   - Reference: 0909.3323 + gr-qc/0412095 + 0710.5572 + 2601.23279

2. **Worldline Ward Identity Cross-Validation**
   - Verify 2601.23279 tadpole via two independent methods
   - Machine-precision validation
   - Reference: gr-qc/0412095 + 0710.5572 + 2601.23279

3. **THz Fano Band-Folding Implementation**
   - Band folding model from 2502.03077 in Rust
   - Geometry-dependent resonance tuning
   - Reference: 2502.03077 + 2307.01186

### Sprint 58-60 (Next 10-12 weeks)
4. **Nonlinear TCMT Framework (Kerr)**
   - Bifurcation analysis, power-dependent Q
   - Reference: 1805.09265 + 1901.03122 + 0909.3323

5. **BIC Loss Engineering for Photon-Graviton**
   - Apply lattice engineering (2303.12264) to gravitational system
   - Reference: 2303.12264 + 2505.00396 + 2601.23279

### Long-term (6+ months)
6. **Higher-loop Amplitudes** (symbolic integration)
7. **Strong-field cosmology** (curved spacetime worldline)

---

## Key Equations by Topic

### Ruan-Fan TCMT (0909.3323)
- **Eq 8**: Temporal coupled-mode equation (base form)
- **Eq 14**: Scattering cross-section = sigma_s
- **Eq 21**: Absorption cross-section = sigma_a
- **Eq 23**: Fano line shape with asymmetry q

### Worldline QFT (gr-qc/0412095)
- **2-parameter integral**: A(s1, s2) = integral representation for one-loop amplitude
- **Covariant path integral**: Manifestly relativistic approach

### Photon-Graviton Complete (2601.23279)
- **3 diagrams**: Irreducible (J1, J2, J3) + tadpole + external-leg
- **Ward identity**: Sum of all diagrams satisfies gravitational gauge invariance

### Maksimov Constraints (2505.00396)
- **Energy conservation**: kappa1 - kappa2 = gamma_l (coupling/decay relation)
- **Time-reversal symmetry**: Pairs of real zeros in transmission

### Nonlocal STCMT (2307.01186)
- **Spatio-temporal extension**: Couples spatial gradients to temporal modes

---

## Citation Impact & Reliability

### Tier 1 (Foundational, 80+ citations each)
- 0909.3323 (200+) - Ruan-Fan TCMT
- gr-qc/0412095 (80+) - Worldline QFT
- 2505.00396 (emerging) - Maksimov constraints

### Tier 2 (Major extensions, 50-120 citations)
- 1105.2503 (120+) - Plasmonic Fano
- 1805.09265 (100+) - BIC mechanism
- 1901.03122 (90+) - BIC continuum limit
- 0710.5572 (50+) - One-loop structure

### Tier 3 (Specialized, 20-50 citations)
- 2303.12264 (35+) - Lattice engineering
- 2307.01186 (40+) - Nonlocal STCMT
- 2502.03077 (30+) - Band-folding Fano
- 2601.23279 (emerging) - Complete photon-graviton

### Tier 4 (Niche, <20 citations)
- 1701.02929 (15+) - Quasinormal modes
- 1704.06477 (20+) - Optical forces
- 1410.4148 (10+) - Tree-level gravity
- 2401.08346 (12+) - Cosmological oscillations

---

## Implementation Guidance for Rust

### For Photon-Graviton TCMT Module
```rust
// Expected file structure
crates/gr_core/src/
  photon_graviton/
    mod.rs                 -- Exports
    types.rs               -- Coupling/decay parameters
    tcmt_equations.rs      -- TCMT solver
    fano_lineshape.rs      -- Asymmetry q calculation
    cross_sections.rs      -- Absorption/scattering
    tests.rs               -- E-073 validation
```

### For Nonlinear Extensions
```rust
// Kerr-TCMT additions
kerr_tcmt/
  coupling_nonlinear.rs    -- Chi-3 coupling terms
  power_dependent_q.rs     -- Q as function of intensity
  bistability.rs           -- Bifurcation analysis
```

---

## Key Takeaways

1. **Ruan-Fan 0909.3323 is foundational**: All subsequent TCMT work builds on it or acknowledges constraints it establishes.

2. **Photon-Graviton TCMT is MISSING**: We have the QFT amplitude (2601.23279) and the TCMT framework (0909.3323), but no unified theory bridging them.

3. **Maksimov 2505.00396 just changed the game**: Generic symmetry constraints on TCMT parameters enable rational design (2025 result, very recent).

4. **BIC physics is mature in linear regime**: Non-linear and loss mechanisms remain open.

5. **Cosmological mixing is constrained**: Conformal invariance severely limits graviton-photon oscillations in expanding universes (2401.08346).

6. **Nonlocal effects are emerging**: STCMT and band-folding show Fano generation without traditional symmetry breaking (2023-2025).

---

**Last updated**: 2026-02-19
**Analysis includes**: All 15 papers with abstracts, key equations, and methodology
**Recommended next action**: Start Sprint 56 with Photon-Graviton TCMT module (2-3 weeks)
