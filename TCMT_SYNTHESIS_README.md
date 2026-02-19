# Fano TCMT and Photon-Graviton Synthesis (2026-02-19)

## Quick Links to Synthesis Documents

All documents are generated from analysis of 15 arXiv papers (2004-2026) and are ready for immediate use.

### Documents in This Synthesis

1. **docs/TCMT_PHOTON_GRAVITON_SYNTHESIS.md** (24 KB)
   - Full 10-part technical synthesis with all details
   - Timeline, equations, gaps, problems, research directions
   - Best for: Deep research, writing papers, comprehensive reference
   - Read time: 20-30 minutes

2. **docs/TCMT_QUICK_REFERENCE.md** (9.5 KB)
   - Executive summary with one-sentence paper summaries
   - Problem table, roadmap, key equations by topic
   - Best for: Quick lookup, team briefing, decision-making
   - Read time: 5-10 minutes

3. **docs/TCMT_SYNTHESIS_INDEX.md** (12 KB)
   - Navigation guide for all three documents
   - Paper reference tables, usage scenarios
   - Best for: Finding specific information, understanding document structure
   - Read time: 5 minutes

4. **registry/TCMT_IMPLEMENTATION_PLAN.toml** (6.8 KB)
   - Actionable sprint planning with claims and deliverables
   - S56-1, S56-2, S56-3 with validation matrix
   - Best for: Sprint leads, implementation owners, claim registry
   - Read time: 5 minutes

---

## Key Finding: Missing Photon-Graviton TCMT Bridge

**Current State**:
- QFT side: Complete one-loop worldline photon-graviton amplitude (Ahmadiniaz et al. 2601.23279)
- TCMT side: Generic temporal coupled-mode framework (Ruan-Fan 0909.3323)
- **Gap**: No unified TCMT for photon-graviton with clear asymmetry parameter q_grav

**Impact**: CRITICAL (would bridge quantum field theory and resonant scattering)
**Feasibility**: HIGH (2-3 sprints)
**Blocker**: NONE (E-073 and C-823 already complete)

---

## Sprint 56-57 Roadmap

### S56-1: Photon-Graviton TCMT (2.5 weeks)
- Map worldline amplitude to TCMT parameters
- Derive gravitational asymmetry parameter q_grav
- Expected claims: C-824, C-825, C-826, C-827

### S56-2: Ward Identity Cross-Validation (1.5 weeks)
- Verify tadpole contribution via two methods
- Machine-precision validation
- Expected claims: C-828, C-829, C-830

### S56-3: THz Band-Folding Fano (2.5 weeks)
- Implement band-folding model from Ge et al. 2502.03077
- Demonstrate no-symmetry-breaking Fano generation
- Expected claims: C-831, C-832, C-833

---

## 7 Unsolved Problems (Priority Ranked)

| Problem | Feasibility | Sprints | Impact |
|---------|-------------|---------|--------|
| Photon-Graviton TCMT | HIGH | 2-3 | CRITICAL |
| Nonlinear TCMT (Kerr) | MEDIUM | 3-4 | HIGH |
| Loss in Ultra-High-Q | MEDIUM | 3-4 | HIGH |
| Nonlocal TCMT Dispersion | MEDIUM | 2-3 | MEDIUM |
| Strong-field Gravity | VERY HIGH | 6+ | MEDIUM |
| Three-loop Amplitudes | VERY HIGH | 6+ | MEDIUM |
| Fano-BIC in Loss | MEDIUM | 2-3 | MEDIUM |

---

## 15 Papers at a Glance

### Tier 1: Foundational (3 papers)
- **0909.3323** Ruan & Fan (2009) - TCMT framework
- **gr-qc/0412095** Bastianelli & Schubert (2004) - Worldline QFT
- **2505.00396** Maksimov et al. (2025) - TCMT symmetry constraints

### Tier 2: Major Extensions (4 papers)
- **1105.2503** Gallinet & Martin (2011) - Plasmonic Fano
- **1805.09265** Bogdanov et al. (2018) - BIC mechanism
- **1901.03122** Abujetas et al. (2019) - BIC continuum limit
- **0710.5572** Bastianelli et al. (2007) - One-loop structure

### Tier 3: Specialized (4 papers)
- **2303.12264** Fan et al. (2023) - Lattice engineering
- **2307.01186** Overvig et al. (2023) - Nonlocal STCMT
- **2502.03077** Ge et al. (2025) - Band-folding Fano
- **2601.23279** Ahmadiniaz et al. (2026) - Complete photon-graviton

### Tier 4: Niche/Exploratory (4 papers)
- **1701.02929** Kristensen et al. (2017) - Quasinormal modes
- **1704.06477** Gao et al. (2017) - Optical forces
- **1410.4148** Bjerrum-Bohr et al. (2014) - Tree-level gravity
- **2401.08346** Anninos et al. (2024) - Cosmological oscillations

---

## Recent Game-Changer: Maksimov et al. 2505.00396 (2025)

Published just this year, this paper derives generic constraints on TCMT parameters from first principles:

**Energy Conservation**: kappa1 - kappa2 = gamma_l

**Time-Reversal Symmetry**: Pairs of real zeros in transmission

**Innovation**: Unidirectional guided modes (asymmetric coupling without symmetry breaking)

This enables rational design of photon-graviton TCMT systems.

---

## How to Use These Documents

**5-minute read**: TCMT_QUICK_REFERENCE.md

**15-minute decision**: 
1. TCMT_QUICK_REFERENCE.md (Problems section)
2. TCMT_IMPLEMENTATION_PLAN.toml (S56-1, S56-2, S56-3)

**30-minute research prep**:
1. TCMT_QUICK_REFERENCE.md (full)
2. TCMT_PHOTON_GRAVITON_SYNTHESIS.md (Part 1, 2, 3)

**45-minute deep dive**:
1. All of TCMT_PHOTON_GRAVITON_SYNTHESIS.md
2. TCMT_SYNTHESIS_INDEX.md for navigation

---

## Implementation Priority

**Sprint 56-57**: S56-1 (Photon-Graviton TCMT) - CRITICAL PATH
- Highest impact (bridges QFT and photonics)
- No blockers
- High feasibility (2-3 weeks)

**Sprint 58-60**: S58-4 (Nonlinear TCMT) and S59-5 (BIC Loss Engineering)

**Sprint 61+**: S61-6 (Multi-loop Amplitudes) - Long-term research frontier

---

## Key Insights

1. **TCMT matured 2009-2025**: From angular momentum channels (Ruan-Fan) to generic symmetry constraints (Maksimov)

2. **BIC physics well-understood**: Q-factor scaling laws, lattice engineering, fabrication-robust designs

3. **Nonlocal effects emerging**: Band-folding enables Fano without traditional symmetry breaking

4. **Photon-graviton amplitude complete**: One-loop done (3 diagrams), higher loops remain for future

5. **Generic TCMT constraints just discovered**: Energy and time-reversal symmetry enable design-first approach

6. **Bridge is ready to build**: QFT (2601.23279) and TCMT (0909.3323 + 2505.00396) are both mature

---

## Next Actions

- [ ] Read TCMT_QUICK_REFERENCE.md (5 min)
- [ ] Review TCMT_IMPLEMENTATION_PLAN.toml (5 min)
- [ ] Decide: Start S56-1 this week or next sprint?
- [ ] Assign: Owners for S56-2, S56-3
- [ ] Create: Claims C-824..C-833 in claims.toml when ready

---

**Generated**: 2026-02-19
**Papers**: 15 (all open access arXiv)
**Status**: Complete and ready for implementation
**Contact**: See TCMT_SYNTHESIS_INDEX.md for document details
