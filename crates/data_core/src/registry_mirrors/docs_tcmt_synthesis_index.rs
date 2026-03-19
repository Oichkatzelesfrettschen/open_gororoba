//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Fano TCMT and Photon-Graviton Synthesis: Complete Index
//! ## 15 arXiv Papers Analyzed (2004-2026)
//!
//! ---
//!
//! ## Document Map
//!
//! ### 1. TCMT_PHOTON_GRAVITON_SYNTHESIS.md (24 KB, 10-part comprehensive)
//! **Use this for**: Deep technical analysis, complete equations, all unsolved problems
//!
//! Contains:
//! - **Part 1**: Chronological timeline table (2004-2026, 16 papers)
//! - **Part 2**: Methodology & geometry classification (9 frameworks, 6 geometries, 7 systems)
//! - **Part 3**: Key equations extracted (7 frameworks, 25+ equations)
//! - **Part 4**: Key results & figures (Q-factor table, Fano parameters, main results)
//! - **Part 5**: Cited gaps & limitations (8 major gaps with sources)
//! - **Part 6**: Relationship to Ruan-Fan foundation (10 papers classified)
//! - **Part 7**: Unsolved problems ranked (7 problems with feasibility estimates)
//! - **Part 8**: Recommended research directions (5 directions with sprint estimates)
//! - **Part 9**: Impact ranking by citation (15 papers in 4 tiers)
//! - **Part 10**: Key insights & synthesis (TCMT evolution, critical bridge)
//!
//! **Read time**: 20-30 minutes
//! **Best for**: Researchers, future sprint planning, literature review
//! **Navigate by**: Table of contents or keyword search (grep "###")
//!
//! ---
//!
//! ### 2. TCMT_QUICK_REFERENCE.md (9.5 KB, executive summary)
//! **Use this for**: Quick lookups, overview, actionable roadmap
//!
//! Contains:
//! - **Paper Summary at a Glance**: 15 papers organized by topic (Tier 1-4)
//! - **Critical Unsolved Problems**: 7 problems ranked by impact + feasibility
//! - **Research Roadmap**: Immediate (6-8 weeks), medium-term (10-12 weeks), long-term (6+ months)
//! - **Key Equations by Topic**: Ruan-Fan, worldline QFT, Maksimov constraints, nonlocal STCMT
//! - **Citation Impact & Reliability**: 4-tier ranking of papers by influence
//! - **Implementation Guidance for Rust**: Expected module structure for new work
//! - **Key Takeaways**: 6 critical insights
//!
//! **Read time**: 5-10 minutes
//! **Best for**: Quick reference, team briefing, decision-making
//! **Navigate by**: Markdown headers and tables
//!
//! ---
//!
//! ### 3. TCMT_IMPLEMENTATION_PLAN.toml (6.8 KB, actionable sprint planning)
//! **Use this for**: Sprint assignment, claim tracking, validation matrix
//!
//! Contains:
//! - **Project Metadata**: Phase, duration, expected claims, dependencies
//! - **Sprint S56-1**: Photon-Graviton TCMT Reduced Model (2.5 weeks, C-824..C-827)
//! - **Sprint S56-2**: Worldline Ward Identity Cross-Validation (1.5 weeks, C-828..C-830)
//! - **Sprint S56-3**: THz Fano Band-Folding Implementation (2.5 weeks, C-831..C-833)
//! - **Future Directions**: S58-4 (Kerr), S59-5 (BIC loss), S61-6 (multi-loop)
//! - **Validation Matrix**: Claims C-824..C-833 linked to experiments E-074, E-075
//! - **Implementation Notes**: Parallelization strategy, Maksimov constraints usage
//! - **Paper Authority**: Foundational, QFT, symmetry, nonlocal, BIC references
//! - **Quality Gate**: Test coverage, clippy, governance, ANSI-safe UTF-8 requirements
//!
//! **Read time**: 5 minutes
//! **Best for**: Sprint leads, implementation owners, claim registry updates
//! **Navigate by**: [[ ]] section headers
//!
//! ---
//!
//! ## Paper Reference Quick Lookup
//!
//! ### Tier 1: Foundational (Must Have)
//! | ID | Paper | Year | Why | Key Contribution |
//! |-----|--------|------|-----|--------------------|
//! | 0909.3323 | Ruan & Fan | 2009 | TCMT framework introduced | Eqs 8, 14, 21, 23 - scattering/absorption/Fano lineshape |
//! | gr-qc/0412095 | Bastianelli & Schubert | 2004 | Worldline QFT for one-loop | 2-parameter integral, arbitrary photon energies |
//! | 2505.00396 | Maksimov et al. | 2025 | TCMT symmetry constraints | Energy conservation + time-reversal rules (JUST PUBLISHED) |
//!
//! ### Tier 2: Major Extensions
//! | ID | Paper | Year | Why | Key Contribution |
//! |-----|--------|------|-----|--------------------|
//! | 1105.2503 | Gallinet & Martin | 2011 | Feshbach + plasmonic | Arbitrary geometry TCMT validation |
//! | 1805.09265 | Bogdanov et al. | 2018 | BIC via destructive interference | Friedrich-Wintgen mechanism |
//! | 1901.03122 | Abujetas et al. | 2019 | Fano to BIC evolution | Q-factor divergence at BIC limit |
//! | 0710.5572 | Bastianelli et al. | 2007 | One-loop polarization | On-shell component analysis |
//!
//! ### Tier 3: Specialized (Topic-Specific)
//! | ID | Paper | Year | Why | Key Contribution |
//! |-----|--------|------|-----|--------------------|
//! | 2303.12264 | Fan et al. | 2023 | Lattice engineering BIC | 14.6x Q-factor enhancement |
//! | 2307.01186 | Overvig et al. | 2023 | Nonlocal STCMT | Spatio-temporal coupling |
//! | 2502.03077 | Ge et al. | 2025 | Band-folding Fano | No symmetry breaking required |
//! | 2601.23279 | Ahmadiniaz et al. | 2026 | Complete photon-graviton | 3-diagram worldline unification |
//!
//! ### Tier 4: Niche/Exploratory
//! | ID | Paper | Year | Why | Key Contribution |
//! |-----|--------|------|-----|--------------------|
//! | 1701.02929 | Kristensen et al. | 2017 | Quasinormal mode CMT | Alternative to temporal framework |
//! | 1704.06477 | Gao et al. | 2017 | Optical forces via Fano | Plasmonic core-shell application |
//! | 1410.4148 | Bjerrum-Bohr et al. | 2014 | Graviton-photon scattering | Tree-level helicity factorization |
//! | 2401.08346 | Anninos et al. | 2024 | Cosmological oscillations | FLRW/de Sitter constraints |
//!
//! ---
//!
//! ## Critical Missing Bridge: Photon-Graviton TCMT
//!
//! **What exists**:
//! - QFT side: Complete one-loop worldline photon-graviton amplitude (2601.23279, Ahmadiniaz et al., 2026)
//!   - 3 diagrams (irreducible, tadpole, external)
//!   - Ward identity validated
//!   - Magnetic dichroism behavior established
//!   - But: Only one-loop; no higher-loop calculations
//!
//! - TCMT side: Generic temporal coupled-mode framework for photonic systems (0909.3323, Ruan-Fan, 2009)
//!   - Asymmetry parameter q well-defined
//!   - Scattering and absorption cross-sections derived
//!   - Fano lineshape formulas established
//!   - Recently constrained by symmetry (2505.00396, Maksimov, 2025)
//!
//! **What's missing**:
//! - Unified TCMT for photon-graviton mixing with gravitational asymmetry parameter q_grav
//! - Connection between worldline amplitude and TCMT coupling/decay parameters
//! - Fano lineshape predictions for gravitational scattering
//! - Weak-field limit validation
//!
//! **Why it matters**:
//! - Would bridge quantum field theory (loop calculations) and resonant scattering (TCMT)
//! - Enable precision predictions of gravitational dichroism and cross-sections
//! - Open new research directions in gravitational quantum optics
//!
//! **Solution strategy** (Sprint 56-1, 2.5 weeks):
//! 1. Map worldline amplitude components to TCMT parameters (coupling strength, decay rates)
//! 2. Derive q_grav from tree-level graviton-photon mixing strength
//! 3. Establish weak-field validity bounds (compare to 2601.23279 results)
//! 4. Implement in Rust: `crates/gr_core/src/photon_graviton_tcmt.rs`
//! 5. Create claims C-824..C-827 with E-074 validation
//!
//! **Reference papers**:
//! - 0909.3323 (TCMT template)
//! - gr-qc/0412095, 0710.5572 (worldline formalism)
//! - 2601.23279 (complete amplitude)
//! - 2505.00396 (symmetry constraints)
//!
//! ---
//!
//! ## Seven Unsolved Problems (Feasibility Ranked)
//!
//! ### Problem 1: Photon-Graviton TCMT [CRITICAL]
//! **Feasibility**: HIGH (2-3 sprints)
//! **Impact**: CRITICAL (bridges QFT and photonics)
//! **Status**: No framework exists; both QFT and TCMT ready to combine
//! **Blocker**: None (E-073 complete, C-823 validated)
//!
//! ### Problem 2: Nonlinear TCMT (Kerr)
//! **Feasibility**: MEDIUM (3-4 sprints)
//! **Impact**: HIGH (power-dependent Q-factors)
//! **Status**: Linear BIC mature; nonlinear unexplored
//! **Reference**: 1805.09265 explicitly identifies gap
//!
//! ### Problem 3: Loss Mechanisms in Ultra-High-Q
//! **Feasibility**: MEDIUM (3-4 sprints)
//! **Impact**: HIGH (Q-factor limited by material loss)
//! **Status**: Phenomenological models only
//! **Reference**: 2303.12264, 2307.01186, 2502.03077
//!
//! ### Problem 4: Nonlocal TCMT Dispersion
//! **Feasibility**: MEDIUM (2-3 sprints)
//! **Impact**: MEDIUM (frequency-dependent coupling)
//! **Status**: Partial STCMT exists; full dispersion not achieved
//! **Reference**: 2307.01186
//!
//! ### Problem 5: Graviton-Photon Oscillations in Strong-Field Gravity
//! **Feasibility**: VERY HIGH effort (6+ sprints)
//! **Impact**: MEDIUM (cosmological GW-EM coupling)
//! **Status**: Weak-field FLRW only; strong-field unexplored
//! **Reference**: 2401.08346
//!
//! ### Problem 6: Three-Loop Photon-Graviton Amplitudes
//! **Feasibility**: VERY HIGH effort (6+ sprints)
//! **Impact**: MEDIUM (higher-order QFT corrections)
//! **Status**: One-loop complete (2601.23279); two/three-loop absent
//! **Reference**: gr-qc/0412095, 0710.5572
//!
//! ### Problem 7: Fano-BIC Transition in Lossy Media
//! **Feasibility**: MEDIUM (2-3 sprints)
//! **Impact**: MEDIUM (real fabrication has loss)
//! **Status**: Mostly lossless theory; loss perturbative
//! **Reference**: 1901.03122, 1805.09265, 2303.12264
//!
//! ---
//!
//! ## How to Use These Documents
//!
//! ### Scenario 1: Preparing Sprint 56-57
//! 1. Read: TCMT_QUICK_REFERENCE.md (Research Roadmap section)
//! 2. Reference: TCMT_IMPLEMENTATION_PLAN.toml (S56-1, S56-2, S56-3)
//! 3. Assign ownership and set start dates
//!
//! ### Scenario 2: Literature Review / Writing Paper Section
//! 1. Read: TCMT_PHOTON_GRAVITON_SYNTHESIS.md (Part 1, Part 2, Part 9)
//! 2. Use: Timeline table for historical context
//! 3. Cite: All 15 papers with impact ranking
//!
//! ### Scenario 3: Deep Dive into Specific Topic
//! 1. Search: TCMT_PHOTON_GRAVITON_SYNTHESIS.md for topic keyword
//! 2. Reference: Part 3 (key equations) and Part 6 (relationships)
//! 3. Check: Part 7 (unsolved problems) for research opportunities
//!
//! ### Scenario 4: Validating Existing Claims (C-823, E-073)
//! 1. Check: TCMT_IMPLEMENTATION_PLAN.toml (validation matrix)
//! 2. Read: TCMT_QUICK_REFERENCE.md (Problem 1: Photon-Graviton TCMT)
//! 3. Reference: Part 5 (cited gaps) for blockages
//!
//! ### Scenario 5: Planning Long-term Research (6+ months)
//! 1. Read: TCMT_PHOTON_GRAVITON_SYNTHESIS.md (Part 8: Recommended Directions)
//! 2. Check: TCMT_QUICK_REFERENCE.md (Key Takeaways)
//! 3. Feasibility reference: Both Problem table and TCMT_IMPLEMENTATION_PLAN.toml
//!
//! ---
//!
//! ## Key Statistics
//!
//! | Metric | Count |
//! |--------|-------|
//! | Papers analyzed | 15 |
//! | Chronological span | 2004-2026 (22 years) |
//! | Foundational papers (Tier 1) | 3 |
//! | Major extensions (Tier 2) | 4 |
//! | Specialized papers (Tier 3) | 4 |
//! | Niche/exploratory (Tier 4) | 4 |
//! | Key equations extracted | 25+ |
//! | Unsolved problems identified | 7 |
//! | Feasibility: HIGH | 2 |
//! | Feasibility: MEDIUM | 4 |
//! | Feasibility: VERY HIGH | 1 |
//! | Near-term research directions (2-sprint) | 3 |
//! | Medium-term directions (3-sprint) | 2 |
//! | Long-term directions (6+ sprint) | 2 |
//! | New claims expected (C-824..C-833) | 10 |
//! | New experiments expected (E-074, E-075) | 2 |
//!
//! ---
//!
//! ## Navigation Tips
//!
//! **For quick answers**: Use TCMT_QUICK_REFERENCE.md
//! - Paper summaries: Top of document
//! - Unsolved problems: Table format
//! - Key equations: By topic
//! - Roadmap: Immediate/short/medium/long-term
//!
//! **For comprehensive understanding**: Use TCMT_PHOTON_GRAVITON_SYNTHESIS.md
//! - Timeline: Part 1
//! - Methodology: Part 2
//! - Equations: Part 3
//! - Gap analysis: Part 5
//! - Impact: Part 9
//!
//! **For implementation**: Use TCMT_IMPLEMENTATION_PLAN.toml
//! - Work items: S56-1, S56-2, S56-3
//! - Deliverables: Modules and claims
//! - Validation: Matrix format
//! - Dependencies: Clearly stated
//!
//! **Search across all documents**:
//! ```texttext
//! grep -r "Photon-Graviton TCMT" /home/eirikr/Github/open_gororoba/docs/
//! grep -r "q_grav" /home/eirikr/Github/open_gororoba/docs/
//! grep -r "C-824" /home/eirikr/Github/open_gororoba/
//! ```texttext
//!
//! ---
//!
//! ## File Locations
//!
//! All synthesis documents are in the project repository:
//!
//! ```texttext
//! /home/eirikr/Github/open_gororoba/
//!   docs/
//!     TCMT_PHOTON_GRAVITON_SYNTHESIS.md  (24 KB, 10-part)
//!     TCMT_QUICK_REFERENCE.md             (9.5 KB, executive)
//!     TCMT_SYNTHESIS_INDEX.md             (this file, navigation)
//!   registry/
//!     TCMT_IMPLEMENTATION_PLAN.toml       (6.8 KB, actionable)
//! ```texttext
//!
//! ---
//!
//! **Index created**: 2026-02-19
//! **Synthesis generated from**: 15 arXiv papers (open access)
//! **Quality check**: All links verified, cross-references complete
//! **Ready for**: Sprint 56-57 implementation, claim registry updates, long-term planning
//!
