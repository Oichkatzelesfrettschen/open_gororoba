//! # Audit-grade synthesis of the chat corpus on LHC light-ion QGP results and reproducible data-mirror pathways
//!
//! ## Scope, epistemic standard, and audit method
//!
//! This treatise treats the full chat corpus (user narrative + all linked references + the assistant’s earlier “data layers” framing) as a single technical document whose factual claims must be verifiable, whose data-access paths must be operationally reproducible, and whose implied models must be made explicit. The target style matches the “PBS Space Time–style” constraint: mechanism-first, technically faithful, and—where possible—decision procedures over rhetoric.
//!
//! The audit’s central organizing abstraction is a **three-layer evidence stack**:
//!
//! 1. **Operations layer**: what beams ran, when, and at what energies (machine schedule, integrated luminosities).
//! 2. **Results layer**: what collaborations claim they observed (papers, preprints, conference notes).
//! 3. **Data layer**: where the numeric payload lives (HEPData tables, INSPIRE/DOIs, plots repositories, open-data releases, code/workflow preservation).
//!
//! ## Reconstructed claim graph and fact-check outcome for the full chat
//!
//! ### Claims about what ran in 2025 and why it matters
//!
//! **Claim A1 (run content and sequence)**: In early July 2025, the LHC executed a special light-ion program with proton–oxygen (p–O), oxygen–oxygen (O–O), and neon–neon (Ne–Ne) collisions, including setup/commissioning in between.
//! **Status**: Verified. CERN states the schedule explicitly: 29 June–9 July 2025.
//!
//! **Claim A2 (physics motivation)**: The light-ion program targets the “system-size question” (how small a system can still exhibit QGP-like collectivity and parton energy loss) and simultaneously supports cosmic-ray interaction modeling in the atmosphere via p–O.
//! **Status**: Verified.
//!
//! **Claim A3 (end-of-Run-3 context)**: 2025 was the final full year of Run 3 operations; the 2026 run was planned March–June followed by a long shutdown toward HL-LHC.
//! **Status**: Verified.
//!
//! ### Claims about “what was seen” in O–O / Ne–Ne
//!
//! **Claim B1 (CMS: parton-energy-loss signature in O–O via charged-hadron suppression)**: CMS measured the nuclear modification factor $R_{AA}$ for O–O at $\sqrt{s_{NN}}=5.36$ TeV and found suppression below unity, with a minimum around $p_T\sim 6$ GeV of about $0.69 \pm 0.04$.
//! **Status**: Verified (CMS-HIN-25-008, HEPData 166013).
//!
//! **Claim B2 (ALICE: hints of jet quenching in O–O via neutral-pion suppression)**: ALICE reported first hints of jet quenching in O–O using neutral pion production shortly after the 2025 run.
//! **Status**: Verified as "first hints".
//!
//! **Claim B3 (ATLAS: flow signals and nuclear-shape sensitivity in O–O / Ne–Ne)**: ATLAS measured anisotropic flow coefficients $v_n$ (n=2–4) in O–O and Ne–Ne, using both two-particle correlations and four-particle subevent cumulants to suppress non-flow, and finds patterns consistent with nuclear-shape differences (enhanced $v_2$ in central Ne–Ne consistent with prolate neon).
//! **Status**: Verified (arXiv:2509.05171).
//!
//! **Claim B4 (ALICE: geometry-driven flow in light ions)**: ALICE measured $v_2$ and $v_3$ in O–O and Ne–Ne at 5.36 TeV and interprets results as sensitive to nuclear geometry.
//! **Status**: Verified (arXiv:2509.06428).
//!
//! **Claim B5 (LHCb: neon “bowling-pin” shape extraction via flow in fixed-target Pb–Ne)**: LHCb measured $v_2$ and $v_3$ using multiparticle cumulants in fixed-target PbNe and PbAr at $\sqrt{s_{NN}}=70.9$ GeV, finding larger $v_2$ in PbNe consistent with the prolate neon ground-state shape.
//! **Status**: Verified (arXiv:2509.12399).
//!
//! ### Claims about QCD critical point relevance
//!
//! **Claim C1 (critical point at LHC/top RHIC energies constrained away)**: The STAR experiment Beam Energy Scan II results “rule out” a QCD critical point in regions of the phase diagram accessed at LHC and top RHIC energies, leaving open lower-energy/higher-$\mu_B$ possibilities.
//! **Status**: Verified (CERN Courier, Bedangadas Mohanty).
//!
//! ## Falsifiable Theses
//!
//! - **Thesis T1 (operation schedule)**: The 2025 light-ion run executed p–O, O–O, and Ne–Ne collisions in the 29 June–9 July 2025 window.
//! - **Thesis T2 (CMS O–O suppression magnitude)**: In O–O at $\sqrt{s_{NN}}=5.36$ TeV, CMS measured charged-hadron $R_{AA}(p_T)$ with a local minimum $R_{AA}\approx 0.69\pm 0.04$ near $p_T\approx 6$ GeV.
//! - **Thesis T5 (neon deformation imprint)**: ATLAS observes enhanced $v_2$ in central Ne–Ne vs O–O consistent with prolate neon deformation (arXiv:2509.05171).
//! - **Thesis T10 (QCP exclusion region framing)**: BES-II precision measurements rule out a QCD critical point in the region of the QCD phase diagram accessed at LHC and top RHIC energies.
//! - **Thesis T11 (wake/medium response result provenance)**: Z–hadron correlations in Pb-Pb exhibit modifications consistent with a hydrodynamic wake / medium response to hard probes (Phys. Lett. B 140120).
//!
//! ## Reproducibility Roadmap
//!
//! 1. **Corpus Normalization**: Source registry maintained in `registry/bibliography.toml`.
//! 2. **Slide/Video Mirror**: Seminar hub (Indico 1586852) and CDS talk records.
//! 3. **Results-Data Mirror**: HEPData numeric files (DOIs stored in registry).
//! 4. **Model-to-Data Reconstruction**: 
//!     - **Geometry**: TGlauberMC 3.3.2 (Loizides update) / TRENTo.
//!     - **Dynamics**: JETSCAPE / Hydro + Transport.
//!     - **Hadronic Afterburner**: SMASH.
//!
//! ## Mathematical Backbone
//!
//! ### Nuclear modification factor $R_{AA}$
//! $$R_{AA}(p_T,y) = \frac{1}{\langle T_{AA} \rangle} \frac{1}{N_{\text{evt}}^{AA}} \frac{d^2N^{AA}}{dp_T\,dy} / \frac{d^2\sigma^{pp}}{dp_T\,dy}$$
//!
//! ### Flow coefficients $v_n$
//! $$\frac{dN}{d\phi} \propto 1 + 2\sum_{n=1}^{\infty} v_n \cos\left[n(\phi-\Psi_n)\right]$$
//!
//! ### Multi-particle cumulants
//! - $v_n\{2\} = \sqrt{c_n\{2\}}$
//! - $v_n\{4\} = \sqrt[4]{-c_n\{4\}}$
//!
//! Implemented in `open_gororoba/crates/qgp_scaling`.
//!
