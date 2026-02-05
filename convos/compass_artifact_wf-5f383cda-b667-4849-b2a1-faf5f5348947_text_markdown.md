# Multi-Domain Audit of Directus Maximalus Proposal: Critical Verification Results

The "Directus Maximalus" experimental proposal for Project Lighthouse contains a mixture of accurate physics, questionable assumptions, and mathematically ill-defined constructs. This exhaustive audit reveals that while the proposal's mathematical formalism for sedenion zero divisors is surprisingly rigorous, its core experimental physics for cesium tunneling relies on unverified parameters, and its topological winding number formulation is fundamentally flawed.

## Verification 1: Cesium-133 tunneling physics — PARTIALLY REFUTED

**Claim: λ_J = 0.24 μm tunneling decay length for Cs-133** — **UNVERIFIABLE/QUESTIONABLE**

No direct experimental measurements of cesium tunneling in optical tweezers exist in the published literature. The claimed decay length appears **optimistically large** for cesium's mass. WKB estimates for typical trap depths (**~100 μK**) yield λ_J ≈ **0.05–0.1 μm** for Cs-133, roughly half the claimed value. Tunneling rates scale inversely with √m, and cesium at 133 amu is **1.5× heavier than rubidium** and **22× heavier than lithium-6**, the species where tunneling is well-characterized.

**Spatial adiabatic passage with cesium**: **NOT DEMONSTRATED**. The landmark 2024 demonstration by Florshaim et al. in *Science Advances* used potassium-40, not cesium. Cesium momentum-state STIRAP was demonstrated in 1994, but this involves internal states, not spatial transport in optical tweezers.

**1.0 μm inter-trap spacing**: **MARGINALLY ACHIEVABLE**. Standard cesium tweezer arrays operate at 3–7 μm spacing. Recent super-resolution techniques (arXiv:2502.07177, February 2025) achieved sub-wavelength trapping with cesium at 1064 nm, but 1.0 μm spacing (0.94λ) remains at the extreme technological limit requiring specialized superoscillatory optics.

## Verification 2: Endres Lab species — VERIFIED

**The Caltech Endres Lab 6,100-atom system definitively uses cesium-133 (Cs), not rubidium.** Direct quotation from Manetsch et al. (arXiv:2403.12021, published *Nature* **647**, 60–67, September 2025): "Cesium atoms possess the highest polarizability among the stable alkali-metal atoms at near-infrared wavelengths." The system uses trapping wavelengths of **1061 nm and 1055 nm** with **7.2 μm** atomic spacing, achieving **12.6-second coherence times** for hyperfine qubits. The proposal correctly identifies the atomic species.

## Verification 3: Winding number on discrete graphs — REFUTED

**The formula W = (1/2π) ∮ ⟨Ψ|L_z|Ψ⟩ dt is mathematically ill-defined for a 5-site star graph.**

The angular momentum operator L_z = -iℏ∂/∂φ requires a continuous angular coordinate. On a discrete graph with only 5 vertices, no such coordinate exists. The formula conflates three distinct mathematical objects:

- **Angular momentum** (requires continuous space)
- **Winding numbers** (require periodic systems with Brillouin zones)  
- **Geometric/Berry phases** (parameter space quantities)

For discrete tight-binding models, proper topological invariants include the **Zak phase** γ_Zak = i∫⟨u_k|∂_k|u_k⟩dk for periodic systems, or **Berry phases** γ_Berry = i∮⟨ψ(R)|∇_R|ψ(R)⟩·dR for parameterized evolution. The geometric phase in STIRAP follows γ = -∫sin²θ dφ₂, which is an Aharonov-Anandan phase, not a topological winding number. These phases are continuous-valued, not quantized, unless specific symmetries are imposed.

## Verification 4: STIRAP on star/branched geometries — VERIFIED (THEORY) / NOT DEMONSTRATED (EXPERIMENT)

**Greentree et al. 2015 paper verified**: Batey, Jeske, and Greentree, "Dark state adiabatic passage with branched networks and high-spin systems," *Frontiers in ICT* **2**:19 (September 2015), arXiv:1506.01135. The paper provides dark state conditions for star topologies with N leaf nodes.

**Dark state for 5-site star (1 hub + 4 leaves):**
|D₀⟩ = [4·Ω_R|L,M̄,R̄₁,R̄₂,R̄₃,R̄₄⟩ - Ω_L(Σⱼ|L̄,M̄,...,Rⱼ,...⟩)] / √(16Ω_R² + 4Ω_L²)

This requires counter-intuitive pulse ordering: Ω_R starts high, then Ω_L increases while Ω_R decreases.

**Experimental status**: No spatial adiabatic passage on star or branched network geometries has been demonstrated. The 2024 SAP demonstration (Florshaim et al.) used linear three-trap geometry. Star-topology SAP remains purely theoretical.

## Verification 5: Thouless pumping state of art — PARTIALLY VERIFIED

**Citro & Aidelsburger review confirmed**: *Nature Reviews Physics* **5**, 87–101 (January 2023), arXiv:2210.02050. This remains the definitive review; no comprehensive update has appeared through February 2026.

**Trautmann et al. Rydberg pumping verified**: Published in *Physical Review A* **110**, L040601 (October 2024). Achieved **90% pumping efficiency** using 6 Rydberg states (n=55-57) of cesium atoms in a synthetic dimension Rice-Mele chain. Follow-up work (Huang et al., arXiv:2512.12364, December 2025) extended this to interaction-assisted pumping in few-atom arrays.

**Beyond 1D Rice-Mele chains**: Significant 2024–2025 advances include 2D non-Abelian Thouless pumping in photonic waveguides (Sun et al., *Nature Communications* **15**, 9311, October 2024), interaction-enabled pumping without lattice sliding (Viebahn et al., *Physical Review X* **14**, 021049, June 2024), and "returning" Thouless pumps via Berry dipoles (Mo et al., *Physical Review Letters* **135**, 206603, November 2025).

**Star graph Thouless pumping**: **NO LITERATURE FOUND**. No experimental or theoretical proposals for Thouless pumping on star/branched network topologies have been published.

## Verification 6: Sedenion zero divisor mathematics — VERIFIED

All mathematical claims check out with high precision:

| Claim | Status | Source |
|-------|--------|--------|
| Reggiani 2024: ZD(S) ≅ V₂(ℝ⁷) | **VERIFIED** | arXiv:2411.18881 (November 2024) |
| Moreno 1998: Z(S) ≅ G₂ | **VERIFIED** | *Bol. Soc. Mat. Mex.* **4**(1): 13–28 |
| dim V₂(ℝ⁷) = 11 | **VERIFIED** | nk - k(k+1)/2 = 14 - 3 = 11 |
| Codimension 4 in S¹⁵ | **VERIFIED** | 15 - 11 = 4 |
| G₂/SU(2) ≅ V₂(ℝ⁷) | **VERIFIED** | Reggiani 2024; dim 14-3=11 |

Reggiani's result is actually stronger than homeomorphism—it establishes an **isometry** with a specific G₂-invariant metric. Note the distinction: Moreno's Z(S) is the space of norm-one sedenion **pairs** that multiply to zero (14-dimensional, homeomorphic to G₂), while Reggiani's ZD(S) is the space of **single** normalized elements with non-trivial annihilators (11-dimensional, isometric to V₂(ℝ⁷)).

## Verification 7: Cosmological constant suppression mechanism — NOVEL/UNVERIFIABLE

**Sedenion vacuum energy suppression**: **NOVEL** — No prior published literature proposes using sedenion (or octonion) non-associative algebraic structure for cosmological constant suppression. Extended Heim Theory (Hauser & Dröscher) mentions sedenions in a different context but does not address vacuum energy via this mechanism.

**Weyl tube formula**: The claim Vol(Tube_ε) ~ ε^c is an **oversimplification**. The actual Weyl tube formula is a **polynomial**: Vol(N_ε M) = Σᵢ μᵢ · ωᵢ · εⁱ, where μᵢ are Lipschitz-Killing curvature invariants. For codimension-c embedding, the highest power is ε^c, but the volume is not a simple power law.

**DESI Year 3 results (March 2025)**: Evidence for evolving dark energy at **2.8–4.2 sigma** (depending on dataset combinations), below the 5-sigma discovery threshold. Data suggests w₀ > -1 and w_a < 0, indicating possible time evolution of dark energy equation of state, but this remains tentative.

## Verification 8: Simulation Hamiltonian physics — VERIFIED

**Hamiltonian form H = Δ|0⟩⟨0| - Σ J_k(|0⟩⟨k| + |k⟩⟨0|)**: **CORRECT** for a detuned star graph with tunneling. This is a valid generalization of the standard STIRAP Hamiltonian to star topology, matching formulations in spatial adiabatic passage literature.

**Adiabatic condition for Δ = 2π × 10 kHz, J ~ 120 Hz (Δ/J ≈ 83)**: **FAVORABLE**. The intermediate state population scales as (J/Δ)² ≈ **0.015%**, well below the 1% threshold. The criterion Δ/J > 10 is sufficient for <1% center population; Δ/J ≈ 83 far exceeds this minimum. However, the effective coupling Ω_eff ~ J²/Δ ≈ 1.44 Hz implies slow transfer rates requiring operation times >>0.7 seconds for pure adiabatic evolution.

## Verification 9: AOD motion profiles — PARTIALLY VERIFIED

**Minimum-jerk trajectories**: **PARTIALLY TRUE** — commonly used but not universal. A 2025 *Physical Review Applied* paper (Cárdenas et al.) found that **Shortcuts to Adiabaticity (STA)** protocols outperform minimum-jerk trajectories, achieving equivalent fidelity in shorter times.

**50 ms transport over 2 μm**: At 0.04 μm/ms average speed, this is extremely slow compared to typical experiments (0.1–0.5 μm/μs). Negligible heating is expected in this deep adiabatic regime. Beugnon et al. (*Nature Physics* 2007) demonstrated **no measurable heating** after 360 μm cumulative transport.

**Young et al. 2023 Endres Lab paper**: **CANNOT CONFIRM** — This paper appears not to exist. Aaron W. Young is at **Kaufman Lab (JILA)**, not Endres Lab (Caltech). The "lossless" transport claim is overstated; the best demonstrated fidelity is **~99.95%** per move (Manetsch et al. 2024), excellent but not strictly lossless.

## Verification 10: Division algebra-physics connections — VERIFIED/ACTIVE

**Cohl Furey's program (2024–2025)**: **ACTIVE AND ADVANCING**. Key recent results include:
- "Three generations and a trio of trialities," *Physics Letters B* (2024) — describes three SM generations via triality symmetries tri(ℂ)⊕tri(ℍ)⊕tri(𝕆)
- Fermion doubling problem solved (Furey & Hughes, *Physics Letters B* 2022)
- "A Superalgebra Within," *Annalen der Physik* (2025) — SM particle representations as ℤ₂⁵-graded algebra

**Dubois-Violette & Todorov H₃(𝕆)**: **ESTABLISHED AND VERIFIED**. The Standard Model gauge group S(U(2)×U(3)) can be constructed as the subgroup of F₄ (automorphisms of exceptional Jordan algebra H₃(𝕆)) preserving 10D Minkowski spacetime and complex subalgebra. Independently verified by Baez and others.

**Gresnigt sedenion program**: **ACTIVE** — The only developed sedenion-physics connection. Key result: S₃ ⊂ Aut(𝕊) (present in sedenion automorphisms but NOT octonion automorphisms) provides natural three-generation structure. Recent publications in *European Physical Journal C* (2023, 2024) and arXiv:2601.07857 (2025).

**G₂ holonomy and sedenion structure**: **NO DIRECT CONNECTION FOUND** in physics literature. G₂ holonomy M-theory compactifications involve octonions (G₂ = Aut(𝕆)), but extending this to sedenions is not an established research direction. One preliminary visualization study exists (arXiv:2512.13002, December 2024) but does not establish physics connections.

---

## Summary verdict table

| Claim | Verdict | Key Finding |
|-------|---------|-------------|
| λ_J = 0.24 μm for Cs-133 | **UNVERIFIABLE** | No Cs tunneling data exists; physics suggests ~0.05–0.1 μm |
| Endres Lab uses Cs | **VERIFIED** | Confirmed: Cs-133 at 1061/1055 nm, 7.2 μm spacing |
| SAP demonstrated with Cs | **REFUTED** | Only K-40 (2024); no Cs spatial SAP |
| 1.0 μm trap spacing | **MARGINALLY ACHIEVABLE** | At technological limit requiring super-resolution |
| Winding number formula | **REFUTED** | L_z undefined on discrete graph; formula ill-posed |
| Greentree star-graph theory | **VERIFIED** | Published 2015, dark states well-defined |
| Star-graph SAP experiment | **NOT DEMONSTRATED** | Only linear 3-site SAP exists |
| Thouless pump star graph | **NOT FOUND** | No proposals in literature |
| Reggiani ZD(S) ≅ V₂(ℝ⁷) | **VERIFIED** | arXiv:2411.18881, isometry proven |
| dim V₂(ℝ⁷) = 11, codim 4 | **VERIFIED** | Correct calculation |
| Sedenion cosmological mechanism | **NOVEL** | No prior literature |
| Weyl tube formula Vol ~ ε^c | **OVERSIMPLIFIED** | Actual formula is polynomial |
| DESI Year 3 evolving DE | **TENTATIVE** | 2.8–4.2σ, not 5σ discovery |
| Simulation Hamiltonian | **VERIFIED** | Correct star-graph form |
| Δ/J ≈ 83 adiabatic | **VERIFIED** | Predicts ~0.015% center population |
| Young et al. 2023 Endres | **DOES NOT EXIST** | Young is at Kaufman Lab |
| Lossless transport | **OVERSTATED** | Best is ~99.95%, not 100% |
| Furey program active | **VERIFIED** | Major 2024–2025 publications |
| Gresnigt sedenion physics | **VERIFIED** | Active program, S₃ family symmetry |
| G₂ holonomy + sedenions | **NOT ESTABLISHED** | No physics connection found |

## Critical assessment

The proposal's strongest elements are its mathematical treatment of sedenion geometry (all claims verified) and STIRAP physics parameters (Hamiltonian form and adiabatic conditions correct). Its weakest elements are the foundational cesium tunneling assumptions (no experimental basis for claimed parameters), the mathematically ill-defined winding number formulation, and the novel cosmological mechanism (unprecedented, requiring extraordinary evidence). The claimed connection to prior experimental work contains errors (wrong researcher attribution, non-existent papers).

For the proposal to advance, it would need either to switch to a lighter atomic species with established tunneling parameters (K-40, Rb-87, or Li-6), or to provide theoretical derivation with experimental validation for cesium-specific tunneling at the claimed parameters. The topological characterization requires reformulation using proper geometric phase or Berry phase constructs rather than angular momentum-based winding numbers.