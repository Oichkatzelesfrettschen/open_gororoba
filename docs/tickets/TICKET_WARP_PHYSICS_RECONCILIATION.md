# Plan: Warp Ring Physics Reconciliation

Owner: eirikr
Created: 2026-02-04
Status: TODO

## Goal
Elevate the "Silicon-Gold-Ice VIII" Warp Ring simulation from cinematic pseudo-physics to a rigorously defended Analogue Optics experiment (Phase 6.5 -> Phase 7.0).

## 1. Provenance & Bibliography (The "Spine")
- [ ] **Fetch Sources**: Download the 4 key citations to `data/external/papers/` (or `references/` if pdfs).
    - CSC KTH (Ray Theory)
    - Babar & Weaver (Gold n/k)
    - Pan et al (Ice Pressure)
    - NIST (Silicon)
- [ ] **Update BIBLIOGRAPHY.md**: Add formal citations.
- [ ] **Update PROVENANCE.local.json**: Record hashes.

## 2. Documentation Reframing (The "Story")
- [ ] **Update CLAIMS_EVIDENCE_MATRIX.md**:
    - Refine C-409/C-412.
    - Explicitly distinguish "Analogue Gravity" (Metamaterial) from "Metric Engineering" (Spacetime).
- [ ] **Create PHYSICS_MANIFEST.md**: A new doc detailing the "Layer A/B/C" model proposed in the critique.

## 3. Code Implementation (The "Engine")
- [ ] **Create `src/gemini_physics/optics/grin_solver.py`**:
    - RK4 Integrator.
    - `ComplexMaterial` class (Drude/Lorentz support).
    - `RayState` struct with intensity/power.
- [ ] **Create `src/scripts/visualization/animate_warp_v8_rigorous.py`**:
    - Import `grin_solver`.
    - Implement the "Power Pipeline" (Photon -> Plasmon heat -> Parton decay).
    - Annotate with "Analogue Optics" terminology.

## 4. Execution
- [ ] Run a single-frame render of V8 to verify physics and aesthetics.
