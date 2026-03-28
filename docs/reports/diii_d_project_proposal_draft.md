# DIII-D Project Information Form: Draft Proposal

## Project Lead
- **Full Legal Name:** Eric Hilgart
- **Email:** eric@hilg.art
- **ORCID ID:** (registered at orcid.org)

## Project Title
Cayley-Dickson Associator as Magnetic Field Phase-Geometry Diagnostic for ELM and Disruption Detection

## Funding Organization
Internal (Self-funded independent research)

## Funding Award Identifier
Internal

## Research Topical Area of Interest
Control of Damaging Transients

## Project Description

This project proposes to validate a novel magnetic field diagnostic based on the
Cayley-Dickson (CD) algebraic associator for detecting ELMs and disruption
precursors in DIII-D tokamak discharges. The CD associator embeds time series
of magnetic field measurements (e.g., Mirnov coil B_theta) into 32-dimensional
hypercomplex algebra via Takens delay embedding, then computes the
non-associativity norm A = ||(a*b)*c - a*(b*c)||. This scalar diagnostic tracks
the phase-geometry complexity of cross-channel magnetic coupling.

### Preliminary Results

The CD associator has been validated across multiple data sources:

1. **BOUT++ simulations (completed):** Hasegawa-Wakatani drift-wave turbulence
   shows 78:1 A ratio between turbulent and laminar states. Peeling-ballooning
   ELM simulation (elm_pb, 68x64 tokamak grid) shows 2.4:1 modulation through
   ELM growth. Blob2d coherent filament shows 4x lower A than turbulence,
   confirming the coherent-structure suppression phenotype.

2. **GRMHD accretion disk (completed):** nubhlight MRI simulation shows 20:1
   A ratio at MRI onset, tracking the topology transition from laminar seed
   field to turbulent MHD.

3. **MAST experimental data (completed):** FAIR-MAST public Mirnov coil data
   (shot 30420, 650K samples) shows 15.8% temporal variation in A across the
   discharge, with A dipping at plasma formation (field ordering) and peaking
   during flat-top (maximum MHD activity). Pure Rust pipeline (zarrs + cd_kernel).

4. **Heliospheric validation (completed):** 128-week THEMIS sweep with 4498
   curated crossings: CD alone achieves F1=0.860 (best individual detector).
   CD+PVI union achieves 86.6% detection. Seven spacecraft missions validated.

5. **Embedding dimension optimization:** 16D/32D/64D sweep shows 32D is optimal
   (78:1 contrast vs 16x at 64D), justifying the pathion algebra dimension.

### What DIII-D Data Would Add

DIII-D Mirnov coil data from well-documented ELM-containing and disruption
discharges would provide:

1. **ELM detection validation on a large conventional tokamak** (vs. MAST
   spherical tokamak). DIII-D's extensive ELM database with labeled ELM times
   allows precision-recall analysis.

2. **Disruption precursor detection:** Does A rise before disruption onset?
   DIII-D's disruption database with locked-mode and thermal-quench timing
   provides ground truth.

3. **Cross-machine generalization:** Combined with MAST (spherical) and
   BOUT++ (simulation), DIII-D data would complete the conventional-tokamak
   validation.

4. **Comparison with existing diagnostics:** CD vs. Mirnov spectrograms,
   locked-mode amplitude, and standard MHD analysis tools.

### Requested Data
- Mirnov coil time series (B_theta, B_r) for 10-20 well-documented discharges
  spanning: Type I ELMs, Type III ELMs, locked modes, disruptions, and
  quiescent H-mode (control)
- Discharge metadata: ELM times, disruption onset, confinement mode transitions

### Software
All analysis tools are open-source Rust code at
github.com/Oichkatzelesfrettschen/open_gororoba. The CD kernel, Zarr/NetCDF
readers, and analysis binaries are pure Rust with no proprietary dependencies.
