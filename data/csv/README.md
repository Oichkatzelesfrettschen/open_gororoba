# Data Index: CSV Datasets

This directory contains raw output from various simulation runs and theoretical models related to the Surreal-Noncommutative Quantum Gravity project.

## 1. Simulation Outputs (Active)
*   **`sedenion_field_metrics_3d.csv`**: Time-series data from `src/sedenion_field_sim.py`. Tracks `mean_associator` (non-associativity magnitude) and `mean_energy` over time in a 3D Sedenion field.
*   **`modular_chaos_N*.csv`**: Entropy evolution data from `src/modular_classical_sim.py` for different Hilbert space sizes $N$.
*   **`nonlocal_ZZ_slopes.csv`**: (Hypothetical/To-Be-Generated) Slope data from nonlocal spin chain runs.

## 2. Legacy/Static Data (Contextual)
*   **`Zero-Divisor_Adjacency_Matrix__*.csv`**: Adjacency lists for the zero-divisor graphs of Sedenions (16D), Pathions (32D), and Chingons (64D).
*   **`Spectrum_*.csv`**: Various FFT and spectral analysis dumps from previous "Gemini" sessions (e.g., `fft_2d_spectrum.csv`).
*   **`Refined_*.csv`**: Post-processed datasets from earlier "Refinement" phases.

## 3. Missing Metadata (Gaps)
*   **Units**: Most files lack headers specifying physical units (e.g., $t$ in seconds vs steps).
*   **Parameters**: Filenames often don't capture full parameter sets (e.g., coupling constants $J$, $\alpha$). Use the corresponding `src/` scripts to trace provenance.
