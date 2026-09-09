# Identifiability experiment compiler

This experiment integrates the 2026-09-09 nuisance-orthogonal distinguishability analysis with open_gororoba.

For target signature `t`, nuisance design matrix `N`, and whitening metric `W`, define `r = (I - P_N)t` and `I_eff = r^T W r`.

The computational result is conditional on the declared nuisance family. No finite nuisance basis proves that every conventional mechanism has been excluded.

## Executed synthetic results

- Symmetry/commutator design: identifiable fraction 0.995361, principal angle 84.479 degrees. At a deliberately modest synthetic amplitude, Monte Carlo power was 0.75975 at an empirically calibrated two-sided 1% false-positive rate.
- Boundary-differential design: initial identifiable fraction 0.228671, principal angle 13.219 degrees; after plausible cross-condition nuisance expansion, the surviving fraction collapsed to 0.00199202. This falsifies the present design as robustly identifiable.
- Exact target-shaped nuisance injection in the commutator experiment collapses the surviving fraction to 5.37e-16, as required.

`results_summary.csv` records the selected-design run performed in the chat compute environment. The compact Python kernel in this directory reproduces the model families and falsification invariants, but does not reproduce the greedy schedule-selection/Monte-Carlo pipeline that generated every CSV field.

## Run the compact prototype

`python -m pip install -r experiments/identifiability/requirements.txt`

`python experiments/identifiability/identifiability_compiler.py`

`python -m pytest experiments/identifiability/test_identifiability_compiler.py`

## Repository integration

Do not duplicate physics already present in `quantum_core` and `materials_core`. The next physical experiment must use the repository's Lifshitz and optical-material forward models, then build nuisance tangent vectors and experiment schedules around those predictions.

Claims in this directory are staging artifacts. Canonical claim state remains in the repository control-plane/registry workflow.

## Integration branch

This experiment is prepared on branch `chat-2026-09-09-identifiability`. It is intentionally not merged into `main` without review and native-build integration.
