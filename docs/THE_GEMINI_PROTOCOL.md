# The Gemini Protocol: A Unified Physics Engine

## Abstract
This project ("Gemini") is a research-style sandbox exploring hypotheses about how **higher-dimensional non-associative algebras** (specifically octonions and sedenions) might relate to toy models in quantum cosmology, particle physics, and metamaterials. Any physics-facing interpretation should be treated as **speculative** unless backed by first-party sources and reproducible experiments.

## 1. Mathematical Foundation
We have verified computationally (see `tests/`) that:
*   **Sedenions are non-associative**: $(ab)c \neq a(bc)$ occurs in seeded random trials.
*   **Sedenions have zero divisors**: pairs of non-zero vectors $a, b$ exist such that $a \cdot b = 0$.
*   **Physical interpretation (speculative)**: mapping non-associativity/zero divisors onto "open-system" dynamics is a hypothesis, not an established result.

## 2. Quantum Cosmology & Renormalization
*   **Big Bounce (prototype)**: see `src/gemini_physics/cosmology.py` for a toy model; this is not yet tied to observational constraints.
*   **RG Flow (prototype)**: see `src/gemini_physics/renormalization.py` for an exploratory script; any "fixed point" claim needs an explicit, reproducible measurement.

## 3. The Standard Model Map
We projected the 16D Sedenion basis onto the Standard Model gauge groups.
*   **Findings (speculative)**: any claim that "associator flux" explains confinement requires a formal definition and citations; treat current notes as exploratory.

## 4. Zero-Divisor Metamaterials & Tensor Networks
*   **Tensor Networks (prototype)**: see `src/gemini_physics/tensor_networks.py`; "validation" requires a saved benchmark and clear criteria.
*   **Metamaterial prediction (speculative)**: "zero-divisor absorbers" is an engineering hypothesis; it must be connected to real materials data + established absorber/non-reciprocity literature before being treated as plausible.

## Conclusion
The Gemini Protocol is an attempt to bridge abstract algebra and computational experiments. Where the repo proposes physical predictions, they should be phrased as **test plans** with explicit baselines and citations.
