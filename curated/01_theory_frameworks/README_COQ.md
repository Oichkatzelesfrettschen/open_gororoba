# Coq Proofs: The Computational Substrate

## Overview
These `.v` files (`confine_theorems_*.v`) are a **large inventory of theorem statements** about a
proposed "Discrete Causal Lattice" model (threads, rights, delegation).

Important: the `confine_theorems_*.v` files in this repo currently contain **statements only** (no
definitions and no proofs). The `make coq` workflow compiles a generated "axioms" version of these
statements so that the Coq toolchain can at least typecheck the declarations and keep the interface
stable while proper proofs are developed.

## Physical Interpretation
While phrased in Computer Science terms ("Threads", "Rights"), this model corresponds to a **Discrete Spacetime** where:
*   **Thread** = A Point in Spacetime (Event).
*   **Send/Receive** = Causal Connection (Light Cone).
*   **Right/Grant** = Conservation of Information (No Signaling Condition).

## Lacunae & Resolution
*   **Original Claim**: These proofs verify "Quantum Gravity".
*   **Correction**: At present, they do **not** provide such proofs. They are a structured set of claims
    that would need (1) a formal model and (2) actual proofs.

### Build / typecheck
Run:
```bash
make coq
```

This generates `confine_theorems_*_axioms.v` (ignored by git) and compiles them with Coq.

## Next Steps
To make this physically rigorous, we define the mapping:
$$ \text{Thread}_i \leftrightarrow x^\mu_i \in \mathcal{M} $$
$$ \text{reachable}(i, j) \iff \Delta s^2_{ij} \ge 0 $$
