<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements: Coq Proof Checking

This repo contains `.v` files under `curated/01_theory_frameworks/`.

Install Coq (example using `opam`):
```bash
opam install coq
```

Then run:
```bash
make coq
```

Notes:
- The Makefile checks for `coqc` on PATH (Rocq/Coq compiler).
- The current `confine_theorems_*.v` are statement inventories without proofs.
- `make coq` generates `confine_theorems_*_axioms.v` (gitignored) to typecheck the interface.
