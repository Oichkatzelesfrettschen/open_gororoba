<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Rocq Proof Checking

This repo contains `.v` files under `proofs/` and companion curated theory
surfaces.

Install Rocq, for example with `opam`:

```ignore
opam install rocq
```

Then run:

```ignore
make rocq
```

Notes:

- The Makefile checks for `coqc` on `PATH`.
- Keep proof checking as an explicit toolchain lane with its own versioned
  requirements.
