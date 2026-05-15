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
make rocq-makefile-check
make rocq
```

Notes:

- The Makefile checks for `coqc` on `PATH`.
- The proof Makefile also requires `rocq` on `PATH` for the native Rocq 9.x
  proof lane.
- Prefer `eval "$(opam env --switch rocq-9.1.1)"` for the pinned local switch,
  but the proof lane may run in CI if a compatible `rocq` binary is already on
  `PATH`.
- `make rocq-makefile-check` treats any `rocq makefile` warning as an error.
  This specifically blocks logical-root drift such as "No common logical root."
- Keep proof checking as an explicit toolchain lane with its own versioned
  requirements.
