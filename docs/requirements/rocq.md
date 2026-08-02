<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: Rocq Proof Checking

This repo contains `.v` files under `proofs/` and companion curated theory
surfaces.

Install the pinned Rocq and Flocq packages in the same `opam` switch:

```bash
opam install rocq-core.9.1.1 rocq-stdlib.9.1.1 coq-flocq.4.2.2
eval "$(opam env --switch rocq-9.1.1)"
```

Then run:

```bash
make rocq-project-check
make rocq-makefile-check
make -C proofs vos
make -C proofs vok
```

Notes:

- The proof Makefile requires `OPAM_SWITCH_PREFIX`, which keeps Rocq and the
  compiled Flocq objects in one opam switch.
- `coq-flocq.4.2.2` supplies the `Flocq.Core` dependency used by
  `FP24Representable.v`.
- `make rocq-project-check` compares every `.v` file under `proofs/theories`
  and `proofs/verified` with `_RocqProject` in both directions.
- `make rocq-makefile-check` treats any `rocq makefile` warning as an error.
  This specifically blocks logical-root drift such as "No common logical root."
- Keep proof checking as an explicit toolchain lane with its own versioned
  requirements.
