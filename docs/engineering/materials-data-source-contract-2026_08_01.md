---
description: materials_data to materials_core source and generated-artifact contract
last_verified: 2026-08-01
evidence_class: source_contract
---

# Materials data source contract

The materials stack has two distinct surfaces. `crates/materials_data/data/`
owns machine-readable optical and crystallographic inputs. Its `build.rs`
compiles those inputs into `OUT_DIR/generated_*.rs`. `crates/materials_core`
consumes the generated constants and keeps the numerical models and public
API. The generated Rust files are build outputs, not alternate source data.

## Authority crosswalk

| Domain | Canonical input | Generator or consumer | Evidence boundary |
| --- | --- | --- | --- |
| Tabulated refractive index | `crates/materials_data/data/nk/*.csv` | `crates/materials_data/build.rs` -> `generated_nk_tables.rs` | CSV bytes and parser row counts require retained hashes |
| Optical parameters | `crates/materials_data/data/optical/*.toml` | `crates/materials_data/build.rs` -> `generated_optical_params.rs` | TOML values are source; emitted constants are derived |
| Space groups | `crates/materials_data/data/space_groups.csv` | `crates/materials_data/build.rs` -> `generated_crystal_tables.rs` | The build asserts exactly 230 rows |
| Crystal structures | `crates/materials_data/data/crystal/crystal_structures.toml` | `crates/materials_data/build.rs` -> `CRYSTAL_STRUCTURE_TABLE` | `materials_core::crystal_symmetry` converts names to typed enums |
| Character tables | `crates/materials_data/data/crystal/character_tables.toml` | `crates/materials_data/build.rs` -> `CHARACTER_TABLE_REGISTRY` | Matrix dimensions are checked during code generation |
| Runtime models | `crates/materials_core/src/optical_database/` and model modules | `materials_core` | Model code consumes generated constants and documents literature provenance |

## Bounded absence and migration boundary

The expected paths `crates/materials_core/data/optical_constants.toml` and
`crates/materials_core/data/crystal_symmetry.toml` are absent from the
checkout. This absence is bounded: `materials_core/Cargo.toml` depends on
`materials_data`, and the build script reads the `materials_data/data/` paths
listed above. No runtime lookup of the absent paths exists in the source
surface inspected for this contract.

`crates/materials_core/src/optical_database.rs` and
`crates/materials_core/src/crystal_symmetry.rs` still contain transcribed or
compatibility code. The contract does not declare those files obsolete. A
future migration must prove numerical parity between each extracted source
table and its generated replacement before removing a duplicate surface.

## Replay and closure

The source contract closes only when a retained manifest records the SHA-256
of every canonical input, the Rust toolchain, enabled features, and the
generated output or an equivalent parity result. The executable manifest gate
is:

```bash
cargo run -p gororoba_cli_data --bin experiment-manifest -- verify <manifest.toml>
```

The P2 structural row remains open until a bounded module slice has a parity
test, a source ownership record, and a measured reduction in duplicate or
monolithic code. The 7,600-line optical database is data-heavy and does not
qualify as a safe split target by line count alone.

## Evidence commands

```bash
find crates/materials_data/data -type f -print | sort
rg -n "materials_data::|cargo:rerun-if-changed=data/" crates/materials_core crates/materials_data
cargo test -p materials_data
cargo test -p materials_core
```

These commands identify the inputs, trace consumers, and validate both the
code-generation crate and its primary consumer. They do not replace source
hash retention for a research result.
