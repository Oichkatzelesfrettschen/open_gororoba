# Issue #127: Full Build-Time Data Generation Scope

Date: 2026-05-13
Parent: #127 "PH-MOD: convert crystal_symmetry.rs + optical_database.rs to build-time data generation"

## WHY

`optical_database.rs` and `crystal_symmetry.rs` were monoliths because Rust
forced data definitions to be inline. The PH-MOD splits (commits 3a265bcc -
7c420319) carved the monoliths into single-domain submodules but the _data_
(material parameters, character tables, structure entries) still lives in
Rust source files. Build-time codegen moves that data into machine-readable
TOML/CSV registries that:

1. **Edit-as-data**: literature updates become TOML diffs rather than Rust
   source edits, which means non-programmers can contribute.
2. **Per-PR auditable**: a structure-entry addition is one TOML row, not a
   3-line `make()` Rust call.
3. **Reusable across crates**: any future crate can include the same data
   without re-implementing tables.
4. **Test-pinned**: each codegen pass emits a count, a hash, and a schema
   signature that regression-protects entries.

## WHAT (recursive scope)

Phases 2-4 of `materials_data` build-time codegen are already complete:

- Phase 2 (n,k tables, #56): COMPLETE
- Phase 3 (Drude + Lorentz optical params, #57): COMPLETE
- Phase 4 (230 ITA space groups, #58): COMPLETE

The remaining phases under #127 are 5 through 8:

### Phase 5: MineralMetadata TOML codegen

Status: PARTIAL. tourmaline already codegen'd via lorentz_models.toml
(commit 5dde8021). The remaining 34 `*_metadata()` accessors are still
Rust-source-only.

Scope:

- New file `crates/materials_data/data/optical/mineral_metadata.toml`.
- One `[[mineral]]` entry per accessor with all 12 MineralMetadata fields.
- `build.rs::emit_mineral_metadata()` to emit `pub static <NAME>_METADATA:
  MineralMetadata = MineralMetadata { ... }` (or a single
  `pub static MINERAL_METADATA: &[MineralMetadata]` array indexed by name).
- `materials_core::optical_database::*_metadata()` accessors become thin
  shims pointing to the codegen consts.

Estimate: 35 entries × 12 fields TOML; ~300 lines of build.rs codegen.

### Phase 6: CrystalStructureInfo TOML codegen

Status: NOT STARTED. 109 entries currently inline as Rust `make()` calls.

Scope:

- New file `crates/materials_data/data/crystal/crystal_structures.toml`.
- One `[[structure]]` entry per structure with all 15 CrystalStructureInfo
  fields (name, sg_num, sg_sym, point_group, lattice_system, centering,
  a/b/c, alpha/beta/gamma, atoms_per_unit_cell, density, reference).
- `build.rs::emit_crystal_structures()` reads TOML, validates, emits
  `pub static CRYSTAL_STRUCTURES: &[CrystalStructureInfo]`.
- `crystal_symmetry::known_crystal_structures()` becomes
  `materials_data::CRYSTAL_STRUCTURES.iter().cloned().collect()`.

Estimate: 109 entries × 15 fields TOML (~1600 lines); ~150 lines build.rs.

### Phase 7: Character tables TOML/CSV codegen

Status: NOT STARTED. 1865 lines in `character_tables.rs` are the largest
remaining hardcoded data block in the workspace.

Scope:

- TOML schema: per point group, list of conjugacy classes (name, count,
  representative), list of irreducible representations (label, dimension,
  functions), characters matrix (one Vec<(re, im)> per (irrep, class)).
- Build.rs validation: every point group must have a square table
  (n_irreps == n_classes), Schur orthogonality `sum n_k * |chi_i(C_k)|^2
  == |G|`, etc.
- `CharacterTable::for_point_group()` becomes a generated O(1) lookup.

Estimate: 32 point groups × variable-sized tables (~600 lines TOML);
~400 lines build.rs (parsing + orthogonality validation).

### Phase 8: Sellmeier coefficient TOML codegen

Status: NOT STARTED. Sellmeier dispersion models for LiNbO3, fused silica
etc. are currently inline in optical_database.rs.

Scope:

- New file `crates/materials_data/data/optical/sellmeier_models.toml`.
- Per material: `[[sellmeier]] name + a + b + c arrays`.
- `materials_core` constructors call the codegen consts.

Estimate: ~15 Sellmeier materials; ~200 lines TOML; ~80 lines build.rs.

## HOW (execution plan)

Each phase is a self-contained PR that:

1. **TOML schema**: define the data file format. Document field semantics
   in the file header.
2. **Reference port**: copy values from the current Rust source into TOML
   ensuring byte-identical numerical roundtrip (tests pin this).
3. **build.rs emitter**: parse TOML, validate fields + invariants, emit
   Rust source into `OUT_DIR`. Include in `materials_data/src/lib.rs`.
4. **Caller refactor**: replace Rust source data with the codegen const
   references. Tests must still pass without value drift.
5. **Regression tests**: pin codegen counts + hash signature so future
   edits can't silently lose entries.

## Dependencies / ordering

- Phases 5, 6, 7, 8 are independent of each other.
- Each phase depends on the corresponding submodule split (already done
  via #138/#139).
- Phase 7 has the highest LOC reduction in materials_core source (1865
  lines moved to TOML) but the highest implementation risk (character-
  table orthogonality validation is non-trivial).
- Phase 6 has the cleanest scope-to-impact ratio.

## Acceptance criteria

For each phase:

- `cargo nextest run -p materials_core --lib` passes with no value drift.
- `cargo clippy -- -D warnings` clean.
- Workspace inheritance preserved in any new Cargo.toml entries.
- `materials_data::tests::*` test count grows by at least 1 invariant test
  per new codegen const family.

## Out of scope

- COD federated OPTIMADE query for further crystal-structure expansion
  beyond 109 entries. This requires a network-fetch step at build time
  which violates the project's deterministic-build policy. Solved instead
  by hand-curated TOML augmented as needed.
- Sellmeier extension to anisotropic materials (separate phase if needed).
- Migration of `materials_data` to a `data/v1/` versioned schema. Future
  consideration after all 4 phases land.
