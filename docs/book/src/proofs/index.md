<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/book_docs.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/canonical/control_plane.sqlite3 -->

# Proof Inventory

The Rocq proof inventory is indexed through the canonical SQLite control
plane and exported into human-readable surfaces. The standard docs refresh
path emits the theorem mirror and theorem index together with the rest of the
control-plane web docs bundle.

Primary sources for this lane:

- `proofs/_RocqProject` for the live proof-file manifest
- `proofs/verified/*.v` for kernel-checked proof files
- `docs/THEOREMS.md` for the generated compatibility view

Recommended workflow:

```sh
cargo run -p gororoba_cli_data --bin provenance -- \
  --db registry/canonical/control_plane.sqlite3 \
  export-control-plane

cargo run -p gororoba_cli_data --bin registry-emit -- \
  control-plane-docs
```

After export, use `docs/THEOREMS.md` as the web-readable theorem index and
the proof files under `proofs/` as the canonical source material. Use
`docs/generated/THEOREMS_REGISTRY_MIRROR.md` when you want the fuller
registry-style mirror.
