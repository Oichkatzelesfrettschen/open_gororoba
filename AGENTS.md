# AGENTS

Embedded core contract.

```toml
[contract]
authority = "AGENTS.md"
format = "embedded_core_plus_overflow"
style = "concise_compact_infodense_imperative"

[identity]
repo_kind = "pure_rust_plus_toml"
control_plane = "registry/*.toml"
runtime = "Cargo workspace crates/*"
python_role = "verifier_support_only"
rust_profiles_doc = "registry/agents_contract.toml:[rust_optimization]"
rust_modularity_doc = "registry/agents_contract.toml:[rust_modularization|crate_discovery|dependency_alignment|iteration_loop]"

[shims]
claude = "CLAUDE.md -> AGENTS.md"
gemini = "GEMINI.md -> AGENTS.md"

[overflow]
file = "registry/agents_contract.toml"
machine_docs = ["registry/crate_source_scout.toml", "registry/mcp_server_matrix.toml"]
edit_core_here = true
edit_overflow_for = "extended_matrices_profiles_inventory"
editor_reminder = "For additional instructions: edit registry/agents_contract.toml first; keep all policy text compact, infodense, imperative."

[paths]
workspace = "Cargo.toml"
lockfile = "Cargo.lock"
tasks = "Makefile"
registry = "registry/"
verification = "src/verification/"
analysis = "src/scripts/analysis/"
generated_docs = "build/docs/generated/"
hooks = ".githooks/"

[markdown]
track_default = false
allowlist = ["AGENTS.md", "CLAUDE.md", "GEMINI.md", "README.md", "**/README.md"]
rules = [
  "Allowlist markdown files remain in place as TOML-owned entrypoints.",
  "No manual markdown outside allowlist.",
  "When unmanaged markdown appears, migrate content into canonical TOML via normalize_* scripts and refresh markdown inventory/owner maps.",
  "Do not use archival move/delete as the default policy response for unmanaged markdown; perform TOML-first rewrite and ownership mapping.",
  "No manual edits to generated mirrors.",
]

[invariants]
ascii_only = true
warnings_as_errors = true
registry_first = true
claims_hypothesis_until_verified = true

[init]
steps = [
  "Capture branch/commit/scoped status.",
  "Load AGENTS.md + relevant registry TOML.",
  "Plan hypergranular steps + validation gates.",
  "Parallelize independent reads/search only.",
  "Implement smallest-surface deltas first.",
]

[exec]
rules = [
  "Prefer existing make/script targets.",
  "Keep shared Rust deps in Cargo.toml [workspace.dependencies].",
  "For CUDA-backed Rust crates, prefer cudarc runtime integration over build-time nvcc pipelines unless explicitly scoped otherwise.",
  "Keep domain kernels in crate-local modules and promote reusable parts into focused workspace crates.",
  "On dependency graph changes, run make dep-audit before pre-push.",
  "Update canonical TOML, then regenerate mirrors.",
  "Record provenance; label synthetic substitutions.",
  "Append citations; never delete.",
]

[gates]
required = [
  "cargo clippy --workspace -j$(nproc) -- -D warnings",
  "cargo test --workspace -j$(nproc)",
  "make ascii-check",
]
recommended = [
  "make rust-smoke",
  "make mcp-smoke",
  "make cargo-deny-check",
  "make registry",
  "make check",
]
dependency_change = ["make dep-audit", "make cargo-deny-check"]
pre_push = ["make pre-push-gate"]
pre_push_strict = ["make pre-push-gate-strict"]
governance = ["make governance-gate"]
legacy_alias = "wave6-gate -> governance-gate"
hook_setup = "make hooks-install"
hook_setup_strict = "make hooks-install-strict"
registry_verifiers = [
  "make registry-verify-markdown-inventory",
  "make registry-verify-markdown-owner",
  "make registry-verify-markdown-governance",
  "make registry-verify-schema-signatures",
  "make registry-verify-crossrefs",
  "make registry-verify-mirrors",
]

[claims]
rules = [
  "Map claim to first-party evidence.",
  "Attach reproducible artifact or verifier.",
  "Keep cross-registry refs consistent.",
]

[handoff]
sections = ["what_changed", "why_changed", "validation_run", "open_risks"]

[forbidden]
items = [
  "Warnings in final state.",
  "Manual mirror edits.",
  "New manual markdown outside allowlist.",
  "Promotion of unverifiable claims.",
]
```
