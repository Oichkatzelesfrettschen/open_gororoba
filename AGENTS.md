---
description: open_gororoba -- pure-Rust scientific computing workspace (Cayley-Dickson algebras, cosmology, LBM, Rocq proofs, SQLite-canonical registry)
last_verified: 2026-05-17
---

# open_gororoba -- agent and developer reference

This file is the canonical operating guide for any agent (Claude,
Codex, Gemini, Mistral, Ollama, DeepSeek, human) working in this
repository. `CLAUDE.md` (also at the repo root) is a Claude-specific
overlay that points back at this file for the shared policies and
states only the Claude-tool deltas on top. There is no `GEMINI.md`
in the tree today; if one is added later it SHOULD follow the same
overlay pattern (or be a literal symlink to this file). If `CLAUDE.md`
and `AGENTS.md` ever disagree on a shared policy, this file is
authoritative.

Peer references this document inherits style from:

- `~/workspaces/mesa/steinmarder/AGENTS.md` -- C11 + CUDA + SASS-RE
  engine; source of the comment-hygiene policy below.
- `~/workspaces/mesa/mesa-26-gororoba/AGENTS.md` -- Mesa 26.1-devel
  fork; source of the `Assisted-by:` commit-trailer policy.

Where the peer projects deal with hardware reverse engineering, this
project deals with mathematical structures (Cayley-Dickson algebras),
fluid dynamics (Lattice Boltzmann), formal verification (Rocq 9.1.1),
and a SQLite-canonical claim/insight/experiment registry. The
hygiene + voice + AI-disclosure rules transfer unchanged; the
hardware-specific tables are replaced with the scientific stack.

## Read first (in this order)

1. This file (`AGENTS.md`).
2. `~/.claude/CLAUDE.md` -- global user policies (ASCII,
   warnings-as-errors, no shortcuts, TodoWrite discipline,
   AskUserQuestion exhaustively).
3. `~/.claude/projects/-home-eirikr-Github-open-gororoba/memory/MEMORY.md`
   -- per-project memory index.
4. `~/AGENTS.md` -- home-level cross-project policies.

## High-value entry points

| Path | Why it matters |
|------|----------------|
| `Cargo.toml` (root) | Workspace members + `[workspace.lints]` (warnings-as-errors source of truth) |
| `rust-toolchain.toml` | Nightly pin; do not bump without coordinating the gate. |
| `.githooks/pre-push` | Six-gate chain (lfs, cache, ansi, terminology, rust-regression, governance). |
| `Makefile` | Top-level lanes (`make rust-clippy`, `make integrity`, `make cpd-audit`). |
| `registry/canonical/control_plane.sqlite3` | Canonical write target for the claim/insight/experiment registry. |
| `registry/*.toml` | AUTO-GENERATED read-only compat exports. Do NOT hand-edit. |
| `crates/gororoba_gpu_bridge/` | Canonical type vocabulary (`ComputeBackend`, `HardwareCaps`, `StoragePrecision`). |
| `crates/gororoba_gpu_vulkan/` | Shared Vulkan helpers (Instance, Adapter, Device, ShaderModule, DispatchScope). |
| `crates/gororoba_gpu_cubecl/` | Shared cubecl-wgpu probe + test-support macros. |
| `crates/gororoba_gpu_cuda/` | Shared CUDA helpers (Context, DeviceProbe, CompileOptions, ModuleRegistry, LaunchConfig, Telemetry, OptiX). |
| `crates/cd_kernel/` | Cayley-Dickson tower (8D..1024D), TurboQuant, multi-backend kernels. |
| `crates/lbm_3d/` | CPU D3Q19 reference solver (BGK + MRT). |
| `crates/lbm_vulkan/` + `crates/lbm_3d_cuda/` | GPU LBM backends. |
| `proofs/` | Rocq 9.1.1 formal proofs; `make -C proofs vos` for interface check, `make -C proofs vok` for body check. |
| `docs/engineering/registry_canonical_architecture.md` | Four-layer registry flow: SQLite -> compat TOMLs -> Rust mirrors -> docs. |

## Top-line operating rules

- **ASCII only**. No emojis. No smart quotes. No en/em dashes. No
  box-drawing characters. The `ansi-check` and `terminology-gate`
  pre-push hooks reject violations.
- **Warnings-as-errors** via `[workspace.lints]` in root `Cargo.toml`.
  Do NOT bypass with crate-local `#![allow(warnings)]`. Narrow-scope
  `#[allow(clippy::<lint>)]` with a documented rationale is permitted.
- **SQLite-canonical registry**. The 36 TOML files under `registry/`
  are AUTO-GENERATED. The canonical write path is
  `registry/canonical/control_plane.sqlite3`. See the
  "Registry: SQLite-canonical" section below for the exact mutation
  workflow.
- **Pure Rust**. No `.sh` scripts. No `.py` analysis scripts. Use
  PyO3 if a Python library must be wrapped; call it from a typed
  Rust binary.
- **No symlinks** as workarounds. Use a separate `CARGO_TARGET_DIR`
  per worktree.
- **Pre-push hook** at `.githooks/pre-push` (active via
  `core.hooksPath`) runs the 6-gate chain. Do not skip with
  `--no-verify` unless explicitly directed and rationale documented
  in the commit message.

## Build environment

- Toolchain: `rust-toolchain.toml` pins nightly-2026-03-05,
  edition 2024.
- Default target dir for gates: `CARGO_TARGET_DIR=.cache/gate-target`.
- Per-worktree experimental dirs: `.cache/exp-<name>-target/`.
- Cache budget: gate-target <= 200G; full `.cache` <= 250G; sweep
  with `make cache-sweep` (cargo-sweep --maxsize 100GB) or
  `make cache-sweep-soft` for the in-flight 150G soft cap.

### Canonical build

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo build --workspace --profile release-gate
```

Per-crate, with gate semantics:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p <crate> --all-targets --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo nextest run -p <crate> --lib --cargo-profile release-gate
```

### Pre-push 6-gate chain

| # | Gate | Purpose |
|---|---|---|
| 1 | git-lfs handoff | Pre-push hook chains to lfs |
| 2 | cache-check | Soft cap (150G) + hard cap (200G) on `.cache/` |
| 3 | terminology-gate | 8 banned legacy terms; prefer `sign_imbalance` for the renamed crate vocabulary. |
| 4 | ansi-check | Reject non-ASCII bytes |
| 5 | rust-regression-scoped | Scoped clippy + nextest on changed-crate closure |
| 6 | governance-gate-readonly | Verify registry `schema_signatures.toml` matches checked-in TOMLs |

Verify hook state: `git config --get core.hooksPath` should print
`.githooks`. The file at `.git/hooks/pre-push` is an unused git-lfs
stub kept for transparency.

## Registry: SQLite-canonical (since 2026-03-23)

- Canonical write target: `registry/canonical/control_plane.sqlite3`.
- `registry/*.toml` are AUTO-GENERATED read-only compat exports.
  Every TOML file in `registry/` starts with the header
  `# AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT.`
- Source manifest: `registry/source_manifest.toml` declares the 36
  TOMLs that participate in compatibility round-trip verification.
- Architecture walkthrough:
  `docs/engineering/registry_canonical_architecture.md`.
- Audit metric taxonomy:
  `docs/engineering/repo_audit_metric_taxonomy.md`.

### Mutation workflow (MANDATORY)

1. Edit via `gororoba-db` CLI (in `crates/gororoba_db/`) against the
   SQLite.
2. Re-export compatibility lanes:
   `cargo run -p gororoba_cli_data --bin provenance -- export-control-plane`.
3. Refresh integrity hashes:
   `make integrity-resolution`.
4. Commit the SQLite delta + regenerated TOMLs + regenerated
   markdown together in a single atomic commit.

Never hand-edit a file whose first line is the AUTO-GENERATED
header. Doing so will desync the canonical store and produce
content_sha mismatches at the next governance-gate run.

## GPU backend foundation (Wave B)

The workspace consolidates Vulkan / cubecl-wgpu / CUDA call sites
through three helper crates:

| Helper crate | What it owns | Replaces |
|---|---|---|
| `gororoba_gpu_vulkan` (`ash` feature) | `InstanceBuilder`, `Adapter::pick`, `DeviceBuilder`, `ShaderModule::from_wgsl`, `DescriptorSetLayoutSpec`, `ComputePipelineBuilder`, `DispatchScope`. | 10+ hand-rolled Entry::load + create_instance + queue-family scan sites. |
| `gororoba_gpu_cubecl` (`cubecl` feature) | `Runtime::probe` (panic-safe wgpu adapter probe); `test_support::skip_if_unavailable!` macro. | 4 hand-rolled 11-line `catch_unwind` probes. |
| `gororoba_gpu_cuda` (`cudarc` feature) | `Context::with_default_device`, `DeviceProbe::query`, `CompileOptions::for_arch`, `ModuleRegistry::load`, `Buffer<T>`, `ManagedBuffer<T>`, `LaunchConfig::launch_1d/2d/3d`, `telemetry::Telemetry`, `optix::PipelineBuilder`. | 48+ ad-hoc `CudaContext::new(0)` + 35+ NVRTC compile-options + 28+ PTX-load sites. |

All three crates default to no-op (empty `default` features).
Enabling the SDK feature pulls in the relevant deps. New code MUST
route through these helpers, not re-invent the boilerplate.

## Scientific stack pinning

- `statrs` 0.18.0 requires `nalgebra` 0.33; do not upgrade
  `nalgebra` without resolving `statrs` first.
- `gauss-quad` 0.2.4, `kodama` 0.3.0, `kiddo` 5.2.4, `petgraph` 0.7,
  `wide` 0.7.
- `cudarc` 0.19.1 (NVRTC runtime compilation), `burn` 0.16+.
- See the per-project MEMORY for pinning rationale and Rocq proof
  patterns (`ring_simplify+lra`, `cbv+ring_simplify`, tower rewrites,
  fuel recursion, Boolean reflection).

## Scientific debt classes (eleven)

Tracked in `plans/repo_debt_roadmap_2026_04_11.toml`:

1. `DEBT-NUMERICAL-ALGORITHM` (e.g., bounded_nelder_mead duplication)
2. `DEBT-STRUCTURAL-ARCHITECTURE` (e.g., data_core fetch/parse mixing)
3. `DEBT-DUPLICATION` (CPD clusters; PMD CPD lane in `make cpd-audit`)
4. `DEBT-TEST-VERIFICATION` (gate-audit narrowness; coverage)
5. `DEBT-BUILD-WORKSPACE` (Makefile sprawl; xtask migration target)
6. `DEBT-DATA-PROVENANCE` (default features and network isolation)
7. `DEBT-GENERATED-ARTIFACT` (registry_mirrors quarantine)
8. `DEBT-DOCUMENTATION-REQUIREMENTS` (audit lane coverage in REQUIREMENTS.md)
9. `DEBT-SCIENTIFIC-EVIDENCE` (claims linked to reproducible artifacts)
10. `DEBT-SUPPLY-CHAIN` (cargo-deny, geiger, machete)
11. `DEBT-FORMAL-VERIFICATION` (admits, axioms, parameters tracked in registry)

## Stage references and baselines

- Stage A audit pack: `data/output/audit/2026-04-30/` (30+ artifacts).
- Debt baseline TOML: `data/output/debt_baseline_2026_04_30.toml`.
- Baseline git tag: `debt-baseline-v0` on commit `970b4da3`.
- Active in-repo roadmap: `plans/repo_debt_roadmap_2026_04_11.toml`.

## Comment + documentation hygiene policy

Scope: all agents (Claude, Codex, Gemini, Mistral, Ollama, DeepSeek,
human). All checked-in artifacts: source code AND markdown docs.

Sourced from `steinmarder/AGENTS.md` "Comment + documentation
hygiene policy"; adapted for the Rust + Rocq + WGSL surface in this
repo.

### Priority order

1. SOURCE CODE comments (HIGHEST -- code outlives docs).
2. MARKDOWN finding-doc bodies (SECONDARY).
3. FILE-HEADER license blocks (THIRD).

### Source code MUST NOT contain

| Banned pattern | Example | Move to |
|---|---|---|
| Task references | `// task #143` | commit message |
| Issue references | `// see issue #157` | commit message |
| PR numbers | `// PR #25` | commit message or `Closes #25:` trailer |
| Companion-PR breadcrumbs | `// companion to PR #...` | commit message |
| Phase labels | `// Phase 4.4` | commit message |
| Step-of-phase labels | `// Step 1 of Phase 3` | commit message |
| Wave labels in code | `// Wave C2.1 migration` | commit message |
| Session dates | `// (2026-05-15)` | commit message |
| Deictic time | `// as of today`, `currently`, `previously` | rewrite absolute |
| Cross-phase breadcrumbs | `// will be exercised when Wave E lands` | commit message |
| Author tags | `// (eirikr)`, `// @claude` | delete entirely |
| Deictic refs | `// this CD algebra`, `// our LBM solver` | rewrite with the exact crate / module / type name |
| Internal-repo paths | `// per data/output/audit/2026-04-30/foo.csv` | rewrite by content, not by path |
| NEW personal-name copyright on new files | `// Copyright (c) 2026 <git config user.name>` | use the project-collective form below, or SPDX-only |

The `ansi-check` + `terminology-gate` hooks catch the encoding and
banned-term subset of this list automatically; the remaining items
are reviewer-enforced for now (lint TBD).

### Source code MUST contain (when domain-specific)

- Absolute identifiers (crate name + module path) for any
  cross-crate citation, so the reader can navigate without context:

| Subject | Canonical form |
|---|---|
| Cayley-Dickson dimension | "8D (octonions)", "16D (sedenions)", "32D (pathions)" |
| LBM lattice | "D3Q19 BGK", "D3Q27 MRT" |
| Backend | "Vulkan (ash)", "cubecl-wgpu", "CUDA (cudarc)" |
| Precision tier | "FP32", "FP16", "BF16", "FP8 e4m3", "INT8 SoA" |
| Claim / Insight / Experiment / Binary | `C-1234`, `I-0042`, `E-201`, `B-bin_name` (matches registry IDs) |
| Rocq theorem | `Brown1972ChapterIII::brown_3_1_trace`, `Moreno1997::theorem_1_16` |

- Bit-field / SoA layout commentary
  (`SoA: f[channel * n_cells + cell_idx]`).
- Empirical numerical behaviour (`f32 GPU vs f64 CPU drifts ~1e-4
  relative per D3Q19 step at low Mach`).
- Mathematical invariants and non-obvious workaround rationale.
- Reference citation by name + chapter/section, not by line number
  or internal-repo path:

| WRONG (internal) | RIGHT (public) |
|---|---|
| `// per data/sources/moreno_1997_zd_paper.txt:1572` | `// per Moreno (1997), "Zero divisors of the Cayley-Dickson algebras over the reals", J. Algebra 196, Theorem 1.16` |
| `// see audit_2026_04_30.csv row 12` | `// see DEBT-DUPLICATION class in plans/repo_debt_roadmap_2026_04_11.toml` |

### Markdown finding-doc rules

Markdown bodies MAY carry chronology (every finding-doc has dated
frontmatter -- `last_verified`, `evidence_class`, dated filename,
ordered predecessors). That is intentional.

PR# / task# references MUST triangulate with a durable identifier:

| WRONG (PR# alone) | RIGHT (durable primary + PR# cross-link) |
|---|---|
| `the fix landed via PR #38` | `landed in commit f79a37fa (cd_kernel/turboquant/cuda migration to gpu_cuda); PR #38 / branch session/wave-c2-cd-kernel-turboquant-cuda-2026-05-15 for cross-link` |
| `see issue #157` | `see CLAIMS.md row C-1234 (issue #157 if still open)` |

### File-header rules

Mesa-conforming reference: `mesa-26-gororoba/docs/submittingpatches.rst`
(mesa-26 branch). Mesa forbids `Co-authored-by:` trailers for AI
tools (reserved for human co-authors) and specifies the
`Assisted-by:` and `Generated-by:` trailers instead. This project
follows mesa's trailer policy AND keeps file headers minimal +
SPDX-focused.

### NEW source files in this workspace

When a license header is appropriate (match adjacent-file style;
many files in this workspace use SPDX-only with no Copyright line,
which is also fine):

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//
// <one-line file description>
```

- MUST NOT fabricate an individual personal-name Copyright line.
  LLM templates default to `Copyright (c) YYYY <git config user.name>`
  -- strip that.
- MUST NOT add `(LLM-assisted)` or any AI tag to the file header.
  AI disclosure goes in the commit trailer.
- SHOULD omit the Copyright line entirely (SPDX-only header) when
  adjacent maintained files do the same.
- License: most crates are `GPL-2.0-or-later`; a few (lbm_3d_cuda,
  quantum_core, stats_core) carry `MIT OR Apache-2.0`. Match the
  surrounding crate's `Cargo.toml` license field.

### EXISTING source files with upstream copyrights

Pre-existing headers on third-party files (cudarc, ash, cubecl
wrappers if any vendored) MUST be preserved verbatim. The license
itself requires preservation: "The above copyright notice and this
permission notice shall be included in all copies or substantial
portions of the Software."

When ADDING new content to an upstream file, the upstream copyright
stays; do NOT add a second project-collective Copyright line on top
of the upstream header.

### Commit messages and PR descriptions

Source of truth: `mesa-26-gororoba/docs/submittingpatches.rst`.

- MUST NOT use `Co-authored-by:` for AI tools -- mesa reserves this
  trailer for HUMAN co-authors.
- WHEN AI participated in the creative process, disclose with the
  mesa-canonical `Assisted-by:` trailer. This project's development
  is multi-LLM + human in the loop; the canonical concise form is:

  ```text
  Assisted-by: Claude (Opus 4.7 1M context), ChatGPT Codex (5.x), Gemini (Flash/Pro 3.x), Mistral, Ollama, DeepSeek
  ```

  List ACTUAL tools used per commit. One comma-separated
  `Assisted-by:` line is fine, or one line per tool -- both match
  mesa's example syntax `Assisted-by: TOOL (OPTIONAL: MODEL)`.
- WHEN AI generated almost the entire code change, use
  `Generated-by:` instead of `Assisted-by:`.
- Trivial / sub-copyrightable / mechanical changes (autocomplete,
  variable rename, format-fix) MAY omit disclosure per mesa policy,
  though SHOULD still note the tool to ease review.

### Past commits with `Co-Authored-By: Claude` trailer

Commits up through 2026-05-17 used the
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
trailer. This violates mesa policy (which forbids
`Co-authored-by:` for AI tools).

Resolution:

- Existing commits are HISTORICAL ARTIFACTS. We do not force-push
  to rewrite git history; the trailers remain in the log.
- NEW commits from the date this policy lands forward MUST use
  `Assisted-by:` instead.
- Crucially: a `git pre-commit` lint (TBD) will block new commits
  with the old trailer once the implementation lands.

### Short form

Invent no fake individual credit. Preserve all pre-existing credit
verbatim (license requires it). New files use SPDX-only headers
matching adjacent crate style. Commit trailers use mesa's
`Assisted-by:` (never `Co-authored-by:` for AI). List the actual
tools used.

### Commenting voice

Adopted from steinmarder/AGENTS.md "Commenting voice"; adapted to
Rust:

- A short paragraph opens with the load-bearing claim
  ("PUSH-scheme writes are race-free because dst = src + c_i is a
  bijection on cells for each fixed i.").
- Then the primary-source citation, by name -- the function
  (`LbmSolver3D::phase2_streaming`), the paper section, the type
  (`StoragePrecision::Fp32`) -- not an internal-repo file extract.
- Then the consequence in one or two lines, with a fragment of
  code inlined when that is clearer than prose.
- A test name or benchmark reference when the comment exists to
  explain a regression being fixed.
- Env knobs / feature flags described together at the bottom of the
  block when relevant.

The distillate -- subtle shifts on top of the existing voice:

- Default short. One-line trailing comments on the load-bearing line
  beat a function-header paragraph, unless the function as a whole
  encodes a non-obvious invariant.
- One thought per comment; stack them when steps are distinct,
  rather than fusing into a multi-clause sentence.
- Do not paraphrase the next line of code. The comment is about
  WHY the code is shaped that way -- the numerical constraint, the
  paper section, the measurement -- not what it does.
- When the code is mechanical, no comment. Comments earn their
  place by carrying information that does not survive in the code
  itself.
- Anchor citations by name (paper title + section, Rocq theorem
  identifier, registry C-/I-/E-/B- ID) rather than by line number
  or repo-internal path.
- Multi-paragraph blocks are reserved for the genuine
  numerical-quirk WHYs -- the kind of finding that took a long
  bench measurement or a Rocq proof to land.

Three prose-level leans -- subtle, but they are what makes a comment
feel like everything just makes sense:

- State mechanism in the active voice ("the GPU reads f_in and
  writes the periodic-neighbor cell"), not in the passive
  ("f_in is combined with the offset").
- Let sequence be the explanation. "X. Then Y. Then Z." -- one
  sentence per step, each doing one thing -- beats one sentence
  with three clauses.
- Chain each WHY by what the next step needs, so the comment moves
  forward instead of cataloguing in parallel.
  ("...so the descriptor write can resolve the relocation" tells
  the reader where the explanation is heading.)

Example shape:

> The PUSH-scheme kernel reads 19 own-cell f values. It computes
> rho, u, f_eq, applies BGK collision in registers, then writes
> each post-collision f_i to the periodic-neighbor destination cell.
> For any fixed direction i, src -> src + c_i is a bijection on the
> periodic lattice, so the 19 scattered writes per thread are
> race-free without atomics.

This composes with the comment-hygiene rules above (no task #, no
PR #, no Phase X.Y, no deictic refs). Together: short, active,
sequenced, primary-source-grounded, time-invariant.

### LLM-readable markdown style

For files an LLM agent might slice-load (AGENTS.md, CLAUDE.md,
finding-docs, memory entries):

- MUST use heading depth <= 3 levels.
- MUST use exactly one H1 per file (the document title).
- MUST include frontmatter on programmatically-loaded files.
- MUST use language tags on code fences (`bash`, `rust`, `toml`,
  `wgsl`).
- SHOULD prefer tables over bullet lists for 3+ comparable items.
- MUST NOT use emoji, ASCII boxes, banner dividers in rules text
  (banned by `ansi-check` anyway).
- MUST name the exact section or path instead of using relative
  position phrases; the file may be slice-loaded.
- MUST use MUST / MUST NOT / SHOULD imperative voice in rules.

## When ambiguity arises

- Use `AskUserQuestion` early and often.
- Use `TaskCreate`/`TaskUpdate` to plan and track work granularly.
- Prefer the Explore agent for broad codebase searches over running
  multiple greps yourself.
- Never invent file paths; verify with `find`/`grep` first.
- When in doubt about the registry, consult the architecture
  walkthrough at `docs/engineering/registry_canonical_architecture.md`
  before mutating.

## Workspace cross-links

This repo is the primary fluid-dynamics + algebra workspace. Peer
references for cross-cutting workspace docs:

- `~/workspaces/mesa/steinmarder/` -- C11 + CUDA + SASS-RE; source
  of this file's comment-hygiene policy.
- `~/workspaces/mesa/mesa-26-gororoba/` -- Mesa 26.1-devel fork;
  source of this file's `Assisted-by:` commit-trailer policy.
- `~/workspaces/mesa/steinmarder/docs/workspace/` -- cross-cutting
  ccache/distcc/sccache wiring documentation (referenced where
  relevant for cargo + rustc parallel-build hygiene).
