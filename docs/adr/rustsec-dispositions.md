# RUSTSEC Advisory Dispositions

## Context

`cargo deny check advisories` ran against the workspace on 2026-04-30
under Phase 1 (T-110..T-113) of the elucidate-and-build-out plan.
Snapshot before any change:

- 7 advisories ignored in `deny.toml`
- 2 NEW advisories surfaced that were not in the ignore list
  (RUSTSEC-2026-0104, RUSTSEC-2026-0105)
- 1 yanked-warning (fastrand)

This ADR records the per-advisory disposition decisions and the actions
taken on 2026-04-30. The plan stipulated that every ignored advisory
either gets resolved or has a written, exposure-bounded rationale.

## Decision

| Advisory          | Severity | Status   | Action                                               |
| ----------------- | -------- | -------- | ---------------------------------------------------- |
| RUSTSEC-2026-0098 | HIGH     | RESOLVED | cargo update -p rustls-webpki --precise 0.103.13     |
| RUSTSEC-2026-0099 | HIGH     | RESOLVED | (same upgrade)                                       |
| RUSTSEC-2026-0104 | HIGH     | RESOLVED | (same upgrade; this one was new since the plan)      |
| fastrand yank     | --       | RESOLVED | cargo update -p fastrand                             |
| RUSTSEC-2025-0141 | CRITICAL | IGNORED  | Dev-dep only; no runtime exposure                    |
| RUSTSEC-2022-0081 | MEDIUM   | IGNORED  | Build-dep only; no runtime artifact                  |
| RUSTSEC-2024-0436 | MEDIUM   | IGNORED  | Proc-macro; compile-time only                        |
| RUSTSEC-2024-0384 | MEDIUM   | IGNORED  | Runtime via minifb; tracked for upstream resolution  |
| RUSTSEC-2026-0097 | HIGH     | IGNORED  | We do not install custom rand loggers; exposure zero |
| RUSTSEC-2026-0105 | unmaint  | IGNORED  | core2 yanked, no upgrade path; via image -> rav1e    |

After: `advisories ok, bans ok, licenses ok, sources ok`.

## 2026-08-02 dependency refresh

The current cargo-deny 0.20.2 database adds advisories for dependencies that
were not present in the April baseline. The fixed releases are adopted in the
workspace lockfile:

| Crate | Old versions | Locked release | Resolution |
| ----- | ------------ | -------------- | ---------- |
| anyhow | 1.0.102 | 1.0.103 | `cargo update -p anyhow --precise 1.0.103` |
| crossbeam-epoch | 0.9.18 | 0.9.20 | `cargo update -p crossbeam-epoch --precise 0.9.20` |
| lopdf | 0.38, 0.39, 0.40 | 0.42.0 | upgrade `lopdf`, `pdf-extract`, and `unpdf` |
| memmap2 | 0.9.10 | 0.9.11 | workspace minimum and lockfile update |
| quick-xml | 0.39.2 direct use | 0.41.0 | workspace minimum and lockfile update |

RUSTSEC-2026-0097 and RUSTSEC-2026-0105 no longer match the current resolved
graph, so their stale deny.toml entries are removed. The historical entries
above remain as the record of their earlier dispositions.

The lockfile retains quick-xml 0.39.2 only through the build-time
`wayland-scanner` proc-macro. Wayland protocol XML is trusted build input, and
the application XML readers use quick-xml 0.41.0. RUSTSEC-2026-0194 and
RUSTSEC-2026-0195 therefore remain explicitly bounded exceptions until
wayland-scanner changes its dependency range.

The lockfile retains `proc-macro-error2` only through the `iai-callgrind`
development benchmark macros. RustSec reports no safe upgrade, so
RUSTSEC-2026-0173 remains an explicit compile-time exception.

The lockfile retains `ttf-parser` 0.20.0 through plotters and 0.25.1 through
lopdf. RustSec reports these releases as unmaintained with no safe upgrade.
The exceptions are limited to plotting font support and PDF font parsing;
they do not hide a known fixed vulnerability. RUSTSEC-2026-0192 remains an
explicit upstream-maintenance exception.

`spin 0.10.0` remains a yanked transitive dependency through Burn 0.20.1 and
CubeCL 0.9.0. The workspace does not request `spin` directly, and the current
Burn dependency range does not accept the latest `spin 0.12.x` release.
Replacing or locally patching that crate would change the numerical backend
closure without a demonstrated security fix. The warning remains visible in
the cargo-deny report and is tracked for the next compatible Burn or CubeCL
upgrade; no deny ignore hides it.

Removal triggers for the four advisory exceptions are an upstream
dependency-range change or a verified replacement that preserves the affected
feature. The `spin` warning is removed only after Burn or CubeCL adopts a
non-yanked compatible release. The `cargo deny check` command remains
warnings-visible and fails on any new advisory that is not listed here.

## Per-advisory analysis

### RUSTSEC-2026-0098 / 0099 / 0104 -- rustls-webpki (RESOLVED)

Three advisories on the same crate, all addressed by the same point release.

- **0098**: name-constraint matching bug.
- **0099**: name-constraint matching bug.
- **0104**: reachable panic in `BorrowedCertRevocationList::from_der` /
  `OwnedCertRevocationList::from_der` for a syntactically valid empty
  BIT STRING in `onlySomeReasons`. Reachable before signature verification.

Path: `chromiumoxide`/`reqwest` -> `hyper-rustls` -> `rustls 0.23.37` ->
`rustls-webpki 0.103.10`. The 0.103.13 release fixes all three.

Resolution:

```bash
cargo update -p rustls-webpki --precise 0.103.13
```

Locked package only updated; 90 unchanged dependencies behind latest.

### RUSTSEC-2025-0141 -- bincode 1.3.3 unmaintained (IGNORED, exposure zero)

Path:

```
bincode v1.3.3
└── iai-callgrind v0.16.1   [dev-dependencies]
    └── cd_kernel ... (workspace tree)
```

The `[dev-dependencies]` annotation is decisive: bincode 1.3.3 is only
linked when running `cargo bench` against `iai-callgrind`-enabled tests.
Production binaries do not load it.

Production uses **bincode 2.0.1** via `burn-core` -> `neural_homotopy`,
and 2.0.x is past the advisory's affected range.

Mitigation: zero. We do not benchmark against untrusted bincode input.

Removal trigger: when `iai-callgrind` upgrades past bincode 1.3.3.

### RUSTSEC-2022-0081 -- json 0.12.4 (IGNORED, build-time only)

Path:

```
json v0.12.4   [build-dependencies via periodic-table-on-an-enum]
└── periodic-table-on-an-enum v0.3.2
    └── materials_core
```

`json` is loaded only by the build script of `periodic-table-on-an-enum`.
No runtime artifact contains `json` code. Even a maximally adversarial
`json`-parsing bug cannot reach runtime.

Removal trigger: when `periodic-table-on-an-enum` upgrades to a non-`json`
build dep, or when we replace it with `chemical-elements`/`atom-data`.

### RUSTSEC-2024-0436 -- paste 1.0.15 unmaintained (IGNORED, compile-time only)

Path: `paste -> argmin -> algebra_experimental`.

`paste` is a `proc-macro` crate. Its output is expanded at compile time;
no `paste` code ships in any binary. The advisory exists because the
crate is unmaintained, not because of an exploitable bug.

Removal trigger: when `argmin` migrates to `pastey`/`paste-up` (recommended
by the `paste` README) or when `paste` resumes maintenance.

### RUSTSEC-2024-0384 -- instant 0.1.13 unmaintained (IGNORED, bounded exposure)

Path:

```
instant v0.1.13
└── minifb v0.28.0
    └── gororoba_cli_physics
        └── gororoba_cli
```

`instant` IS in the runtime (it provides cross-platform `Instant::now`),
contrary to the prior deny.toml comment that called it a "wasm targets"
dep. Live use is timing in the `minifb` framebuffer/UI display path. On
non-wasm Linux targets (the deployment platform), `instant` is a thin
shim over `std::time::Instant`; the unmaintained-status risk is small.

Removal trigger: `minifb` upgrades to non-`instant` time source, or we
replace `minifb` with another framebuffer crate (`pixels`, `softbuffer`).

### RUSTSEC-2026-0097 -- rand soundness (IGNORED, exposure zero)

The advisory concerns soundness when a custom `rand` logger is installed
via `getrandom::register_custom_getrandom`. We do not install any custom
rand logger anywhere in the workspace. `grep -rn 'register_custom_getrandom'
crates/` returns zero matches.

Path covers three rand versions in the dep graph:

```
rand 0.8.5  via cauchy -> lax -> ndarray-linalg -> qua_ten_net -> quantum_core
rand 0.9.2  via argmin -> algebra_experimental
rand 0.10.0 direct workspace use in algebra_analysis
```

Mitigation: assert at workspace-config time that no crate calls
`register_custom_getrandom`. T-118 supply-chain-gate should add this
grep as a steady-state check.

Removal trigger: rand publishes a soundness-fixed patch.

### RUSTSEC-2026-0105 -- core2 yanked (IGNORED, no upgrade path)

NEW advisory not in the original plan. Path:

```
core2 v0.4.0
└── bitstream-io v4.9.0
    └── rav1e v0.8.1
        └── ravif v0.13.0
            └── image v0.25.10
                ├── docpipe
                ├── gororoba_cli_quantum
                ├── lbm_vulkan
                └── pdfium-render
```

`core2` is yanked from crates.io and the maintainer has stopped working on
it. RustSec lists no safe upgrade. Suggested alternatives -- `embedded-io`
or `no-std-io2` -- would require `bitstream-io` (or rav1e/ravif/image
upstream) to migrate, which is beyond our control.

Mitigation: we do not deserialize untrusted AVIF/AV1 streams in any
production path. The `image` crate is loaded for static-asset rendering
in `docpipe`, `gororoba_cli_quantum`, and `lbm_vulkan` UI; inputs are
all locally-controlled.

Removal trigger: when `bitstream-io` migrates off `core2`, or when we
replace the AVIF/AV1 dep chain with another image format.

## Process

- Whenever `cargo deny check` adds a NEW error, document the disposition
  in this ADR before adding to `deny.toml ignore`.
- The `[advisories] ignore` list in `deny.toml` carries a one-line
  rationale per entry pointing back to this ADR.
- T-118 will add `make supply-chain-gate` chaining cargo-deny + machete
  - geiger-drift; it should also include a grep-check that
    `register_custom_getrandom` has no callers (so RUSTSEC-2026-0097
    exposure stays zero).
- T-119 tracks burn 0.21 evaluation as upstream blocker (when burn 0.21+
  releases with bincode 2 only, `iai-callgrind` is the last dep on
  bincode 1.x in this repo).

## Related

- Plan: `plans/elucidate-and-build-out-nested-hollerith.md` (Phase 1).
- deny.toml `[advisories].ignore` carries one-line per-entry pointers
  to the sections above.
- Phase 0 baseline at `data/output/debt_baseline_2026_04_30.toml`.
- Stage A audit captured the prior state at
  `data/output/audit/2026-04-30/04-gates/deny-check.txt`.
