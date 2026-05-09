# Dependency upgrade history

This file records security-driven and policy-driven dependency upgrades. Each
entry should explain WHY (advisory or policy), WHAT changed (versions and
features), and HOW (verification steps).

## Schema

Each entry is a level-3 heading dated `YYYY-MM-DD` followed by:

- `Crate`: the package name as it appears in Cargo.lock.
- `From / To`: version strings before and after.
- `Advisory ID`: RUSTSEC, GHSA, or CVE identifier when applicable.
- `Trigger`: what flagged the upgrade (audit run, advisory, policy).
- `Exposure analysis`: what realistic attack surface this affected in our usage.
- `Verification`: commands run to confirm no regression.
- `Reviewer`: GitHub username of the person who approved the change.

Entries are most-recent-first.

---

## 2026-05-09 -- rustls-webpki 0.103.10 -> 0.103.13 (already in lockfile)

- Crate: `rustls-webpki`
- From: `0.103.10` (per data/output/audit/2026-04-30/04-gates/deny-check.txt)
- To:   `0.103.13` (per current Cargo.lock at commit 2f7b8ff4)
- Advisory ID: RUSTSEC-2026-0104
- Trigger: cargo deny check at the 2026-04-30 audit pass.
- Exposure analysis:
  - Affected via the chain `reqwest -> hyper-rustls -> rustls -> rustls-webpki`.
  - Local consumers: `data_core`, `chromiumoxide`, `lit_search`.
  - The vulnerability is a panic on malformed Certificate Revocation List
    parsing. Our HTTPS use is for SDSS, NASA, DESI, and a handful of
    arXiv mirrors -- these endpoints terminate TLS via OCSP stapling rather
    than CRL distribution. CRL exposure is therefore minimal in practice.
- Verification:
  - `cargo update -p rustls-webpki` reported "Locking 0 packages" on
    2026-05-09; the lockfile already pinned 0.103.13.
  - `grep -A2 'name = "rustls-webpki"' Cargo.lock` confirms version 0.103.13.
  - `cargo deny check advisories` reported "advisories ok" on 2026-05-09.
- Reviewer: @eirikr (auto-applied via lockfile drift between audit and review).

## 2026-05-09 -- image 0.25 default-features disabled (avif chain dropped)

- Crate: `image`
- From: `image = "0.25"` (default features include `avif`)
- To:   `image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }`
- Advisory ID: RUSTSEC-2026-0105 (transitive via core2 yanked)
- Trigger: cargo deny check yanked-version warning, audited 2026-05-09.
- Exposure analysis:
  - The yanked dep is `core2 v0.4.0`, reached through
    `core2 -> bitstream-io -> rav1e -> ravif -> image`.
  - `ravif` only ships when the `image/avif` feature is enabled (default).
  - Source-tree audit (`grep -rn 'Avif|avif|ImageFormat::Avif|save_avif|load_avif' crates/`)
    found zero genuine AVIF usage. The single false-positive match was a
    citation in a generated registry mirror (Taghavifar 2013).
  - Consumers (`docpipe`, `gororoba_cli_quantum`, `lbm_vulkan`) use only
    PNG and JPEG. `docpipe::ImageFormat::Jpeg` and `Raw` enum variants are
    docpipe-internal classifications, not invocations of `image::Image`.
- Verification:
  - `cargo build --workspace` (run with `CARGO_TARGET_DIR=.cache/gate-target`).
  - `cargo deny check` afterwards.
  - Per-crate `cargo check` for `gororoba_cli_quantum`, `docpipe`,
    `lbm_vulkan`.
- Reviewer: @eirikr (Stage B B-G4).

## 2026-05-09 -- rand 0.8.5 -> 0.8.6, 0.9.2 -> 0.9.3, 0.10.0 -> 0.10.1

- Crate: `rand` (three versions in dep graph)
- From: `0.8.5`, `0.9.2`, `0.10.0`
- To:   `0.8.6`, `0.9.3`, `0.10.1`
- Advisory ID: RUSTSEC-2026-0097 (informational unsound)
- Trigger: `cargo audit --json` 2026-05-09.
- Exposure analysis:
  - Unsoundness manifests only when ALL of these hold:
    1. The `log` and `thread_rng` features are enabled.
    2. A custom `log::Logger` is registered.
    3. The custom logger calls `rand::rng()` (or `thread_rng()`) and any
       `TryRng`/`RngCore` method.
    4. `ThreadRng` reseeds while inside the logger callback (every 64 KiB).
    5. Either trace-level logging is on, or warn-level + `getrandom`
       cannot supply a fresh seed.
  - This pipeline does not register a custom `log::Logger` that calls into
    `rand`; the default `env_logger` and the logging in CLI binaries
    consult `rand` only outside the logger callback. Real exploitability
    is zero in our usage. The upgrade is still cheap and patched
    versions are SemVer-compatible.
- Verification:
  - `cargo update -p 'rand@0.10.0' --precise 0.10.1` (and equivalents
    for 0.9.2 and 0.8.5) updated each occurrence in Cargo.lock.
  - `cargo audit` warning count went from 6 to 5 (the rand entry is
    gone; remaining 5 are bincode 1.x/2.x, instant 0.1, json 0.12,
    paste 1.0, all `unmaintained` not `unsound`).
  - `cargo build --workspace` to be re-verified after the multi-version
    bump (rand has minor API surface changes between 0.9 -> 0.10).
- Reviewer: @eirikr (security-driven, beyond Stage B scope).

---

## How to add a new entry

1. Edit this file directly. It is NOT a SQLite-derived export; it is hand
   maintained in git.
2. Use the section template above. Keep entries terse but specific.
3. If the change requires a deny.toml ignore (e.g., advisory has no upgrade
   path), add a parallel entry to `deny.toml` referencing this file by section
   anchor.
4. Commit with message `fix(deps): <crate> <from> -> <to> (<advisory>)`.

## See also

- `data/output/audit/2026-04-30/04-gates/deny-check.txt` (Stage A audit).
- `~/.claude/plans/stage-b-debt-resolution.md` (B-G3, B-G4 task definitions).
- `Cargo.lock` is the source of truth for current versions; this file
  documents *upgrade events*, not *current state*.
