# SQLite canonical-write-plane design for the registry layer

This document captures the proposed design for extending `crates/gororoba_db/`
with `Claim`, `Insight`, and `Experiment` mutation subcommands. The goal is
to retire hand-editing of `registry/*.toml` files (which became read-only
auto-generated exports on 2026-03-23) and route all writes through the
canonical SQLite database at `registry/canonical/control_plane.sqlite3`.

## Why this file exists

Stage B plan tasks B-R1 (C-441 status_note), B-R2 (C-932 status_note),
B-R5 (E-197 status_note), and any future per-row registry edit cannot be
executed via direct TOML edit because the `registry/*.toml` files are
generated outputs. The current `gororoba-db` CLI exposes `Planning` and
`Requirements` mutators but lacks `Claim`, `Insight`, and `Experiment`
mutators. This document lays out the extension that closes the gap.

## Stack confirmation (May 2026)

A May 2026 research pass confirmed the workspace already pins the right
crates for this problem class. No version bumps are needed before
implementation:

| Crate                | Version    | Role                                       |
|----------------------|------------|--------------------------------------------|
| rusqlite             | 0.39       | Synchronous, single-process SQLite driver  |
| rusqlite_migration   | 2.5        | Versioned migrations (uses user_version)   |
| toml                 | 1.1        | Read-only ingest of TOML payloads          |
| toml_edit            | 0.25       | Round-trip preservation of comments/order  |
| clap                 | 4.5 derive | CLI argument parsing                       |
| jsonschema           | 0.30       | Pre-INSERT row validation                  |
| schemars             | latest     | Derive JSON Schema from Rust structs       |
| blake3               | 1.8        | Internal content-addressing                |
| sha2                 | 0.11       | Existing content_sha256 fields             |

Rationale for keeping rusqlite over sqlx/sea-orm/diesel:

- This is a synchronous, write-heavy CLI tool against one file-on-disk.
  sqlx/sea-orm pull async runtimes we do not need; diesel adds a DSL and
  migration tooling that fights FTS5 virtual tables.
- rusqlite 0.39 (released 2025) tightened the API: u64/usize ToSql/FromSql
  are off by default, and the statement cache is now opt-in. We should
  build a long-lived `CachedStatement` per hot query.

## CLI surface (proposed)

The existing CLI has these mutator subcommands:

```
gororoba-db planning <action> ...
gororoba-db requirements <action> ...
```

Add three more:

```
gororoba-db claim       <action> ...
gororoba-db insight     <action> ...
gororoba-db experiment  <action> ...
```

Each new mutator's `<action>` enum supports:

- `update --id <ID> [--status-note <NOTE>] [--field key=value ...]`
- `show --id <ID>`
- `touch --id <ID>` (no-op timestamp bump for re-export)
- `history --id <ID>` (chronological deltas from claim_revisions table)

A `MutationCommon` flattened struct carries fields shared across actions:

```rust
#[derive(Args, Debug)]
struct MutationCommon {
    /// Entity identifier (e.g., C-441, I-001, E-201).
    #[arg(long)]
    id: String,
    /// New status_note value. Plain text; ASCII only.
    #[arg(long)]
    status_note: Option<String>,
    /// Reviewer GitHub username; defaults to $USER.
    #[arg(long)]
    actor: Option<String>,
    /// Free-form reason recorded in claim_revisions.
    #[arg(long)]
    reason: Option<String>,
    /// Print plan but do not commit.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    /// Re-export TOML compatibility lanes after mutation.
    #[arg(long, default_value_t = true)]
    regen_toml: bool,
}
```

## Provenance: claim_revisions audit table

Every mutation must append a row to a new audit table to preserve
provenance:

```sql
CREATE TABLE claim_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id TEXT NOT NULL,
    field_name TEXT NOT NULL,           -- 'status_note', 'last_verified', etc.
    prev_value_sha256 TEXT,             -- BLAKE3 of previous value, NULL on first
    new_value_sha256 TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT,
    ts_utc TEXT NOT NULL,               -- ISO-8601 UTC
    operation TEXT NOT NULL CHECK (operation IN ('update','touch','create','delete')),
    application_id INTEGER,             -- pragma application_id sentinel for CLI vs raw SQL
    FOREIGN KEY (claim_id) REFERENCES claims(id)
);

CREATE INDEX claim_revisions_by_claim ON claim_revisions(claim_id, ts_utc);
```

Mirror tables for `insight_revisions` and `experiment_revisions`.

A `BEFORE UPDATE` trigger on the `claims` table can enforce that updates
only flow through the CLI: check `pragma application_id() = <SENTINEL>`
where `<SENTINEL>` is set by the CLI binary on connection open. Bare
`UPDATE claims SET ...` from external tooling would be rejected.

## Transaction strategy

Wrap every mutation in `BEGIN IMMEDIATE; ... COMMIT;`:

```rust
let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
// 1. Read current row, hash via BLAKE3.
// 2. Validate new row via jsonschema.
// 3. UPDATE claims SET ...
// 4. INSERT INTO claim_revisions (...) VALUES (...).
// 5. tx.commit() -- or rollback on any error.
```

`Immediate` rather than `Deferred` ensures concurrent writers fail fast
rather than block on the upgrade-to-write transition mid-transaction.

## TOML round-trip pitfalls (preserve byte-equivalence)

The compatibility export TOMLs are content-hashed in
`registry/schema_signatures.toml`. Mutations must produce byte-identical
output across re-runs to keep the governance gate green. Specific risks:

1. **Comment loss**: never use `toml::from_str -> mutate -> toml::to_string`
   for a file that may carry comments. Use `toml_edit::DocumentMut` end-to-end.
2. **Key reordering**: emit rows by `ORDER BY id` from SQLite, regardless
   of insertion order. `BTreeMap` ordering in `toml` differs from source
   order; `toml_edit` preserves source order but on a fresh emit must be
   driven by deterministic SELECT.
3. **Array-of-tables blank-line drift**: after each `[[claim]]` block,
   `toml_edit` may drop the surrounding blank line on partial mutation.
   Walk the `DocumentMut`, find each `ArrayOfTables`, set
   `aot.set_trailing("\n")` and per-table `set_prefix("\n")`.
4. **InlineTable vs standard table**: pick one per column type and enforce
   it post-emit. Recommend standard tables except for two-field date pairs.
5. **Float trailing zero**: `1.0` and `1.00` both round-trip to `f64` but
   serialize differently. Format all floats through a per-column precision
   stored in the JSON Schema; never use `Display` directly.
6. **Trailing newline**: emit exactly one `\n` at EOF. Add a unit test
   that asserts `s.ends_with('\n') && !s.ends_with("\n\n")`.

## FTS5 row triggers (replace explicit re-index)

Add a new migration `Mxxx_fts5_triggers.sql` containing the standard
external-content FTS5 trigger pattern:

```sql
-- After INSERT
CREATE TRIGGER claims_ai AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, id, title, status_note)
    VALUES (new.rowid, new.id, new.title, new.status_note);
END;

-- After DELETE
CREATE TRIGGER claims_ad AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, id, title, status_note)
    VALUES ('delete', old.rowid, old.id, old.title, old.status_note);
END;

-- After UPDATE
CREATE TRIGGER claims_au AFTER UPDATE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, id, title, status_note)
    VALUES ('delete', old.rowid, old.id, old.title, old.status_note);
    INSERT INTO claims_fts(rowid, id, title, status_note)
    VALUES (new.rowid, new.id, new.title, new.status_note);
END;
```

Mirror for `insights` and `experiments`. SQLite >= 3.43 supports
`contentless-delete=1` if column reads from the FTS table are not needed.

## Validation: jsonschema integration

Define schemas under `registry/schemas/`:

- `claim.schema.json`
- `insight.schema.json`
- `experiment.schema.json`

Derive them from Rust structs via `schemars::JsonSchema`. Validate every
candidate row before the SQL `UPDATE`:

```rust
let validator = jsonschema::Validator::options()
    .with_draft(Draft::Draft202012)
    .build(&schema_value)?;
match validator.validate(&row_value) {
    Ok(_) => proceed_with_update(),
    Err(errors) => return Err(Error::SchemaValidation(errors.collect())),
}
```

Cache compiled validators behind `OnceLock<jsonschema::Validator>` per
entity type.

## Implementation order (smallest first)

1. **Migration**: add the three `*_revisions` tables with their indexes.
2. **MutationCommon struct**: add to gororoba_db CLI; reuse for all three
   mutators.
3. **Claim::Update** (smallest scope): single field updates only
   (status_note); UPDATE via BEGIN IMMEDIATE; insert into claim_revisions.
4. **Schema validation**: wire jsonschema for claim rows; defer
   insight/experiment validation to step 6.
5. **TOML re-export**: add `--regen-toml` (default true); call existing
   `provenance export-control-plane` after commit.
6. **Insight::Update + Experiment::Update**: mirror Claim's flow.
7. **Show / Touch / History**: read-only commands; smaller surface.
8. **FTS5 row triggers**: migration only; replaces explicit re-index step.
9. **BEFORE UPDATE trigger**: optional hardening; gate behind a feature
   flag for one sprint to verify no test-fixture writes break.
10. **`claim history` subcommand**: human-readable chronological deltas.

## Acceptance criteria

The extension is complete when:

- `gororoba-db claim update --id C-441 --status-note "DESI DR2 confirms..."`
  succeeds and produces:
  - A row in `claim_revisions` with the correct prev/new BLAKE3 hashes.
  - An updated `claims` row in SQLite.
  - Regenerated `registry/claims.toml` with byte-identical formatting
    except for the changed status_note.
  - A regenerated `registry/schema_signatures.toml` matching the new
    content_sha256.
  - `make integrity-resolution` exits 0.
  - `make governance-gate` exits 0.

- The same flow works for `insight update` and `experiment update`.
- Invalid row values are rejected with clap-friendly errors.
- The full pre-push hook chain passes.

## See also

- Stage B plan tasks B-R1, B-R2, B-R5 in
  `~/.claude/plans/stage-b-debt-resolution.md`.
- Three-layer registry architecture commit `6b84c084` (2026-03-19).
- AUTO-GENERATED markers added in commit `46ed069b` (2026-03-23).
- Existing mutator template: `cmd_planning_mutation` and
  `cmd_requirements_mutation` in
  `crates/gororoba_db/src/bin/gororoba_db.rs`.

## Primary sources

- rusqlite 0.39 release notes: https://github.com/rusqlite/rusqlite/releases
- rusqlite_migration changelog: https://cj.rs/rusqlite_migration_docs/changelog/
- toml v0.9 announcement (epage 2025-07): https://epage.github.io/blog/2025/07/toml-09/
- toml vs toml_edit (epage 2023-01, still authoritative):
  https://epage.github.io/blog/2023/01/toml-vs-toml-edit/
- SQLite FTS5 docs: https://sqlite.org/fts5.html
- Simon H, "SQLite FTS5 Triggers" (2021): https://simonh.uk/2021/05/11/sqlite-fts5-triggers/
- jsonschema (Stranger6667): https://github.com/Stranger6667/jsonschema
- schemars: https://github.com/GREsau/schemars
- BLAKE3: https://github.com/BLAKE3-team/BLAKE3
- clap derive subcommands tutorial:
  https://docs.rs/clap/latest/clap/_derive/_tutorial/index.html
