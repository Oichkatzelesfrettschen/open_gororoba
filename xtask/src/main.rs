use anyhow::{Context, Result, bail};
use provenance_store::ProvenanceStore;
use rusqlite::Connection;
use serde::Serialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use tempfile::tempdir;
use verified_core::topology::HardwareTopology;

#[derive(Debug, Serialize)]
struct SchemaSnapshot {
    generated_from: String,
    object_count: usize,
    objects: Vec<SchemaObject>,
}

#[derive(Debug, Serialize)]
struct SchemaObject {
    schema: String,
    name: String,
    object_type: String,
    column_count: i64,
    without_rowid: bool,
    strict: bool,
    columns: Vec<SchemaColumn>,
    foreign_keys: Vec<SchemaForeignKey>,
    indexes: Vec<SchemaIndex>,
}

#[derive(Debug, Serialize)]
struct SchemaColumn {
    cid: i64,
    name: String,
    declared_type: String,
    not_null: bool,
    default_value: Option<String>,
    primary_key_position: i64,
    hidden: i64,
}

#[derive(Debug, Serialize)]
struct SchemaForeignKey {
    id: i64,
    seq: i64,
    ref_table: String,
    from_column: String,
    to_column: String,
    on_update: String,
    on_delete: String,
    match_kind: String,
}

#[derive(Debug, Serialize)]
struct SchemaIndex {
    seq: i64,
    name: String,
    unique: bool,
    origin: String,
    partial: bool,
    columns: Vec<SchemaIndexColumn>,
}

#[derive(Debug, Serialize)]
struct SchemaIndexColumn {
    seqno: i64,
    cid: i64,
    name: Option<String>,
    descending: bool,
    collation: String,
    key: bool,
}

#[derive(Debug, Serialize)]
struct HostProfile {
    physical_core_ids: Vec<usize>,
    physical_core_count: usize,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    worker_budget: usize,
    cargo_jobs: usize,
    rayon_threads: usize,
    rust_test_threads: usize,
    nextest_test_threads: usize,
    pytest_workers: usize,
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        bail!("usage: cargo run -p xtask -- <db-docs|host-profile> [args]");
    };
    match command.as_str() {
        "db-docs" => run_db_docs(args.any(|arg| arg == "--check")),
        "host-profile" => {
            let mut format = "shell".to_string();
            let mut iter = args.peekable();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--format" => {
                        let Some(value) = iter.next() else {
                            bail!("host-profile --format requires a value");
                        };
                        format = value;
                    }
                    other => bail!("unknown host-profile argument: {other}"),
                }
            }
            run_host_profile(&format)
        }
        other => bail!("unknown xtask command: {other}"),
    }
}

fn run_host_profile(format: &str) -> Result<()> {
    let profile = detect_host_profile();
    match format {
        "shell" => {
            println!("HOST_PHYSICAL_CORES={}", profile.physical_core_count);
            println!(
                "HOST_PHYSICAL_CORE_IDS=\"{}\"",
                join_usize(&profile.physical_core_ids)
            );
            println!("HOST_L3_CACHE_BYTES={}", profile.l3_cache_bytes);
            println!(
                "HOST_L3_SAFE_WORKING_SET_BYTES={}",
                profile.l3_safe_working_set_bytes
            );
            println!("HOST_WORKER_BUDGET={}", profile.worker_budget);
            println!("HOST_CARGO_JOBS={}", profile.cargo_jobs);
            println!("HOST_RAYON_THREADS={}", profile.rayon_threads);
            println!("HOST_RUST_TEST_THREADS={}", profile.rust_test_threads);
            println!("HOST_NEXTEST_TEST_THREADS={}", profile.nextest_test_threads);
            println!("HOST_PYTEST_WORKERS={}", profile.pytest_workers);
        }
        "json" => {
            println!("{}", serde_json::to_string_pretty(&profile)?);
        }
        "budget" => {
            println!("{}", profile.worker_budget);
        }
        other => bail!("unsupported host-profile format: {other}"),
    }
    Ok(())
}

fn run_db_docs(check_only: bool) -> Result<()> {
    let repo_root = repo_root()?;
    let outputs = generate_schema_outputs(&repo_root)?;
    write_or_check(
        &repo_root.join("db/schema.sql"),
        &outputs.schema_sql,
        check_only,
    )?;
    write_or_check(
        &repo_root.join("docs/db/schema.json"),
        &outputs.schema_json,
        check_only,
    )?;
    write_or_check(
        &repo_root.join("docs/db/catalog.md"),
        &outputs.catalog_md,
        check_only,
    )?;
    if check_only {
        println!("db-docs OK: generated schema artifacts match committed files");
    } else {
        println!("db-docs OK: regenerated db/schema.sql docs/db/schema.json docs/db/catalog.md");
    }
    Ok(())
}

struct GeneratedSchemaOutputs {
    schema_sql: String,
    schema_json: String,
    catalog_md: String,
}

fn generate_schema_outputs(repo_root: &Path) -> Result<GeneratedSchemaOutputs> {
    let temp_dir = tempdir().context("create temporary directory for schema docs")?;
    let db_path = temp_dir.path().join("schema.sqlite3");
    let _store = ProvenanceStore::open(&db_path)?;
    drop(_store);
    let conn = Connection::open(&db_path).context("open temporary schema sqlite database")?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .context("enable foreign_keys for schema introspection")?;

    let schema_sql = render_schema_sql(&conn)?;
    let snapshot = introspect_schema(&conn)?;
    let schema_json = serde_json::to_string_pretty(&snapshot)?;
    let catalog_md = render_catalog_markdown(&snapshot);
    let _ = repo_root;
    Ok(GeneratedSchemaOutputs {
        schema_sql,
        schema_json,
        catalog_md,
    })
}

fn render_schema_sql(conn: &Connection) -> Result<String> {
    let mut stmt = conn.prepare(
        "SELECT sql
         FROM sqlite_schema
         WHERE sql IS NOT NULL
           AND name NOT LIKE 'sqlite_%'
           AND name NOT IN (
               SELECT name
               FROM pragma_table_list
               WHERE schema = 'main' AND type = 'shadow'
           )
         ORDER BY tbl_name, type DESC, name",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut sql_blocks = Vec::new();
    for row in rows {
        let mut sql = row?;
        if !sql.trim_end().ends_with(';') {
            sql.push(';');
        }
        sql_blocks.push(sql);
    }
    let mut out = String::new();
    out.push_str("-- GENERATED FILE. DO NOT EDIT.\n");
    out.push_str("-- Canonical source: db/migrations/*.sql\n");
    out.push_str("-- Regenerate with: cargo run -p xtask -- db-docs\n\n");
    out.push_str(&sql_blocks.join("\n\n"));
    out.push('\n');
    Ok(out)
}

fn introspect_schema(conn: &Connection) -> Result<SchemaSnapshot> {
    let mut stmt = conn.prepare(
        "SELECT schema, name, type, ncol, wr, strict
         FROM pragma_table_list
         WHERE schema = 'main'
           AND name NOT LIKE 'sqlite_%'
           AND type != 'shadow'
         ORDER BY type, name",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, i64>(5)?,
        ))
    })?;
    let mut objects = Vec::new();
    for row in rows {
        let (schema, name, object_type, ncol, wr, strict) = row?;
        objects.push(SchemaObject {
            schema,
            name: name.clone(),
            object_type,
            column_count: ncol,
            without_rowid: wr != 0,
            strict: strict != 0,
            columns: schema_columns(conn, &name)?,
            foreign_keys: schema_foreign_keys(conn, &name)?,
            indexes: schema_indexes(conn, &name)?,
        });
    }
    Ok(SchemaSnapshot {
        generated_from: "db/migrations/*.sql via SQLite pragma introspection".to_string(),
        object_count: objects.len(),
        objects,
    })
}

fn schema_columns(conn: &Connection, table_name: &str) -> Result<Vec<SchemaColumn>> {
    let sql = format!(
        "SELECT cid, name, type, \"notnull\", dflt_value, pk, hidden
         FROM pragma_table_xinfo('{}')
         ORDER BY cid",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaColumn {
            cid: row.get(0)?,
            name: row.get(1)?,
            declared_type: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            not_null: row.get::<_, i64>(3)? != 0,
            default_value: row.get(4)?,
            primary_key_position: row.get(5)?,
            hidden: row.get(6)?,
        })
    })?;
    let mut columns = Vec::new();
    for row in rows {
        columns.push(row?);
    }
    Ok(columns)
}

fn schema_foreign_keys(conn: &Connection, table_name: &str) -> Result<Vec<SchemaForeignKey>> {
    let sql = format!(
        "SELECT id, seq, \"table\", \"from\", \"to\", on_update, on_delete, match
         FROM pragma_foreign_key_list('{}')
         ORDER BY id, seq",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaForeignKey {
            id: row.get(0)?,
            seq: row.get(1)?,
            ref_table: row.get(2)?,
            from_column: row.get(3)?,
            to_column: row.get(4)?,
            on_update: row.get(5)?,
            on_delete: row.get(6)?,
            match_kind: row.get(7)?,
        })
    })?;
    let mut keys = Vec::new();
    for row in rows {
        keys.push(row?);
    }
    Ok(keys)
}

fn schema_indexes(conn: &Connection, table_name: &str) -> Result<Vec<SchemaIndex>> {
    let sql = format!(
        "SELECT seq, name, \"unique\", origin, partial
         FROM pragma_index_list('{}')
         ORDER BY seq",
        table_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })?;
    let mut indexes = Vec::new();
    for row in rows {
        let (seq, name, unique, origin, partial) = row?;
        indexes.push(SchemaIndex {
            seq,
            columns: schema_index_columns(conn, &name)?,
            name,
            unique: unique != 0,
            origin,
            partial: partial != 0,
        });
    }
    Ok(indexes)
}

fn schema_index_columns(conn: &Connection, index_name: &str) -> Result<Vec<SchemaIndexColumn>> {
    let sql = format!(
        "SELECT seqno, cid, name, desc, coll, key
         FROM pragma_index_xinfo('{}')
         ORDER BY seqno",
        index_name.replace('\'', "''")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(SchemaIndexColumn {
            seqno: row.get(0)?,
            cid: row.get(1)?,
            name: row.get(2)?,
            descending: row.get::<_, i64>(3)? != 0,
            collation: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            key: row.get::<_, i64>(5)? != 0,
        })
    })?;
    let mut columns = Vec::new();
    for row in rows {
        columns.push(row?);
    }
    Ok(columns)
}

fn render_catalog_markdown(snapshot: &SchemaSnapshot) -> String {
    let mut out = String::new();
    out.push_str("<!-- AUTO-GENERATED: DO NOT EDIT -->\n");
    out.push_str("<!-- Source of truth: db/schema.sql -->\n");
    out.push_str(
        "<!-- Generated from: db/migrations/*.sql via cargo run -p xtask -- db-docs -->\n\n",
    );
    out.push_str("# Database Catalog\n\n");
    out.push_str("Generated file. Do not edit.\n\n");
    out.push_str("- Source of truth: `db/schema.sql`\n");
    out.push_str("- Canonical migrations: `db/migrations/*.sql`\n");
    out.push_str("- Regenerate with: `cargo run -p xtask -- db-docs`\n");
    out.push_str(&format!("- Objects: `{}`\n\n", snapshot.object_count));

    for object in &snapshot.objects {
        out.push_str(&format!(
            "## `{}` ({})\n\n",
            object.name, object.object_type
        ));
        out.push_str(&format!(
            "- Strict: `{}`\n- Without rowid: `{}`\n- Declared columns: `{}`\n\n",
            object.strict, object.without_rowid, object.column_count
        ));
        out.push_str("| cid | name | type | not null | default | pk | hidden |\n");
        out.push_str("| --- | --- | --- | --- | --- | --- | --- |\n");
        for column in &object.columns {
            let default_value = column.default_value.as_deref().unwrap_or("");
            out.push_str(&format!(
                "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                column.cid,
                column.name,
                column.declared_type,
                column.not_null,
                default_value.replace('|', "\\|"),
                column.primary_key_position,
                column.hidden
            ));
        }
        out.push('\n');
        if !object.foreign_keys.is_empty() {
            out.push_str("Foreign keys:\n\n");
            out.push_str("| id | seq | table | from | to | on update | on delete | match |\n");
            out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
            for fk in &object.foreign_keys {
                out.push_str(&format!(
                    "| {} | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                    fk.id,
                    fk.seq,
                    fk.ref_table,
                    fk.from_column,
                    fk.to_column,
                    fk.on_update,
                    fk.on_delete,
                    fk.match_kind
                ));
            }
            out.push('\n');
        }
        if !object.indexes.is_empty() {
            out.push_str("Indexes:\n\n");
            out.push_str("| seq | name | unique | origin | partial | columns |\n");
            out.push_str("| --- | --- | --- | --- | --- | --- |\n");
            for index in &object.indexes {
                let cols = index
                    .columns
                    .iter()
                    .map(|col| {
                        let label = col.name.clone().unwrap_or_else(|| "<expr>".to_string());
                        if col.descending {
                            format!("{label} desc")
                        } else {
                            label
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!(
                    "| {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
                    index.seq, index.name, index.unique, index.origin, index.partial, cols
                ));
            }
            out.push('\n');
        }
    }

    out
}

fn detect_host_profile() -> HostProfile {
    let topo = HardwareTopology::current();
    let physical_core_count = topo.physical_core_ids.len().max(1);
    HostProfile {
        physical_core_ids: topo.physical_core_ids.clone(),
        physical_core_count,
        l3_cache_bytes: topo.l3_cache_bytes,
        l3_safe_working_set_bytes: topo.l3_safe_working_set_bytes,
        worker_budget: physical_core_count,
        cargo_jobs: physical_core_count,
        rayon_threads: physical_core_count,
        rust_test_threads: physical_core_count,
        nextest_test_threads: physical_core_count,
        pytest_workers: physical_core_count,
    }
}

fn join_usize(items: &[usize]) -> String {
    items
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn write_or_check(path: &Path, content: &str, check_only: bool) -> Result<()> {
    if check_only {
        let existing = fs::read_to_string(path)
            .with_context(|| format!("read existing generated file {}", path.display()))?;
        if existing != content {
            bail!(
                "generated schema artifact drift detected for {}; run cargo run -p xtask -- db-docs",
                path.display()
            );
        }
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("write generated file {}", path.display()))?;
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("resolve repository root from xtask manifest directory")
}
