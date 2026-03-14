use anyhow::{Context, Result};
use clap::Parser;
use rusqlite::Connection;
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "algebra-theory-crosswalk")]
#[command(about = "Executed algebra/theory crosswalk over current binaries, experiments, claims, and empirical outputs")]
struct Cli {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct CrosswalkRow {
    family: String,
    status: String,
    bibliography_refs: Vec<String>,
    binaries: Vec<String>,
    experiments: Vec<String>,
    claims: Vec<String>,
    empirical_outputs: Vec<String>,
    notes: String,
}

#[derive(Debug, Serialize)]
struct CrosswalkReport {
    generated_at_utc: String,
    db_path: String,
    rows: Vec<CrosswalkRow>,
}

struct CrosswalkSpec<'a> {
    family: &'a str,
    bibliography_refs: &'a [&'a str],
    binary_refs: &'a [&'a str],
    experiment_refs: &'a [&'a str],
    claim_refs: &'a [&'a str],
    empirical_outputs: Vec<String>,
    notes: &'a str,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let report_path = cli.report.unwrap_or_else(|| {
        PathBuf::from("reports").join(format!(
            "algebra_theory_crosswalk_{}.toml",
            chrono::Utc::now().date_naive()
        ))
    });
    let conn = Connection::open(&cli.db)
        .with_context(|| format!("open control-plane DB {}", cli.db.display()))?;
    let binaries = load_string_set(&conn, "SELECT name FROM binaries_cp")?;
    let experiments = load_string_set(&conn, "SELECT id FROM experiments_cp")?;
    let claims = load_string_set(&conn, "SELECT id FROM claims")?;
    let latest_matrix_report = latest_report_path("survey_catalog_matrix_dr3_")
        .unwrap_or_else(|| "reports/survey_catalog_matrix_dr3_2026-03-13.toml".to_string());
    let latest_desi_report = latest_report_path("desi_dr2_bao_reference_")
        .unwrap_or_else(|| "reports/desi_dr2_bao_reference_2026-03-13.toml".to_string());

    let rows = vec![
        build_row(CrosswalkSpec {
            family: "Cayley-Dickson / zero-divisors",
            bibliography_refs: &[
                "de Marrais (2000) box-kite structure of sedenion zero-divisors",
                "Reggiani (2024) geometry of sedenion zero divisors",
                "Wilmot (2025) structure of the Cayley-Dickson algebras",
            ],
            binary_refs: &[
                "amplitude-imbalance-survey",
                "cd-ultrametric-scaling",
                "ultrametric-core-extract",
            ],
            experiment_refs: &["E-002"],
            claim_refs: &["C-449"],
            empirical_outputs: vec![
                "reports/ultrametric_core_report_2026-03-13.toml".to_string(),
                latest_matrix_report.clone(),
            ],
            notes: "Crosswalks the repo's native Cayley-Dickson analysis lane to the refreshed ultrametric and survey outputs rather than pretending the algebra itself is a sky catalog.",
        }, &binaries, &experiments, &claims),
        build_row(CrosswalkSpec {
            family: "Jordan / exceptional Jordan",
            bibliography_refs: &[
                "Exceptional Jordan / Albert-algebra references already surfaced in docs/INSIGHTS.md",
                "Singh delta^2 discussion tracked in repo insight lane",
            ],
            binary_refs: &["lie-jordan-halo-analysis"],
            experiment_refs: &["E-198"],
            claim_refs: &[],
            empirical_outputs: vec![
                "reports/lotss_manga_crossmatch_dr3_2026-03-13.toml",
                "reports/things_lotss_xmatch_dr3_2026-03-13.toml",
            ]
            .into_iter()
            .map(ToString::to_string)
            .collect(),
            notes: "Links the Jordan-family analysis to the empirical radio-stratification outputs where the repo currently reuses that language.",
        }, &binaries, &experiments, &claims),
        build_row(CrosswalkSpec {
            family: "Exceptional Lie / magic-square-related",
            bibliography_refs: &[
                "Damour, Henneaux, Nicolai (2002) E10 and M-theory",
                "repo E7/E8 and magic-square references in docs and artifact narratives",
            ],
            binary_refs: &["lie-jordan-halo-analysis", "algebraic-warp-sweep"],
            experiment_refs: &["E-193", "E-198"],
            claim_refs: &[],
            empirical_outputs: vec![
                latest_matrix_report.clone(),
                "reports/things_metadata_2026-03-13.toml".to_string(),
            ],
            notes: "Treats exceptional-Lie content as a theory overlay linked to current survey artifacts, not as a fake direct observation channel.",
        }, &binaries, &experiments, &claims),
        build_row(CrosswalkSpec {
            family: "Kac-Moody / moonshine",
            bibliography_refs: &[
                "Kac (1990) Infinite-Dimensional Lie Algebras",
                "repo moonshine / exceptional-structure literature pointers",
            ],
            binary_refs: &[],
            experiment_refs: &[],
            claim_refs: &[],
            empirical_outputs: vec![report_path.display().to_string()],
            notes: "This family is currently bibliography-led in the repo. The crosswalk records that state explicitly so it is not confused with an executed survey overlap.",
        }, &binaries, &experiments, &claims),
        build_row(CrosswalkSpec {
            family: "Planck / action / operator / effective-law scaffold",
            bibliography_refs: &[
                "CODATA 2022 / NIST Planck-scale constants",
                "Connes spectral-triple operator framework",
                "variable-order effective-law references from the project theory lane",
            ],
            binary_refs: &[],
            experiment_refs: &[],
            claim_refs: &[],
            empirical_outputs: vec![
                "reports/heliosphere_temporal_overlay_2026-03-13.toml".to_string(),
                latest_desi_report,
            ],
            notes: "Connects the theory scaffold discussed in this thread to the executed empirical outputs without claiming that those outputs are themselves Planck-scale measurements.",
        }, &binaries, &experiments, &claims),
    ];

    let report = CrosswalkReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        db_path: cli.db.display().to_string(),
        rows,
    };
    write_toml_report(&report_path, &report)?;
    println!("Rows:   {}", report.rows.len());
    println!("Report: {}", report_path.display());
    Ok(())
}

fn build_row(
    spec: CrosswalkSpec<'_>,
    binaries: &BTreeSet<String>,
    experiments: &BTreeSet<String>,
    claims: &BTreeSet<String>,
) -> CrosswalkRow {
    let status = if spec.binary_refs.iter().any(|name| binaries.contains(*name)) {
        "linked_runtime"
    } else if !spec.experiment_refs.is_empty() || !spec.claim_refs.is_empty() {
        "db_linked_but_no_binary"
    } else {
        "bibliography_only"
    };
    CrosswalkRow {
        family: spec.family.to_string(),
        status: status.to_string(),
        bibliography_refs: spec.bibliography_refs.iter().map(|s| s.to_string()).collect(),
        binaries: spec.binary_refs
            .iter()
            .filter(|name| binaries.contains(**name))
            .map(|s| s.to_string())
            .collect(),
        experiments: spec.experiment_refs
            .iter()
            .filter(|id| experiments.contains(**id))
            .map(|s| s.to_string())
            .collect(),
        claims: spec.claim_refs
            .iter()
            .filter(|id| claims.contains(**id))
            .map(|s| s.to_string())
            .collect(),
        empirical_outputs: spec.empirical_outputs,
        notes: spec.notes.to_string(),
    }
}

fn load_string_set(conn: &Connection, sql: &str) -> Result<BTreeSet<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut set = BTreeSet::new();
    for row in rows {
        set.insert(row?);
    }
    Ok(set)
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string_pretty(value)?)?;
    Ok(())
}

fn latest_report_path(prefix: &str) -> Option<String> {
    let mut matches = fs::read_dir("reports")
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(|name| name.starts_with(prefix) && name.ends_with(".toml"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.last().map(|path| path.display().to_string())
}
