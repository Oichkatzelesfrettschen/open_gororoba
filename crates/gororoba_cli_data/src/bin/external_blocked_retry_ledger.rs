use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use gororoba_cli::data_governance::{
    DEFAULT_EXTERNAL_SOURCES_PATH, ExternalSourceRule, load_external_sources,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "external-blocked-retry-ledger",
    about = "Append retry/status records for blocked external source contracts"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    sources: PathBuf,
    #[arg(
        long,
        default_value = "reports/external_blocked_retry_ledger_2026-02-24.tsv"
    )]
    ledger: PathBuf,
    #[arg(long, value_delimiter = ',')]
    source_id: Vec<String>,
    #[arg(long, default_value = "attempted")]
    status: String,
    #[arg(long, default_value = "phase3_governance")]
    phase: String,
    #[arg(long, default_value = "codex")]
    actor: String,
    #[arg(long)]
    command: Option<String>,
    #[arg(long, default_value = "")]
    note: String,
    #[arg(long, value_delimiter = ',')]
    evidence_ref: Vec<String>,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    seed_missing: bool,
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

    let sources = load_external_sources(&args.sources)?;
    let blocked: BTreeMap<String, ExternalSourceRule> = sources
        .source
        .iter()
        .filter(|rule| rule.is_blocked())
        .map(|rule| (rule.id.clone(), rule.clone()))
        .collect();

    if blocked.is_empty() {
        anyhow::bail!("no blocked sources found in {}", args.sources.display());
    }

    let existing_ids = read_existing_source_ids(&args.ledger)?;
    let mut target_ids: Vec<String> = if args.source_id.is_empty() {
        if !args.seed_missing {
            anyhow::bail!(
                "provide --source-id <id[,id2]> or set --seed-missing true to append missing blocked sources"
            );
        }
        blocked
            .keys()
            .filter(|id| !existing_ids.contains(*id))
            .cloned()
            .collect()
    } else {
        args.source_id.clone()
    };

    target_ids.sort();
    target_ids.dedup();

    if target_ids.is_empty() {
        println!("EXTERNAL_BLOCKED_RETRY_LEDGER");
        println!("  appended_records=0");
        println!("  detail=no target ids to append");
        return Ok(());
    }

    let mut missing = Vec::new();
    let mut not_blocked = Vec::new();
    for source_id in &target_ids {
        if !blocked.contains_key(source_id) {
            if sources.source.iter().any(|rule| rule.id == *source_id) {
                not_blocked.push(source_id.clone());
            } else {
                missing.push(source_id.clone());
            }
        }
    }
    if !missing.is_empty() {
        anyhow::bail!("unknown source_id(s): {}", missing.join(", "));
    }
    if !not_blocked.is_empty() {
        anyhow::bail!(
            "source_id(s) not blocked and cannot be appended to blocked retry ledger: {}",
            not_blocked.join(", ")
        );
    }

    println!("EXTERNAL_BLOCKED_RETRY_LEDGER");
    println!("  ledger={}", args.ledger.display());
    println!("  dry_run={}", args.dry_run);
    println!("  target_sources={}", target_ids.len());

    if args.dry_run {
        for source_id in &target_ids {
            println!("DRY_RUN source_id={source_id} status={}", args.status);
        }
        return Ok(());
    }

    append_entries(&AppendConfig {
        ledger: &args.ledger,
        now: &now,
        status: &args.status,
        phase: &args.phase,
        actor: &args.actor,
        command_override: &args.command,
        note: &args.note,
        evidence_refs: &args.evidence_ref,
        source_ids: &target_ids,
        blocked: &blocked,
    })?;

    println!("  appended_records={}", target_ids.len());
    Ok(())
}

struct AppendConfig<'a> {
    ledger: &'a Path,
    now: &'a str,
    status: &'a str,
    phase: &'a str,
    actor: &'a str,
    command_override: &'a Option<String>,
    note: &'a str,
    evidence_refs: &'a [String],
    source_ids: &'a [String],
    blocked: &'a BTreeMap<String, ExternalSourceRule>,
}

fn append_entries(config: &AppendConfig<'_>) -> Result<()> {
    let ledger = config.ledger;
    if let Some(parent) = ledger.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }

    let write_header = !ledger.exists() || fs::metadata(ledger)?.len() == 0;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger)
        .with_context(|| format!("open {} for append", ledger.display()))?;

    if write_header {
        writeln!(
            f,
            "timestamp_utc\tsource_id\tstatus\tphase\tactor\tcommand\tnote\tevidence_refs"
        )
        .with_context(|| format!("write header to {}", ledger.display()))?;
    }

    let joined_evidence = config
        .evidence_refs
        .iter()
        .map(|value| sanitize_tsv(value))
        .collect::<Vec<String>>()
        .join("|");

    for source_id in config.source_ids {
        let rule = config
            .blocked
            .get(source_id)
            .with_context(|| format!("missing blocked rule for source_id {}", source_id))?;
        let command = config
            .command_override
            .clone()
            .unwrap_or_else(|| rule.retrieval_method.clone());

        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            sanitize_tsv(config.now),
            sanitize_tsv(source_id),
            sanitize_tsv(config.status),
            sanitize_tsv(config.phase),
            sanitize_tsv(config.actor),
            sanitize_tsv(&command),
            sanitize_tsv(config.note),
            joined_evidence
        )
        .with_context(|| format!("append row to {}", ledger.display()))?;
    }

    Ok(())
}

fn read_existing_source_ids(ledger: &Path) -> Result<BTreeSet<String>> {
    if !ledger.exists() {
        return Ok(BTreeSet::new());
    }
    let raw = fs::read_to_string(ledger).with_context(|| format!("read {}", ledger.display()))?;
    let mut out = BTreeSet::new();
    for (idx, line) in raw.lines().enumerate() {
        if idx == 0 && line.starts_with("timestamp_utc\t") {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut columns = trimmed.split('\t');
        let _timestamp = columns.next();
        if let Some(source_id) = columns.next()
            && !source_id.trim().is_empty()
        {
            out.insert(source_id.trim().to_string());
        }
    }
    Ok(out)
}

fn sanitize_tsv(raw: &str) -> String {
    raw.replace(['\t', '\n', '\r'], " ").trim().to_string()
}
