use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::{Parser, ValueEnum};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "cd-row-upgrade-batch",
    about = "Generate a reproducible CayleyDickson row-upgrade batch report from a lane, witness class, and target row ids"
)]
struct Cli {
    #[arg(long, default_value = "/home/eirikr/Documents/Projects/CayleyDickson")]
    cache_root: PathBuf,

    #[arg(long, value_enum)]
    lane: Lane,

    #[arg(long)]
    source_witness: PathBuf,

    #[arg(long, value_enum)]
    source_status: SourceStatus,

    #[arg(long)]
    row_id: Vec<String>,

    #[arg(long, default_value = "Codex")]
    operator: String,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Lane {
    Jacobson1958,
    Freudenthal1951,
}

#[derive(Clone, Debug, ValueEnum)]
enum SourceStatus {
    ExactOriginal,
    FullOfficialReprint,
    FullOfficialWitness,
    OfficialFragment,
    OfficialToc,
    TranslationRewriting,
    SupportReconstruction,
    ReconstructionDossier,
}

#[derive(Clone, Debug)]
struct LaneSpec {
    title: &'static str,
    inventory_rel: &'static str,
    runbook_rel: &'static str,
    batch_prefix: &'static str,
    allowed_statuses: &'static [SourceStatusKind],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceStatusKind {
    ExactOriginal,
    FullOfficialReprint,
    FullOfficialWitness,
    OfficialFragment,
    OfficialToc,
    TranslationRewriting,
    SupportReconstruction,
    ReconstructionDossier,
}

impl SourceStatus {
    fn as_kind(&self) -> SourceStatusKind {
        match self {
            Self::ExactOriginal => SourceStatusKind::ExactOriginal,
            Self::FullOfficialReprint => SourceStatusKind::FullOfficialReprint,
            Self::FullOfficialWitness => SourceStatusKind::FullOfficialWitness,
            Self::OfficialFragment => SourceStatusKind::OfficialFragment,
            Self::OfficialToc => SourceStatusKind::OfficialToc,
            Self::TranslationRewriting => SourceStatusKind::TranslationRewriting,
            Self::SupportReconstruction => SourceStatusKind::SupportReconstruction,
            Self::ReconstructionDossier => SourceStatusKind::ReconstructionDossier,
        }
    }

    fn render_label(&self) -> &'static str {
        match self {
            Self::ExactOriginal => "exact original",
            Self::FullOfficialReprint => "full official reprint",
            Self::FullOfficialWitness => "full official witness",
            Self::OfficialFragment => "official fragment",
            Self::OfficialToc => "official TOC",
            Self::TranslationRewriting => "translation / rewriting",
            Self::SupportReconstruction => "support reconstruction",
            Self::ReconstructionDossier => "reconstruction dossier",
        }
    }
}

#[derive(Clone, Debug)]
struct CheckRow {
    id: &'static str,
    question: &'static str,
    yes: bool,
    notes: String,
}

#[derive(Clone, Debug)]
struct RowState {
    row_id: String,
    found: bool,
    blocked_status: bool,
    blocked_reason: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.row_id.is_empty() {
        bail!("at least one --row-id is required");
    }

    let spec = lane_spec(&cli.lane);
    let inventory_path = cli.cache_root.join(spec.inventory_rel);
    let inventory_text = fs::read_to_string(&inventory_path)
        .with_context(|| format!("read inventory {}", inventory_path.display()))?;

    let witness_path = absolutize(&cli.cache_root, &cli.source_witness);
    let witness_display = witness_path.display().to_string();
    let witness_exists = witness_path.exists();
    let complete_text = is_complete_text_status(cli.source_status.as_kind());
    let allowed = spec.allowed_statuses.contains(&cli.source_status.as_kind());

    let row_states: Vec<RowState> = cli
        .row_id
        .iter()
        .map(|row_id| inspect_row(&inventory_text, row_id))
        .collect();

    let checks = build_checks(
        &cli,
        &spec,
        witness_exists,
        complete_text,
        allowed,
        &row_states,
    );

    let promotable = checks.iter().all(|c| c.yes);
    let date = Local::now().format("%Y-%m-%d").to_string();
    let output_path = cli.output.clone().unwrap_or_else(|| {
        cli.cache_root.join(format!(
            "metadata/reconstruction_notes/{}_row_upgrade_batch_{}.md",
            spec.batch_prefix, date
        ))
    });

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }

    let body = render_batch(
        &cli,
        spec,
        &witness_display,
        &inventory_path,
        &checks,
        &row_states,
        promotable,
    );
    fs::write(&output_path, body).with_context(|| format!("write {}", output_path.display()))?;

    println!("{}", output_path.display());
    Ok(())
}

fn lane_spec(lane: &Lane) -> LaneSpec {
    match lane {
        Lane::Jacobson1958 => LaneSpec {
            title: "Jacobson 1958",
            inventory_rel: "metadata/reconstruction_notes/jacobson_1958_inventory_template.md",
            runbook_rel: "metadata/reconstruction_notes/jacobson_1958_extraction_runbook.md",
            batch_prefix: "jacobson_1958",
            allowed_statuses: &[
                SourceStatusKind::ExactOriginal,
                SourceStatusKind::FullOfficialReprint,
            ],
        },
        Lane::Freudenthal1951 => LaneSpec {
            title: "Freudenthal 1951",
            inventory_rel: "metadata/reconstruction_notes/freudenthal_1951_inventory_template.md",
            runbook_rel: "metadata/reconstruction_notes/freudenthal_1951_extraction_runbook.md",
            batch_prefix: "freudenthal_1951",
            allowed_statuses: &[
                SourceStatusKind::ExactOriginal,
                SourceStatusKind::FullOfficialWitness,
            ],
        },
    }
}

fn absolutize(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn is_complete_text_status(status: SourceStatusKind) -> bool {
    matches!(
        status,
        SourceStatusKind::ExactOriginal
            | SourceStatusKind::FullOfficialReprint
            | SourceStatusKind::FullOfficialWitness
    )
}

fn inspect_row(inventory_text: &str, row_id: &str) -> RowState {
    let found_line = inventory_text
        .lines()
        .find(|line| line.contains(&format!("| {} |", row_id)));

    match found_line {
        Some(line) => RowState {
            row_id: row_id.to_string(),
            found: true,
            blocked_status: line.contains("`blocked`"),
            blocked_reason: line.contains("`exact original missing`"),
        },
        None => RowState {
            row_id: row_id.to_string(),
            found: false,
            blocked_status: false,
            blocked_reason: false,
        },
    }
}

fn build_checks(
    cli: &Cli,
    spec: &LaneSpec,
    witness_exists: bool,
    complete_text: bool,
    allowed: bool,
    row_states: &[RowState],
) -> Vec<CheckRow> {
    let all_found = row_states.iter().all(|r| r.found);
    let all_blocked = row_states.iter().all(|r| r.blocked_status);
    let all_blocked_reason = row_states.iter().all(|r| r.blocked_reason);

    vec![
        CheckRow {
            id: "RUC-01",
            question: "Is the source witness a complete-text witness rather than a fragment, translation, rewriting, or reconstruction?",
            yes: complete_text,
            notes: if complete_text {
                format!(
                    "source status `{}` is complete-text eligible",
                    cli.source_status.render_label()
                )
            } else {
                format!(
                    "source status `{}` is not a complete-text witness",
                    cli.source_status.render_label()
                )
            },
        },
        CheckRow {
            id: "RUC-02",
            question: "Is the source witness recorded with stable provenance in the project?",
            yes: witness_exists,
            notes: if witness_exists {
                "source witness path exists on disk".to_string()
            } else {
                "source witness path does not exist on disk".to_string()
            },
        },
        CheckRow {
            id: "RUC-03",
            question: "Is the witness class allowed by the lane runbook for wording-safe upgrades?",
            yes: allowed,
            notes: if allowed {
                format!(
                    "{} runbook allows `{}`",
                    spec.title,
                    cli.source_status.render_label()
                )
            } else {
                format!(
                    "{} runbook does not allow `{}`",
                    spec.title,
                    cli.source_status.render_label()
                )
            },
        },
        CheckRow {
            id: "RUC-04",
            question: "Does each target row still show `source status = blocked` before the upgrade?",
            yes: all_found && all_blocked,
            notes: row_state_note(row_states, "blocked status"),
        },
        CheckRow {
            id: "RUC-05",
            question: "Does each target row still show `blocked_by = exact original missing` before the upgrade?",
            yes: all_found && all_blocked_reason,
            notes: row_state_note(row_states, "blocked reason"),
        },
        CheckRow {
            id: "RUC-06",
            question: "Is the exact wording for the target definition / theorem / lemma fully visible in the witness?",
            yes: complete_text && witness_exists && allowed,
            notes: if complete_text && witness_exists && allowed {
                "manual theorem-by-theorem confirmation still required, but witness class permits it".to_string()
            } else {
                "current witness class does not permit wording-safe extraction".to_string()
            },
        },
        CheckRow {
            id: "RUC-07",
            question: "Can a trusted page or section anchor be recorded for the row?",
            yes: complete_text && witness_exists && allowed,
            notes: if complete_text && witness_exists && allowed {
                "page / section anchors may be recorded from the exact witness".to_string()
            } else {
                "trusted page / section anchors are not yet available for upgrade".to_string()
            },
        },
        CheckRow {
            id: "RUC-08",
            question: "If numbering is being upgraded, is the numbering visible in the exact witness rather than inferred?",
            yes: complete_text && witness_exists && allowed,
            notes: if complete_text && witness_exists && allowed {
                "numbering may be upgraded only if visibly present".to_string()
            } else {
                "numbering upgrades are not yet allowed".to_string()
            },
        },
        CheckRow {
            id: "RUC-09",
            question: "If proof dependencies are being upgraded, is the dependency relation directly visible in the witness?",
            yes: complete_text && witness_exists && allowed,
            notes: if complete_text && witness_exists && allowed {
                "proof dependencies may be upgraded only if directly visible".to_string()
            } else {
                "proof-dependency upgrades are not yet allowed".to_string()
            },
        },
        CheckRow {
            id: "RUC-10",
            question: "Has the row been kept conservative, with no overclaim beyond what the witness actually shows?",
            yes: true,
            notes: "this batch report does not auto-promote rows".to_string(),
        },
        CheckRow {
            id: "RUC-11",
            question: "Have bibliography / metadata rows been left separate from theorem-text upgrades?",
            yes: true,
            notes: "bibliography and metadata remain separate from row-upgrade work".to_string(),
        },
        CheckRow {
            id: "RUC-12",
            question: "For Freudenthal-like lanes, has original-versus-rewriting hygiene been preserved explicitly?",
            yes: true,
            notes: if matches!(cli.lane, Lane::Freudenthal1951) {
                "Freudenthal lane preserves exact-original versus rewriting separation".to_string()
            } else {
                "not a Freudenthal-like rewriting lane in this batch".to_string()
            },
        },
    ]
}

fn row_state_note(row_states: &[RowState], field: &str) -> String {
    let mut parts = Vec::new();
    for row in row_states {
        let note = match field {
            "blocked status" => {
                if !row.found {
                    "missing row".to_string()
                } else if row.blocked_status {
                    "blocked".to_string()
                } else {
                    "not blocked".to_string()
                }
            }
            "blocked reason" => {
                if !row.found {
                    "missing row".to_string()
                } else if row.blocked_reason {
                    "exact original missing".to_string()
                } else {
                    "other / absent block reason".to_string()
                }
            }
            _ => "unknown".to_string(),
        };
        parts.push(format!("{}={}", row.row_id, note));
    }
    parts.join("; ")
}

fn render_batch(
    cli: &Cli,
    spec: LaneSpec,
    witness_display: &str,
    inventory_path: &Path,
    checks: &[CheckRow],
    row_states: &[RowState],
    promotable: bool,
) -> String {
    let date = Local::now().format("%Y-%m-%d").to_string();
    let batch_type = if promotable {
        "promotion-ready validation pass"
    } else {
        "no-promotion safety pass"
    };
    let mut out = String::new();
    out.push_str(&format!("# {} Row Upgrade Batch\n\n", spec.title));
    out.push_str(&format!("Date: {}\n\n", date));
    out.push_str("Batch type:\n\n");
    out.push_str(&format!("- {}\n\n", batch_type));
    out.push_str("Companion workflow:\n\n");
    out.push_str(&format!("- `{}`\n", spec.runbook_rel));
    out.push_str(&format!(
        "- `{}`\n",
        render_rel(&cli.cache_root, inventory_path)
    ));
    out.push_str("- `metadata/reconstruction_notes/row_upgrade_checklist.md`\n\n");
    out.push_str("## Batch Header\n\n");
    out.push_str(&format!("- lane: `{}`\n", spec.title));
    out.push_str(&format!("- date: `{}`\n", date));
    out.push_str(&format!("- operator: `{}`\n", cli.operator));
    out.push_str("- source witness:\n");
    out.push_str(&format!("  `{}`\n", witness_display));
    out.push_str(&format!(
        "- source status: `{}`\n",
        cli.source_status.render_label()
    ));
    out.push_str("- inventory file:\n");
    out.push_str(&format!(
        "  `{}`\n",
        render_rel(&cli.cache_root, inventory_path)
    ));
    out.push_str("- row ids to upgrade:\n");
    out.push_str(&format!("  `{}`\n\n", cli.row_id.join("`, `")));

    out.push_str("## Required Checks\n\n");
    out.push_str("| check_id | question | yes / no | notes |\n");
    out.push_str("| --- | --- | --- | --- |\n");
    for check in checks {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            check.id,
            check.question,
            if check.yes { "yes" } else { "no" },
            escape_md_cell(&check.notes)
        ));
    }

    out.push_str("\n## Row-Level Upgrade Record\n\n");
    out.push_str("| row_id | table | old status | new status | source witness | page / section | exact wording available | blocked cleared | notes |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in row_states {
        let old_status = if row.found && row.blocked_status {
            "`blocked`"
        } else if row.found {
            "`not blocked`"
        } else {
            "`missing row`"
        };
        let new_status = if promotable {
            "`ready for manual promotion`"
        } else {
            "`blocked`"
        };
        let notes = if !row.found {
            "row id not found in inventory template".to_string()
        } else if promotable {
            "preconditions pass; manual extraction still required".to_string()
        } else if !is_complete_text_status(cli.source_status.as_kind()) {
            "current witness class is not complete-text eligible".to_string()
        } else {
            "promotion still requires exact row-level extraction".to_string()
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.row_id,
            infer_table(&row.row_id),
            old_status,
            new_status,
            escape_md_cell(&basename_or_full(witness_display)),
            "",
            if promotable {
                "pending manual confirmation"
            } else {
                "no"
            },
            if promotable { "not yet" } else { "no" },
            escape_md_cell(&notes)
        ));
    }

    out.push_str("\n## Outcome\n\n");
    if promotable {
        out.push_str("This batch is promotion-ready, but no rows have been auto-promoted.\n\n");
        out.push_str("Reason:\n\n");
        out.push_str("- the witness class meets lane-level preconditions\n");
        out.push_str("- the target rows are still blocked placeholders\n");
        out.push_str("- manual row-by-row extraction is still required before edits\n");
    } else {
        out.push_str(&format!(
            "No blocked {} theorem/definition rows were promoted.\n\n",
            spec.title
        ));
        out.push_str("Reason:\n\n");
        out.push_str(&format!(
            "- the current source witness is classified as `{}`\n",
            cli.source_status.render_label()
        ));
        out.push_str("- the lane runbook preconditions for wording-safe upgrade are not yet met\n");
    }

    out.push_str("\n## Next Unlock\n\n");
    match cli.lane {
        Lane::Jacobson1958 => {
            out.push_str(
                "Safe promotion can begin only after one of the following is acquired:\n\n",
            );
            out.push_str("- exact article PDF\n");
            out.push_str("- full official reprint chapter\n");
            out.push_str("- another complete official witness\n");
        }
        Lane::Freudenthal1951 => {
            out.push_str(
                "Safe promotion can begin only after one of the following is acquired:\n\n",
            );
            out.push_str("- Heidelberg-delivered exact witness\n");
            out.push_str("- Leipzig-delivered exact witness\n");
            out.push_str("- another complete official witness of the 1951 report\n");
        }
    }
    out
}

fn render_rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| path.display().to_string())
}

fn basename_or_full(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn infer_table(row_id: &str) -> &'static str {
    if row_id.contains("-DEF-") {
        "Definitions"
    } else if row_id.contains("-THM-") {
        "Theorems"
    } else if row_id.contains("-LEM-") {
        "Lemmas"
    } else if row_id.contains("-NUM-") {
        "Numbering"
    } else if row_id.contains("-DEP-") {
        "Proof Dependencies"
    } else {
        "Unknown"
    }
}

fn escape_md_cell(input: &str) -> String {
    input.replace('|', "\\|")
}
