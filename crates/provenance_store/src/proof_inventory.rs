//! Rocq `_RocqProject` inventory loader and theorem-table rendering.
//!
//! - `load_proof_inventory(proofs_project_path)`: parses the
//!   `_RocqProject` text file, extracts each `verified/*.v` entry,
//!   computes the theorem stem, and indexes the entries by both stem
//!   and the normalized claim id (`C-<digits>`) derived from the stem.
//! - `load_theorems_from_inventory`: builds Vec<TheoremRecord> from the
//!   inventory, attaching claim-link backreferences and using the
//!   referenced claim's statement as the theorem title when one
//!   matches, else falling back to a humanized stem.
//! - `normalize_claims_against_proof_inventory`: walks the claim
//!   records, runs them through `normalize_claim_record`, recomputes
//!   the canonical formal_proof path from the inventory, and re-renders
//!   compat_toml_text if the resolved path changed.
//! - `render_theorem_markdown`: emits the canonical
//!   docs/THEOREM_REGISTRY.md table from the loaded inventory.

use std::path::Path;

use anyhow::Result;
use camino::Utf8PathBuf;
use provenance_core::{ClaimRecord, TheoremRecord};

use super::{
    ProofInventory, ProofInventoryEntry,
    claim_proofs::{
        canonical_formal_proof_for_claim, link_claims_for_proof,
        normalized_claim_id_from_theorem_stem, render_normalized_claim_compat_toml,
    },
    compat_render::compat_markdown_export_header,
    status_normalize::normalize_claim_record,
    toml_helpers::load_text,
};

pub(crate) fn load_proof_inventory(proofs_project_path: &Path) -> Result<ProofInventory> {
    let raw = load_text(proofs_project_path)?;
    let mut inventory = ProofInventory {
        project_raw: raw.clone(),
        ..ProofInventory::default()
    };
    for line in raw.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("verified/") || !trimmed.ends_with(".v") {
            continue;
        }
        let path = Utf8PathBuf::from(format!("proofs/{trimmed}"));
        let stem = Path::new(trimmed)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or(trimmed)
            .to_string();
        let entry = ProofInventoryEntry {
            stem: stem.clone(),
            path,
        };
        if let Some(claim_id) = normalized_claim_id_from_theorem_stem(&stem) {
            inventory
                .verified_by_claim_id
                .entry(claim_id)
                .or_default()
                .push(entry.clone());
        }
        inventory.verified_entries.push(entry);
    }
    Ok(inventory)
}

pub(crate) fn load_theorems_from_inventory(
    repo_root: &Path,
    proof_inventory: &ProofInventory,
    claims: &[ClaimRecord],
) -> Result<Vec<TheoremRecord>> {
    let mut out = Vec::new();
    for entry in &proof_inventory.verified_entries {
        let linked_claim_ids = link_claims_for_proof(&entry.path, &entry.stem, claims);
        let normalized_claim_id = normalized_claim_id_from_theorem_stem(&entry.stem);
        let title = claims
            .iter()
            .find(|claim| {
                claim.id == entry.stem || normalized_claim_id.as_deref() == Some(&claim.id)
            })
            .map(|claim| claim.statement.clone())
            .unwrap_or_else(|| entry.stem.replace('_', " "));
        if !repo_root.join(entry.path.as_str()).exists() {
            continue;
        }
        out.push(TheoremRecord {
            id: entry.stem.clone(),
            title,
            proof_path: entry.path.clone(),
            status: "kernel_checked".to_string(),
            linked_claim_ids,
            source: "_RocqProject".to_string(),
        });
    }
    Ok(out)
}

pub(crate) fn normalize_claims_against_proof_inventory(
    repo_root: &Path,
    claims: &mut [ClaimRecord],
    proof_inventory: &ProofInventory,
) -> Result<()> {
    for claim in claims {
        normalize_claim_record(claim)?;
        let canonical_formal_proof =
            canonical_formal_proof_for_claim(repo_root, claim, proof_inventory);
        if claim.formal_proof != canonical_formal_proof {
            claim.formal_proof = canonical_formal_proof;
            claim.compat_toml_text = render_normalized_claim_compat_toml(claim)?;
        }
    }
    Ok(())
}

pub(crate) fn render_theorem_markdown(source_label: &str, theorems: &[TheoremRecord]) -> String {
    let mut lines = compat_markdown_export_header(source_label);
    lines.extend([
        "# Theorems".to_string(),
        String::new(),
        format!(
            "This file is generated from the canonical SQLite control plane and currently indexes {} Rocq proof files.",
            theorems.len()
        ),
        String::new(),
        "| Theorem | Proof File | Status | Linked Claims |".to_string(),
        "|---|---|---|---|".to_string(),
    ]);
    for theorem in theorems {
        let claims = if theorem.linked_claim_ids.is_empty() {
            "-".to_string()
        } else {
            theorem.linked_claim_ids.join(", ")
        };
        lines.push(format!(
            "| `{}` | `{}` | {} | {} |",
            theorem.id, theorem.proof_path, theorem.status, claims
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}
