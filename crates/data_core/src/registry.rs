use serde::{Deserialize, Serialize};

// ============================================================================
// Monograph Registry (monograph.toml)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct MonographRegistry {
    pub monograph: MonographMeta,
    // Canonical shape uses [[document]], legacy shape used [[volumes]]/[[volume]].
    #[serde(default, alias = "document", alias = "volume")]
    pub volumes: Vec<Volume>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonographMeta {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default, alias = "updated")]
    pub last_updated: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub authoritative: Option<bool>,
    #[serde(default)]
    pub document_count: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Volume {
    // Legacy volume fields.
    #[serde(default)]
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub chapters: Vec<Chapter>,
    // Canonical document fields.
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub body_markdown: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Chapter {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub sections: Vec<Section>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Section {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub content: String,
}

// ============================================================================
// Lacunae Registry (lacunae.toml)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct LacunaeRegistry {
    pub lacunae: LacunaeMeta,
    pub lacuna: Vec<LacunaEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LacunaeMeta {
    pub updated: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LacunaEntry {
    pub id: String,
    pub area: String,
    pub title: String,
    pub description: String,
    pub priority: String, // "high", "medium", "low"
    pub status: String,   // "open", "in_progress", "closed"
}

// ============================================================================
// Data Artifact Narratives (data_artifact_narratives.toml)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactRegistry {
    // Canonical header key is [data_artifact_narratives], legacy key is [artifacts].
    #[serde(alias = "data_artifact_narratives")]
    pub artifacts: ArtifactsMeta,
    // Canonical entries use [[document]], legacy entries used [[artifact]].
    #[serde(default, alias = "document")]
    pub artifact: Vec<ArtifactEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactsMeta {
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub authoritative: Option<bool>,
    #[serde(default)]
    pub source_markdown_count: Option<u32>,
    #[serde(default)]
    pub document_count: Option<u32>,
    #[serde(default)]
    pub source_markdown: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub id: String,
    // Legacy fields.
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub generated_by: String, // Command or script
    #[serde(default)]
    pub status: String,
    // Canonical document fields.
    #[serde(default)]
    pub source_markdown: String,
    #[serde(default)]
    pub slug: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub content_kind: String,
    #[serde(default)]
    pub claim_refs: Vec<String>,
    #[serde(default)]
    pub path_refs: Vec<String>,
    #[serde(default)]
    pub line_count: Option<u32>,
    #[serde(default)]
    pub body_markdown: String,
}
