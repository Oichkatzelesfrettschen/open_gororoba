use serde::{Deserialize, Serialize};

// ============================================================================
// Monograph Registry (monograph.toml)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct MonographRegistry {
    pub monograph: MonographMeta,
    pub volumes: Vec<Volume>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MonographMeta {
    pub title: String,
    pub authors: Vec<String>,
    pub status: String,
    pub last_updated: String,
    pub version: String,
    pub license: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Volume {
    pub id: String,
    pub title: String,
    pub description: String,
    pub chapters: Vec<Chapter>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Chapter {
    pub id: String,
    pub title: String,
    pub sections: Vec<Section>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Section {
    pub id: String,
    pub title: String,
    pub content: String, // Path to markdown or inline content?
                         // In current monograph.toml, it seems to be inline text or structural.
                         // Let's assume it's just structure for now, or check actual file.
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
    pub artifacts: ArtifactsMeta,
    pub artifact: Vec<ArtifactEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactsMeta {
    pub updated: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub id: String,
    pub path: String,
    pub description: String,
    pub generated_by: String, // Command or script
    pub status: String,
}
