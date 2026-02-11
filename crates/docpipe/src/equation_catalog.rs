//! Equation catalog pipeline utilities inspired by the PhysicsForge scripts.
//!
//! This module provides Rust-native modeling for:
//! - multi-stream inputs (`text`, `pdf_text`, `pdf_ocr`)
//! - normalization and signature generation
//! - deterministic merge and duplicate linking
//! - domain/category classification
//! - parity and gap report modeling
//! - historical CSV conversion and TOML writing

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{DocpipeError, Result};

/// Source stream for equation rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EquationSourceStream {
    Text,
    PdfText,
    PdfOcr,
}

impl EquationSourceStream {
    fn sort_key(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::PdfText => 1,
            Self::PdfOcr => 2,
        }
    }

    /// Stable display name used in output stats.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::PdfText => "pdf_text",
            Self::PdfOcr => "pdf_ocr",
        }
    }
}

/// Catalog domain classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EquationDomain {
    #[serde(rename = "QM")]
    Qm,
    #[serde(rename = "GR")]
    Gr,
    #[serde(rename = "EM")]
    Em,
    #[serde(rename = "Thermo")]
    Thermo,
    #[serde(rename = "Math")]
    Math,
    #[serde(rename = "Experimental")]
    Experimental,
    #[serde(rename = "General")]
    General,
}

impl EquationDomain {
    /// Stable display name used in output stats.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qm => "QM",
            Self::Gr => "GR",
            Self::Em => "EM",
            Self::Thermo => "Thermo",
            Self::Math => "Math",
            Self::Experimental => "Experimental",
            Self::General => "General",
        }
    }

    fn from_hint(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "qm" => Some(Self::Qm),
            "gr" => Some(Self::Gr),
            "em" => Some(Self::Em),
            "thermo" => Some(Self::Thermo),
            "math" => Some(Self::Math),
            "experimental" => Some(Self::Experimental),
            "general" => Some(Self::General),
            _ => None,
        }
    }
}

/// Catalog category classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EquationCategory {
    #[serde(rename = "Fundamental")]
    Fundamental,
    #[serde(rename = "Experimental")]
    Experimental,
    #[serde(rename = "Phenomenological")]
    Phenomenological,
    #[serde(rename = "Derived")]
    Derived,
}

impl EquationCategory {
    /// Stable display name used in output stats.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fundamental => "Fundamental",
            Self::Experimental => "Experimental",
            Self::Phenomenological => "Phenomenological",
            Self::Derived => "Derived",
        }
    }

    fn from_hint(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fundamental" => Some(Self::Fundamental),
            "experimental" => Some(Self::Experimental),
            "phenomenological" => Some(Self::Phenomenological),
            "derived" => Some(Self::Derived),
            _ => None,
        }
    }
}

/// Input row before canonical merge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquationCatalogInputRow {
    pub legacy_eq_id: Option<String>,
    pub equation: String,
    pub framework: String,
    pub domain_hint: Option<String>,
    pub category_hint: Option<String>,
    pub source_doc: String,
    pub source_line: String,
    pub description: String,
    pub verification_status: String,
    pub related_eqs: String,
    pub experimental_test: Option<String>,
    pub importance_score: Option<u32>,
    pub source_stream: EquationSourceStream,
}

impl EquationCatalogInputRow {
    /// Create a minimal row for a given equation and stream.
    pub fn new(
        equation: impl Into<String>,
        framework: impl Into<String>,
        source_stream: EquationSourceStream,
    ) -> Self {
        Self {
            legacy_eq_id: None,
            equation: equation.into(),
            framework: framework.into(),
            domain_hint: None,
            category_hint: None,
            source_doc: String::new(),
            source_line: String::new(),
            description: String::new(),
            verification_status: String::new(),
            related_eqs: String::new(),
            experimental_test: None,
            importance_score: None,
            source_stream,
        }
    }
}

/// Canonical equation row produced by merge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquationCatalogRow {
    pub eq_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_eq_id: Option<String>,
    pub equation: String,
    pub framework: String,
    pub domain: EquationDomain,
    pub category: EquationCategory,
    pub source_doc: String,
    pub source_line: String,
    pub description: String,
    pub verification_status: String,
    pub related_eqs: String,
    pub experimental_test: String,
    pub importance_score: u32,
    pub source_stream: EquationSourceStream,
    pub signature: String,
}

/// Aggregate stats for a merged catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct EquationCatalogStats {
    pub total_rows: usize,
    pub by_framework: BTreeMap<String, usize>,
    pub by_domain: BTreeMap<String, usize>,
    pub by_category: BTreeMap<String, usize>,
    pub by_status: BTreeMap<String, usize>,
    pub by_stream: BTreeMap<String, usize>,
}

/// Parity summary between catalog rows and module inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ParityReport {
    pub row_count: usize,
    pub modules_indexed: usize,
    pub rows_without_module_link: Vec<String>,
    pub unreferenced_modules: Vec<String>,
}

/// One missing-module item in a gap bucket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapItem {
    pub eq_id: String,
    pub importance_score: u32,
    pub source_doc: String,
}

/// Missing-module bucket keyed by framework/domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapBucket {
    pub framework: String,
    pub domain: EquationDomain,
    pub missing_rows: Vec<GapItem>,
}

/// Gap report for rows without module links.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GapReport {
    pub total_missing_module_links: usize,
    pub buckets: Vec<GapBucket>,
}

/// Full TOML-ready catalog payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EquationCatalog {
    pub rows: Vec<EquationCatalogRow>,
    pub stats: EquationCatalogStats,
    pub parity_report: ParityReport,
    pub gap_report: GapReport,
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

/// Normalize equation text for deterministic comparison.
pub fn normalize_equation(raw: &str) -> String {
    let mut cleaned = raw.trim().to_string();
    for (from, to) in [
        ("\\cdot", "*"),
        ("\\times", "*"),
        ("\\left", ""),
        ("\\right", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\!", ""),
        ("**", "^"),
    ] {
        cleaned = cleaned.replace(from, to);
    }

    let mut collapsed = String::with_capacity(cleaned.len());
    let mut last_was_space = false;
    for ch in cleaned.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
            }
            last_was_space = true;
        } else {
            collapsed.push(ch);
            last_was_space = false;
        }
    }
    collapsed.trim().to_ascii_lowercase()
}

/// Signature used for duplicate detection.
pub fn equation_signature(raw: &str) -> String {
    normalize_equation(raw).replace(' ', "")
}

/// Domain classification following PhysicsForge keyword intent.
pub fn classify_domain(equation: &str, description: &str) -> EquationDomain {
    let combined = format!(
        "{} {}",
        equation.to_ascii_lowercase(),
        description.to_ascii_lowercase()
    );

    if contains_any(
        &combined,
        &[
            "electromagnetic",
            "electric",
            "magnetic",
            "maxwell",
            "field",
            "capacitance",
            "inductance",
        ],
    ) {
        return EquationDomain::Em;
    }

    if contains_any(
        &combined,
        &[
            "metric",
            "curvature",
            "einstein",
            "ricci",
            "tensor",
            "spacetime",
            "gravity",
        ],
    ) {
        return EquationDomain::Gr;
    }

    if contains_any(
        &combined,
        &[
            "quantum",
            "wave",
            "psi",
            "hamiltonian",
            "operator",
            "qubit",
            "entangle",
        ],
    ) {
        return EquationDomain::Qm;
    }

    if contains_any(
        &combined,
        &[
            "entropy",
            "temperature",
            "heat",
            "thermal",
            "boltzmann",
            "specific heat",
            "helmholtz",
            "gibbs",
        ],
    ) {
        return EquationDomain::Thermo;
    }

    if contains_any(
        &combined,
        &[
            "group",
            "algebra",
            "lie",
            "symmetry",
            "topology",
            "dimension",
        ],
    ) {
        return EquationDomain::Math;
    }

    if contains_any(
        &combined,
        &["casimir", "measurement", "spectroscopy", "interferometry"],
    ) {
        return EquationDomain::Experimental;
    }

    EquationDomain::General
}

/// Category classification following PhysicsForge keyword intent.
pub fn classify_category(description: &str) -> EquationCategory {
    let lowered = description.to_ascii_lowercase();

    if contains_any(
        &lowered,
        &[
            "kernel",
            "action",
            "lagrangian",
            "hamiltonian",
            "fundamental",
        ],
    ) {
        return EquationCategory::Fundamental;
    }

    if contains_any(
        &lowered,
        &["measurement", "experimental", "observation", "data"],
    ) {
        return EquationCategory::Experimental;
    }

    if contains_any(&lowered, &["model", "phenomenological", "effective"]) {
        return EquationCategory::Phenomenological;
    }

    EquationCategory::Derived
}

/// Heuristic experiment suggestion based on equation/description keywords.
pub fn suggest_experimental_test(equation: &str, description: &str) -> String {
    let combined = format!(
        "{} {}",
        equation.to_ascii_lowercase(),
        description.to_ascii_lowercase()
    );

    for (keyword, suggestion) in [
        (
            "casimir",
            "Casimir force measurement with fractal geometries",
        ),
        ("scalar", "Scalar field interferometry"),
        ("foam", "Quantum foam perturbation detection"),
        ("zpe", "Zero-point energy spectroscopy"),
        ("crystal", "Vibrational spectroscopy in crystals"),
        ("entropy", "Thermal imaging and entropy measurement"),
        ("dimensional", "Dimensional spectroscopy"),
    ] {
        if combined.contains(keyword) {
            return suggestion.to_string();
        }
    }

    "Theoretical validation required".to_string()
}

/// Importance ranking compatible with historical script behavior.
pub fn rank_importance_score(description: &str) -> u32 {
    (description.len() % 10) as u32
}

fn canonical_framework(value: &str, source_stream: EquationSourceStream) -> String {
    let trimmed = value.trim();
    if !trimmed.is_empty() {
        return trimmed.to_string();
    }
    if source_stream == EquationSourceStream::PdfOcr {
        return "OCR".to_string();
    }
    "General".to_string()
}

fn framework_prefix(framework: &str, source_stream: EquationSourceStream) -> &'static str {
    let lowered = framework.to_ascii_lowercase();
    if lowered.contains("aether") {
        return "AE";
    }
    if lowered.contains("genesis") {
        return "GE";
    }
    if lowered.contains("pais") {
        return "PE";
    }
    if lowered.contains("tourmaline") {
        return "TE";
    }
    if lowered.contains("superforce") {
        return "SE";
    }
    if lowered.contains("literature") {
        return "LE";
    }
    if lowered.contains("unified") {
        return "UE";
    }
    if source_stream == EquationSourceStream::PdfOcr {
        return "OE";
    }
    "EQ"
}

fn default_verification_status(source_stream: EquationSourceStream) -> &'static str {
    if source_stream == EquationSourceStream::PdfOcr {
        "OCR_Pending"
    } else {
        "Theoretical"
    }
}

fn next_generated_eq_id(
    prefix: &str,
    counters: &mut BTreeMap<String, u32>,
    used_ids: &BTreeSet<String>,
) -> String {
    let counter = counters.entry(prefix.to_string()).or_insert(0);
    loop {
        *counter += 1;
        let candidate = format!("{prefix}{:03}", *counter);
        if !used_ids.contains(&candidate) {
            return candidate;
        }
    }
}

fn parse_module_ref_path(related_eqs: &str) -> Option<String> {
    let trimmed = related_eqs.trim();
    let stripped = trimmed.strip_prefix("module:")?;
    let path_only = stripped.split('#').next().unwrap_or("").trim();
    if path_only.is_empty() {
        return None;
    }
    Some(path_only.replace('\\', "/"))
}

fn get_csv_value(row: &HashMap<String, String>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| row.get(*key))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Convert one historical CSV artifact into canonical input rows.
pub fn convert_historical_csv_reader<R: Read>(
    reader: R,
    source_stream: EquationSourceStream,
) -> Result<Vec<EquationCatalogInputRow>> {
    let mut csv_reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .flexible(true)
        .from_reader(reader);

    let headers = csv_reader.headers()?.clone();
    let mut rows = Vec::new();

    for record in csv_reader.records() {
        let record = record?;
        let mut map = HashMap::new();
        for (header, value) in headers.iter().zip(record.iter()) {
            map.insert(header.trim().to_ascii_lowercase(), value.to_string());
        }

        let equation = get_csv_value(&map, &["equation", "latex"]).unwrap_or_default();
        if equation.is_empty() {
            continue;
        }

        let framework =
            get_csv_value(&map, &["framework"]).unwrap_or_else(|| "General".to_string());
        let mut row = EquationCatalogInputRow::new(equation, framework, source_stream);
        row.legacy_eq_id = get_csv_value(&map, &["eqid", "equation_id"]);
        row.domain_hint = get_csv_value(&map, &["domain"]);
        row.category_hint = get_csv_value(&map, &["category"]);
        row.source_doc = get_csv_value(&map, &["sourcedoc", "source_doc"]).unwrap_or_default();
        row.source_line = get_csv_value(
            &map,
            &["sourceline", "source_line", "sourceloc", "source_loc"],
        )
        .unwrap_or_default();
        row.description = get_csv_value(&map, &["description"]).unwrap_or_default();
        row.verification_status =
            get_csv_value(&map, &["verificationstatus", "verification_status"]).unwrap_or_default();
        row.related_eqs = get_csv_value(&map, &["relatedeqs", "related_eqs"]).unwrap_or_default();
        row.experimental_test = get_csv_value(&map, &["experimentaltest", "experimental_test"]);
        row.importance_score = get_csv_value(&map, &["importancescore", "importance_score"])
            .and_then(|value| value.parse::<u32>().ok());
        rows.push(row);
    }

    Ok(rows)
}

/// Convert a historical CSV path into canonical input rows.
pub fn convert_historical_csv_path(
    csv_path: &Path,
    source_stream: EquationSourceStream,
) -> Result<Vec<EquationCatalogInputRow>> {
    let bytes = fs::read(csv_path)?;
    convert_historical_csv_reader(bytes.as_slice(), source_stream)
}

/// Build a deterministic canonical catalog from mixed input streams.
pub fn build_catalog(
    mut input_rows: Vec<EquationCatalogInputRow>,
    known_modules: &[String],
) -> EquationCatalog {
    input_rows.sort_by(|a, b| {
        (
            a.source_stream.sort_key(),
            a.source_doc.as_str(),
            a.source_line.as_str(),
            a.framework.as_str(),
            a.equation.as_str(),
            a.description.as_str(),
        )
            .cmp(&(
                b.source_stream.sort_key(),
                b.source_doc.as_str(),
                b.source_line.as_str(),
                b.framework.as_str(),
                b.equation.as_str(),
                b.description.as_str(),
            ))
    });

    let mut used_ids: BTreeSet<String> = BTreeSet::new();
    let mut prefix_counters: BTreeMap<String, u32> = BTreeMap::new();
    let mut first_id_by_signature: HashMap<String, String> = HashMap::new();
    let mut rows = Vec::with_capacity(input_rows.len());

    for input in input_rows {
        let framework = canonical_framework(&input.framework, input.source_stream);
        let description = input.description.trim().to_string();
        let signature = equation_signature(&input.equation);
        let domain = input
            .domain_hint
            .as_deref()
            .and_then(EquationDomain::from_hint)
            .unwrap_or_else(|| classify_domain(&input.equation, &description));
        let category = input
            .category_hint
            .as_deref()
            .and_then(EquationCategory::from_hint)
            .unwrap_or_else(|| classify_category(&description));

        let verification_status = if input.verification_status.trim().is_empty() {
            default_verification_status(input.source_stream).to_string()
        } else {
            input.verification_status.trim().to_string()
        };

        let experimental_test = input
            .experimental_test
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| suggest_experimental_test(&input.equation, &description));

        let importance_score = input
            .importance_score
            .unwrap_or_else(|| rank_importance_score(&description));

        let eq_id = if let Some(legacy_id) = input
            .legacy_eq_id
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            if used_ids.contains(&legacy_id) {
                let prefix = framework_prefix(&framework, input.source_stream);
                next_generated_eq_id(prefix, &mut prefix_counters, &used_ids)
            } else {
                legacy_id
            }
        } else {
            let prefix = framework_prefix(&framework, input.source_stream);
            next_generated_eq_id(prefix, &mut prefix_counters, &used_ids)
        };
        used_ids.insert(eq_id.clone());

        let mut related_eqs = input.related_eqs.trim().to_string();
        if let Some(first_eq_id) = first_id_by_signature.get(&signature) {
            if related_eqs.is_empty() {
                related_eqs = first_eq_id.clone();
            }
        } else {
            first_id_by_signature.insert(signature.clone(), eq_id.clone());
        }

        rows.push(EquationCatalogRow {
            eq_id,
            legacy_eq_id: input.legacy_eq_id,
            equation: input.equation.trim().to_string(),
            framework,
            domain,
            category,
            source_doc: if input.source_doc.trim().is_empty() {
                "unknown".to_string()
            } else {
                input.source_doc.trim().to_string()
            },
            source_line: if input.source_line.trim().is_empty() {
                "0".to_string()
            } else {
                input.source_line.trim().to_string()
            },
            description,
            verification_status,
            related_eqs,
            experimental_test,
            importance_score,
            source_stream: input.source_stream,
            signature,
        });
    }

    let parity_report = build_parity_report(&rows, known_modules);
    let gap_report = build_gap_report(&rows);
    let stats = build_stats(&rows);

    EquationCatalog {
        rows,
        stats,
        parity_report,
        gap_report,
    }
}

fn build_stats(rows: &[EquationCatalogRow]) -> EquationCatalogStats {
    let mut stats = EquationCatalogStats {
        total_rows: rows.len(),
        ..EquationCatalogStats::default()
    };

    for row in rows {
        *stats.by_framework.entry(row.framework.clone()).or_insert(0) += 1;
        *stats
            .by_domain
            .entry(row.domain.as_str().to_string())
            .or_insert(0) += 1;
        *stats
            .by_category
            .entry(row.category.as_str().to_string())
            .or_insert(0) += 1;
        *stats
            .by_status
            .entry(row.verification_status.clone())
            .or_insert(0) += 1;
        *stats
            .by_stream
            .entry(row.source_stream.as_str().to_string())
            .or_insert(0) += 1;
    }

    stats
}

/// Build a parity report for catalog rows and known module paths.
pub fn build_parity_report(rows: &[EquationCatalogRow], known_modules: &[String]) -> ParityReport {
    let row_count = rows.len();
    let rows_without_module_link: Vec<String> = rows
        .iter()
        .filter(|row| parse_module_ref_path(&row.related_eqs).is_none())
        .map(|row| row.eq_id.clone())
        .collect();

    let known_modules_set: BTreeSet<String> = known_modules
        .iter()
        .map(|module| module.trim().replace('\\', "/"))
        .filter(|module| !module.is_empty())
        .collect();

    let referenced_modules: BTreeSet<String> = rows
        .iter()
        .filter_map(|row| parse_module_ref_path(&row.related_eqs))
        .collect();

    let unreferenced_modules = known_modules_set
        .difference(&referenced_modules)
        .cloned()
        .collect::<Vec<_>>();

    ParityReport {
        row_count,
        modules_indexed: known_modules_set.len(),
        rows_without_module_link,
        unreferenced_modules,
    }
}

/// Build a grouped gap report for missing module links.
pub fn build_gap_report(rows: &[EquationCatalogRow]) -> GapReport {
    let mut buckets: BTreeMap<(String, EquationDomain), Vec<GapItem>> = BTreeMap::new();

    for row in rows {
        if parse_module_ref_path(&row.related_eqs).is_some() {
            continue;
        }
        let key = (row.framework.clone(), row.domain);
        buckets.entry(key).or_default().push(GapItem {
            eq_id: row.eq_id.clone(),
            importance_score: row.importance_score,
            source_doc: row.source_doc.clone(),
        });
    }

    let mut bucket_list = Vec::with_capacity(buckets.len());
    for ((framework, domain), mut items) in buckets {
        items.sort_by(|left, right| {
            right
                .importance_score
                .cmp(&left.importance_score)
                .then_with(|| left.eq_id.cmp(&right.eq_id))
        });
        bucket_list.push(GapBucket {
            framework,
            domain,
            missing_rows: items,
        });
    }

    let total_missing_module_links = bucket_list
        .iter()
        .map(|bucket| bucket.missing_rows.len())
        .sum();

    GapReport {
        total_missing_module_links,
        buckets: bucket_list,
    }
}

/// Serialize an equation catalog to TOML text.
pub fn catalog_to_toml(catalog: &EquationCatalog) -> Result<String> {
    toml::to_string_pretty(catalog).map_err(DocpipeError::from)
}

/// Write an equation catalog to a TOML file.
pub fn write_catalog_toml(path: &Path, catalog: &EquationCatalog) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let toml_text = catalog_to_toml(catalog)?;
    fs::write(path, toml_text)?;
    Ok(())
}

fn collect_tex_files(root: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_tex_files(&path, files)?;
        } else if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("tex"))
        {
            files.push(path);
        }
    }
    Ok(())
}

/// Index known equation module files for parity checks.
pub fn index_module_files(modules_dir: &Path) -> Result<Vec<String>> {
    if !modules_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    collect_tex_files(modules_dir, &mut files)?;
    files.sort();

    Ok(files
        .iter()
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .collect())
}
