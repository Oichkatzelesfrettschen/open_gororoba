//! JARVIS-DFT materials database from NIST via Figshare.
//!
//! The Joint Automated Repository for Various Integrated Simulations (JARVIS)
//! provides DFT-computed material properties for thousands of structures.
//!
//! Source: Figshare article 6815699
//! https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699

use crate::fetcher::{download_to_file, FetchConfig, FetchError};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Figshare article ID for JARVIS-DFT 3D dataset.
pub const FIGSHARE_ARTICLE_ID: u64 = 6815699;

/// One JARVIS material record (subset of available fields).
#[derive(Debug, Clone)]
pub struct JarvisMaterial {
    pub jid: String,
    pub formula: String,
    pub elements: Vec<String>,
    pub nelements: usize,
    pub energy_per_atom: Option<f64>,
    pub formation_energy_peratom: Option<f64>,
    pub optb88vdw_bandgap: Option<f64>,
    pub ehull: Option<f64>,
    pub spg_symbol: Option<String>,
    pub spg_number: Option<u32>,
    pub density: Option<f64>,
    pub volume: Option<f64>,
}

/// Figshare file metadata.
#[derive(Debug, Clone)]
pub struct FigshareFile {
    pub id: u64,
    pub name: String,
    pub size: u64,
    pub download_url: String,
}

/// List files in a Figshare article via the public API.
pub fn list_figshare_files(article_id: u64) -> Result<Vec<FigshareFile>, FetchError> {
    let url = format!("https://api.figshare.com/v2/articles/{article_id}");
    let body = crate::fetcher::download_to_string(&url)?;

    let parsed: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| FetchError::Validation(format!("JSON parse error: {e}")))?;

    let files = parsed
        .get("files")
        .and_then(|f| f.as_array())
        .ok_or_else(|| FetchError::Validation("No 'files' array in Figshare response".into()))?;

    let mut result = Vec::new();
    for entry in files {
        let id = entry.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let size = entry.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let download_url = entry
            .get("download_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !download_url.is_empty() {
            result.push(FigshareFile {
                id,
                name,
                size,
                download_url,
            });
        }
    }

    Ok(result)
}

/// Download the JARVIS-DFT JSON dataset to the configured output directory.
pub fn fetch_jarvis_json(config: &FetchConfig) -> Result<PathBuf, FetchError> {
    let dest = config.output_dir.join("jarvis_dft_3d.json");
    if config.skip_existing && dest.exists() {
        return Ok(dest);
    }

    let files = list_figshare_files(FIGSHARE_ARTICLE_ID)?;

    // Find the JSON file (prefer smallest match containing "json")
    let mut json_files: Vec<&FigshareFile> = files
        .iter()
        .filter(|f| f.name.contains("json") || f.name.ends_with(".json"))
        .collect();
    json_files.sort_by_key(|f| f.size);

    let target = json_files
        .first()
        .ok_or_else(|| FetchError::Validation("No JSON file in JARVIS Figshare article".into()))?;

    // Check if it's a zip; if so, download and extract
    if target.name.ends_with(".zip") {
        let zip_dest = config.output_dir.join(&target.name);
        download_to_file(&target.download_url, &zip_dest)?;
        extract_json_from_zip(&zip_dest, &dest)?;
    } else {
        download_to_file(&target.download_url, &dest)?;
    }

    Ok(dest)
}

/// Extract a JSON file from a ZIP archive.
fn extract_json_from_zip(zip_path: &Path, json_dest: &Path) -> Result<(), FetchError> {
    let file = std::fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| FetchError::Validation(format!("ZIP open error: {e}")))?;

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| FetchError::Validation(format!("ZIP entry error: {e}")))?;
        if entry.name().ends_with(".json") {
            let mut out = std::fs::File::create(json_dest)?;
            std::io::copy(&mut entry, &mut out)?;
            return Ok(());
        }
    }

    Err(FetchError::Validation(
        "No .json file found in ZIP archive".into(),
    ))
}

/// Parse JARVIS JSON records from a file.
pub fn parse_jarvis_json(path: &Path) -> Result<Vec<JarvisMaterial>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {e}")))?;

    let records: Vec<serde_json::Value> = serde_json::from_str(&content)
        .map_err(|e| FetchError::Validation(format!("JSON parse error: {e}")))?;

    let mut materials = Vec::with_capacity(records.len());
    for rec in &records {
        let mat = parse_one_record(rec);
        if let Some(m) = mat {
            materials.push(m);
        }
    }

    Ok(materials)
}

/// Parse one JSON record into a JarvisMaterial.
fn parse_one_record(rec: &serde_json::Value) -> Option<JarvisMaterial> {
    let jid = rec.get("jid")?.as_str()?.to_string();

    // Extract atoms structure
    let atoms = rec.get("atoms");
    let atom_elements: Vec<String> = atoms
        .and_then(|a| a.get("elements"))
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Derive formula from elements if not present
    let formula = rec
        .get("formula")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| derive_formula(&atom_elements));

    // Unique sorted elements
    let elements: Vec<String> = rec
        .get("elements")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_else(|| {
            let set: BTreeSet<String> = atom_elements.iter().cloned().collect();
            set.into_iter().collect()
        });

    let nelements = rec
        .get("nelements")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(elements.len());

    // Volume from lattice matrix if not present
    let volume = rec
        .get("volume")
        .and_then(|v| v.as_f64())
        .or_else(|| compute_volume_from_lattice(atoms?));

    Some(JarvisMaterial {
        jid,
        formula,
        elements,
        nelements,
        energy_per_atom: rec.get("energy_per_atom").and_then(|v| v.as_f64()),
        formation_energy_peratom: rec.get("formation_energy_peratom").and_then(|v| v.as_f64()),
        optb88vdw_bandgap: rec.get("optb88vdw_bandgap").and_then(|v| v.as_f64()),
        ehull: rec.get("ehull").and_then(|v| v.as_f64()),
        spg_symbol: rec.get("spg_symbol").and_then(|v| v.as_str()).map(String::from),
        spg_number: rec.get("spg_number").and_then(|v| v.as_u64()).map(|n| n as u32),
        density: rec.get("density").and_then(|v| v.as_f64()),
        volume,
    })
}

/// Derive a formula string from a list of element symbols.
fn derive_formula(elements: &[String]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for el in elements {
        *counts.entry(el.as_str()).or_insert(0) += 1;
    }
    let mut formula = String::new();
    for (el, count) in &counts {
        formula.push_str(el);
        if *count > 1 {
            formula.push_str(&count.to_string());
        }
    }
    formula
}

/// Compute volume as |det(lattice_mat)| from the atoms JSON.
fn compute_volume_from_lattice(atoms: &serde_json::Value) -> Option<f64> {
    let mat = atoms.get("lattice_mat")?.as_array()?;
    if mat.len() != 3 {
        return None;
    }

    let row = |i: usize| -> Option<[f64; 3]> {
        let r = mat[i].as_array()?;
        if r.len() != 3 {
            return None;
        }
        Some([r[0].as_f64()?, r[1].as_f64()?, r[2].as_f64()?])
    };

    let a = row(0)?;
    let b = row(1)?;
    let c = row(2)?;

    // det = a . (b x c)
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    let det = a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2];
    Some(det.abs())
}

/// Sample a random subset of materials.
pub fn sample_materials(
    materials: &[JarvisMaterial],
    n: usize,
    seed: u64,
) -> Vec<JarvisMaterial> {
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let k = n.min(materials.len());
    let mut indices: Vec<usize> = (0..materials.len()).collect();
    indices.shuffle(&mut rng);
    indices[..k]
        .iter()
        .map(|&i| materials[i].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_formula() {
        let elements = vec!["Ag".into(), "Te".into(), "Te".into(), "Tl".into()];
        assert_eq!(derive_formula(&elements), "AgTe2Tl");
    }

    #[test]
    fn test_compute_volume_identity_matrix() {
        let atoms = serde_json::json!({
            "lattice_mat": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        });
        let vol = compute_volume_from_lattice(&atoms).unwrap();
        assert!((vol - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_one_record_minimal() {
        let rec = serde_json::json!({
            "jid": "JVASP-X",
            "formation_energy_peratom": -0.1,
            "atoms": {
                "lattice_mat": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "elements": ["Ag", "Te", "Te", "Tl"]
            }
        });
        let mat = parse_one_record(&rec).unwrap();
        assert_eq!(mat.jid, "JVASP-X");
        assert_eq!(mat.formula, "AgTe2Tl");
        assert_eq!(mat.nelements, 3);
        assert!((mat.volume.unwrap() - 1.0).abs() < 1e-12);
    }
}
