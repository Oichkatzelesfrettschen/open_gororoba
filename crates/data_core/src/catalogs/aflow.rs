//! AFLOW materials database provider (aflowlib.duke.edu).
//!
//! The Automatic FLOW for Materials Discovery (AFLOW) provides DFT-computed
//! material properties for 3.7M+ entries.  We use the AFLUX REST API to
//! paginate through the full database (500 records/page).
//!
//! Source: <https://aflow.org/>
//! API docs: <https://aflow.org/API/aflux/>

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// One AFLOW material record (subset of available fields).
#[derive(Debug, Clone)]
pub struct AflowMaterial {
    /// Unique AFLOW ID.
    pub auid: String,
    /// Chemical formula (e.g., "Al2O3").
    pub compound: String,
    /// Element list (e.g., ["Al", "O"]).
    pub species: Vec<String>,
    /// Number of unique species.
    pub nspecies: usize,
    /// Number of atoms in the unit cell.
    pub natoms: usize,
    /// Enthalpy of formation per atom (eV/atom).
    pub enthalpy_formation_atom: f64,
    /// Electronic band gap (eV).  Zero for metals.
    pub egap: f64,
    /// Density in g/cm^3.
    pub density: Option<f64>,
    /// Volume per atom in A^3/atom.
    pub volume_atom: Option<f64>,
    /// ITC space group number.
    pub spacegroup: Option<u32>,
    /// Pearson symbol (e.g., "cF8").
    pub pearson_symbol: Option<String>,
}

/// AFLUX base URL for the REST API.
const AFLUX_BASE: &str = "https://aflow.org/API/aflux/";

/// Records per AFLUX API page.  AFLUX default maximum is 64 but we request
/// larger pages to reduce round-trips over the full database.
const PER_PAGE: usize = 500;

/// Build the AFLUX query URL for a given page number (0-indexed).
///
/// `$paging(page)` returns one page of PER_PAGE records.
fn aflux_url(page: usize) -> String {
    format!(
        "{}?auid(*),compound(*),Egap(*),enthalpy_formation_atom(*),\
         volume_atom(*),density(*),species(*),nspecies(*),natoms(*),\
         Pearson_symbol_relax(*),spacegroup_relax(*),\
         $paging({page}),format(json)",
        AFLUX_BASE,
        page = page
    )
}

/// Parse a single JSON record from the AFLUX response.
fn parse_one_record(rec: &serde_json::Value) -> Option<AflowMaterial> {
    let auid = rec.get("auid")?.as_str()?.to_string();
    let compound = rec
        .get("compound")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Species can be a comma-separated string or array
    let species = parse_species(rec.get("species")?);
    if species.is_empty() {
        return None;
    }

    let nspecies = rec
        .get("nspecies")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(species.len() as u64) as usize;

    let natoms = rec
        .get("natoms")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(0) as usize;

    let enthalpy_formation_atom = parse_f64_field(rec, "enthalpy_formation_atom")?;
    let egap = parse_f64_field(rec, "Egap")?;
    let density = parse_f64_field(rec, "density");
    let volume_atom = parse_f64_field(rec, "volume_atom");

    let spacegroup = rec
        .get("spacegroup_relax")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .map(|n| n as u32);

    let pearson_symbol = rec
        .get("Pearson_symbol_relax")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    Some(AflowMaterial {
        auid,
        compound,
        species,
        nspecies,
        natoms,
        enthalpy_formation_atom,
        egap,
        density,
        volume_atom,
        spacegroup,
        pearson_symbol,
    })
}

/// Parse species from either a comma-separated string or a JSON array.
fn parse_species(val: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = val.as_array() {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect()
    } else if let Some(s) = val.as_str() {
        s.split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect()
    } else {
        Vec::new()
    }
}

/// Parse a numeric field that may come as number or string.
fn parse_f64_field(rec: &serde_json::Value, key: &str) -> Option<f64> {
    rec.get(key).and_then(|v| {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    })
}

/// Parse a page of AFLUX JSON records.
pub fn parse_aflow_records(json_str: &str) -> Result<Vec<AflowMaterial>, FetchError> {
    let records: Vec<serde_json::Value> = serde_json::from_str(json_str)
        .map_err(|e| FetchError::Validation(format!("AFLOW JSON parse error: {e}")))?;

    let materials: Vec<AflowMaterial> = records.iter().filter_map(parse_one_record).collect();
    Ok(materials)
}

/// Parse the saved AFLOW JSON file into typed records.
pub fn parse_aflow_json(path: &Path) -> Result<Vec<AflowMaterial>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {e}")))?;
    parse_aflow_records(&content)
}

/// Download one page from the AFLUX API.
fn fetch_aflow_page(page: usize) -> Result<String, FetchError> {
    let url = aflux_url(page);
    crate::fetcher::download_to_string(&url)
}

/// Download the full AFLOW dataset (paginated) and save as a single JSON array.
///
/// Paginates through the entire AFLUX result set until an empty page is
/// returned, collecting all records.  Progress is printed to stderr.
pub fn fetch_aflow_dataset(config: &FetchConfig) -> Result<PathBuf, FetchError> {
    let dest = config.output_dir.join("aflow_materials.json");
    if config.skip_existing && dest.exists() {
        return Ok(dest);
    }

    let mut all_records: Vec<serde_json::Value> = Vec::new();
    let mut page = 0;

    loop {
        eprintln!("  AFLOW page {page} ...");
        let body = fetch_aflow_page(page)?;
        let page_records: Vec<serde_json::Value> = serde_json::from_str(&body).map_err(|e| {
            FetchError::Validation(format!("AFLOW page {page} JSON parse error: {e}"))
        })?;

        if page_records.is_empty() {
            break;
        }

        let n = page_records.len();
        all_records.extend(page_records);
        eprintln!("    +{n} records (total: {})", all_records.len());

        // If we got fewer than PER_PAGE, we've reached the last page
        if n < PER_PAGE {
            break;
        }
        page += 1;
    }

    eprintln!("  AFLOW total: {} records", all_records.len());

    let json_out = serde_json::to_string(&all_records)
        .map_err(|e| FetchError::Validation(format!("JSON serialize error: {e}")))?;

    std::fs::create_dir_all(&config.output_dir)?;
    std::fs::write(&dest, json_out)?;

    Ok(dest)
}

/// AFLOW materials database provider for the unified fetch pipeline.
pub struct AflowProvider;

impl DatasetProvider for AflowProvider {
    fn name(&self) -> &str {
        "AFLOW Materials Database"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_aflow_dataset(config)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("aflow_materials.json").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_records() -> String {
        serde_json::json!([
            {
                "auid": "aflow:0001",
                "compound": "Al2O3",
                "species": "Al,O",
                "nspecies": 2,
                "natoms": 10,
                "enthalpy_formation_atom": -1.68,
                "Egap": 5.9,
                "density": 3.95,
                "volume_atom": 8.5,
                "spacegroup_relax": 167,
                "Pearson_symbol_relax": "hR30"
            },
            {
                "auid": "aflow:0002",
                "compound": "Si",
                "species": "Si",
                "nspecies": 1,
                "natoms": 2,
                "enthalpy_formation_atom": 0.0,
                "Egap": 1.12,
                "density": 2.33,
                "volume_atom": 20.0,
                "spacegroup_relax": 227,
                "Pearson_symbol_relax": "cF8"
            },
            {
                "auid": "aflow:0003",
                "compound": "Fe",
                "species": "Fe",
                "nspecies": 1,
                "natoms": 1,
                "enthalpy_formation_atom": 0.0,
                "Egap": 0.0,
                "density": 7.87,
                "volume_atom": 11.8,
                "spacegroup_relax": 229,
                "Pearson_symbol_relax": "cI2"
            }
        ])
        .to_string()
    }

    #[test]
    fn test_parse_aflow_synthetic() {
        let records = parse_aflow_records(&synthetic_records()).unwrap();
        assert_eq!(records.len(), 3);

        assert_eq!(records[0].auid, "aflow:0001");
        assert_eq!(records[0].compound, "Al2O3");
        assert_eq!(records[0].species, vec!["Al", "O"]);
        assert_eq!(records[0].nspecies, 2);
        assert_eq!(records[0].natoms, 10);
        assert!((records[0].enthalpy_formation_atom - (-1.68)).abs() < 1e-6);
        assert!((records[0].egap - 5.9).abs() < 1e-6);
        assert!((records[0].density.unwrap() - 3.95).abs() < 1e-6);
        assert_eq!(records[0].spacegroup, Some(167));
        assert_eq!(records[0].pearson_symbol.as_deref(), Some("hR30"));
    }

    #[test]
    fn test_parse_aflow_missing_fields() {
        let json = serde_json::json!([{
            "auid": "aflow:0010",
            "compound": "Cu",
            "species": "Cu",
            "nspecies": 1,
            "natoms": 1,
            "enthalpy_formation_atom": 0.0,
            "Egap": 0.0
        }])
        .to_string();

        let records = parse_aflow_records(&json).unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].density.is_none());
        assert!(records[0].volume_atom.is_none());
        assert!(records[0].spacegroup.is_none());
        assert!(records[0].pearson_symbol.is_none());
    }

    #[test]
    fn test_parse_aflow_zero_egap() {
        let json = serde_json::json!([{
            "auid": "aflow:0020",
            "compound": "Fe",
            "species": "Fe",
            "nspecies": 1,
            "natoms": 1,
            "enthalpy_formation_atom": 0.0,
            "Egap": 0.0
        }])
        .to_string();

        let records = parse_aflow_records(&json).unwrap();
        assert_eq!(records.len(), 1);
        assert!((records[0].egap).abs() < 1e-12);
    }

    #[test]
    fn test_parse_aflow_empty_species() {
        let json = serde_json::json!([{
            "auid": "aflow:0030",
            "compound": "???",
            "species": "",
            "nspecies": 0,
            "natoms": 0,
            "enthalpy_formation_atom": 0.0,
            "Egap": 0.0
        }])
        .to_string();

        let records = parse_aflow_records(&json).unwrap();
        assert_eq!(
            records.len(),
            0,
            "records with empty species should be skipped"
        );
    }

    #[test]
    fn test_parse_aflow_records_structure() {
        let records = parse_aflow_records(&synthetic_records()).unwrap();
        // Silicon: Egap = 1.12, formation_energy = 0
        let si = &records[1];
        assert_eq!(si.compound, "Si");
        assert!((si.egap - 1.12).abs() < 1e-6);
        assert!((si.enthalpy_formation_atom).abs() < 1e-6);
        assert_eq!(si.species, vec!["Si"]);
        assert_eq!(si.nspecies, 1);
        assert_eq!(si.natoms, 2);
    }

    #[test]
    fn test_aflow_if_available() {
        let path = std::path::Path::new("data/external/aflow_materials.json");
        if !path.exists() {
            eprintln!("Skipping AFLOW live test (no cached file)");
            return;
        }
        let records = parse_aflow_json(path).unwrap();
        assert!(
            records.len() > 100,
            "Expected 100+ AFLOW records, got {}",
            records.len()
        );
        // All records should have non-empty species
        for rec in &records {
            assert!(
                !rec.species.is_empty(),
                "Record {} has empty species",
                rec.auid
            );
        }
    }

    #[test]
    fn test_parse_species_from_array() {
        let val = serde_json::json!(["Al", "O"]);
        let species = parse_species(&val);
        assert_eq!(species, vec!["Al", "O"]);
    }

    #[test]
    fn test_parse_species_from_string() {
        let val = serde_json::json!("Fe,Ni,Cr");
        let species = parse_species(&val);
        assert_eq!(species, vec!["Fe", "Ni", "Cr"]);
    }

    #[test]
    fn test_parse_f64_field_from_string() {
        let rec = serde_json::json!({"val": "3.14"});
        assert!((parse_f64_field(&rec, "val").unwrap() - 3.14).abs() < 1e-6);
    }
}
