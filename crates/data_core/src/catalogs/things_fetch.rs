//! Fetch/provider support for the THINGS catalog.

use super::things::{
    THINGS_MPIA_DATA_URL, THINGS_MPIA_ROOT, THINGS_VIZIER_ROOT, ThingsCubeManifestEntry,
    extract_href_values, parse_manifest_entry_from_filename, preferred_things_cube_entries,
    things_name_key, write_things_cube_manifest_csv,
};
use crate::fetcher::{
    DatasetProvider, FetchConfig, FetchError, download_to_file, download_to_string,
};
use std::{fs, path::PathBuf};

/// Rust-native provider for the canonical THINGS catalog tables.
pub struct ThingsTablesProvider;

/// Rust-native provider for preferred THINGS HI velocity cubes.
///
/// Preference order per galaxy:
/// 1. RO_CUBE_THINGS.FITS
/// 2. NA_CUBE_THINGS.FITS
pub struct ThingsPreferredCubesProvider;

/// Parse the MPIA THINGS landing page into a manifest of FITS data products.
pub fn parse_things_cube_manifest_html(html: &str) -> Vec<ThingsCubeManifestEntry> {
    let mut entries = Vec::new();
    for href in extract_href_values(html) {
        if !href.ends_with(".FITS") {
            continue;
        }
        let filename = href.rsplit('/').next().unwrap_or(&href).to_string();
        let Some((galaxy_slug, weighting, product_kind)) =
            parse_manifest_entry_from_filename(&filename)
        else {
            continue;
        };
        let url = if href.starts_with("http://") || href.starts_with("https://") {
            href
        } else {
            format!("{THINGS_MPIA_ROOT}{href}")
        };
        entries.push(ThingsCubeManifestEntry {
            galaxy_key: things_name_key(&galaxy_slug),
            galaxy_slug,
            filename,
            weighting,
            product_kind,
            url,
        });
    }
    entries.sort_by(|left, right| {
        left.galaxy_slug
            .cmp(&right.galaxy_slug)
            .then_with(|| left.product_kind.cmp(&right.product_kind))
            .then_with(|| left.weighting.cmp(&right.weighting))
    });
    entries
}

/// Discover all THINGS cube/data-product links from the MPIA landing page.
pub fn discover_things_cube_manifest() -> Result<Vec<ThingsCubeManifestEntry>, FetchError> {
    let html = download_to_string(THINGS_MPIA_DATA_URL)?;
    let entries = parse_things_cube_manifest_html(&html);
    if entries.is_empty() {
        return Err(FetchError::Validation(
            "THINGS MPIA landing page exposed no FITS data products".to_string(),
        ));
    }
    Ok(entries)
}

impl DatasetProvider for ThingsTablesProvider {
    fn name(&self) -> &str {
        "THINGS Tables"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("things");
        fs::create_dir_all(&dir)?;
        for filename in ["table1.dat", "table4.dat", "refs.dat", "ReadMe"] {
            let destination = dir.join(filename);
            if config.skip_existing && destination.exists() {
                continue;
            }
            let url = format!("{THINGS_VIZIER_ROOT}/{filename}");
            download_to_file(&url, &destination)?;
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("things");
        dir.join("table1.dat").exists()
            && dir.join("table4.dat").exists()
            && dir.join("refs.dat").exists()
            && dir.join("ReadMe").exists()
    }
}

impl DatasetProvider for ThingsPreferredCubesProvider {
    fn name(&self) -> &str {
        "THINGS Preferred HI Cubes"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let base_dir = config.output_dir.join("things");
        let cube_dir = base_dir.join("cubes");
        fs::create_dir_all(&cube_dir)?;
        let manifest_entries = preferred_things_cube_entries(&discover_things_cube_manifest()?);
        if manifest_entries.is_empty() {
            return Err(FetchError::Validation(
                "No preferred THINGS cubes were discoverable".to_string(),
            ));
        }
        for entry in &manifest_entries {
            let destination = cube_dir.join(&entry.filename);
            if config.skip_existing && destination.exists() {
                continue;
            }
            download_to_file(&entry.url, &destination)?;
        }
        write_things_cube_manifest_csv(
            &base_dir.join("things_cube_manifest.csv"),
            &manifest_entries,
        )?;
        Ok(cube_dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let manifest = config
            .output_dir
            .join("things")
            .join("things_cube_manifest.csv");
        let cube_dir = config.output_dir.join("things").join("cubes");
        manifest.exists() && cube_dir.exists()
    }
}
