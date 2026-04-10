//! Fetch implementation for desi_bao. See desi_bao.rs for record types and parsers.

use super::desi_bao::{desi_dr1_bao, desi_dr2_bao, BaoMeasurement};
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use std::path::{Path, PathBuf};

/// DESI BAO text files from the Cobaya sampler GitHub repository.
const DESI_BAO_BASE: &str = "https://raw.githubusercontent.com/CobayaSampler/bao_data/master/";

/// DESI BAO dataset provider (fetches individual text files).
pub struct DesiBaoProvider;

impl DatasetProvider for DesiBaoProvider {
    fn name(&self) -> &str {
        "DESI DR1 BAO"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("desi_bao");
        std::fs::create_dir_all(&dir)?;

        let files = [
            "desi_2024_gaussian_bao_BGS_z0.295.txt",
            "desi_2024_gaussian_bao_LRG1_z0.510.txt",
            "desi_2024_gaussian_bao_LRG2_z0.706.txt",
            "desi_2024_gaussian_bao_LRG3+ELG1_z0.930.txt",
            "desi_2024_gaussian_bao_ELG2_z1.317.txt",
            "desi_2024_gaussian_bao_QSO_z1.491.txt",
            "desi_2024_gaussian_bao_Lya_z2.330.txt",
        ];

        let mut downloaded = 0usize;
        for fname in &files {
            let output = dir.join(fname);
            if config.skip_existing && output.exists() {
                downloaded += 1;
                continue;
            }
            let url = format!("{}{}", DESI_BAO_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    downloaded += 1;
                    log::info!("Saved {}", fname);
                }
                Err(e) => {
                    log::warn!("Failed to download {}: {}", fname, e);
                }
            }
        }

        write_bao_csv(&dir.join("desi_dr1_bao_summary.csv"), &desi_dr1_bao())?;
        write_bao_csv(&dir.join("desi_dr2_bao_summary.csv"), &desi_dr2_bao())?;
        std::fs::write(
            dir.join("README.txt"),
            format!(
                "DESI BAO summary lane.\nRemote Cobaya mirror files staged: {downloaded}/{}.\nBounded summary CSVs for DR1 and DR2 are always emitted from the exact in-repo table values.\n",
                files.len()
            ),
        )?;

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("desi_bao").exists()
    }
}

fn write_bao_csv(path: &Path, rows: &[BaoMeasurement]) -> Result<(), FetchError> {
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|err| FetchError::Validation(format!("create CSV {}: {err}", path.display())))?;
    for row in rows {
        writer.serialize(row).map_err(|err| {
            FetchError::Validation(format!("write CSV {}: {err}", path.display()))
        })?;
    }
    writer.flush()?;
    Ok(())
}
