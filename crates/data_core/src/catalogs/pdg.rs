use crate::fetcher::FetchError;
use csv::ReaderBuilder;
use std::path::Path;

/// A distilled PDG mass reference row backed by locally cached PDG surfaces.
#[derive(Debug, Clone, PartialEq)]
pub struct PdgMassEntry {
    pub particle: String,
    pub mass_gev: f64,
    pub uncertainty_gev: f64,
    pub source_file: String,
    pub source_page: usize,
}

/// Parse a distilled PDG mass reference CSV.
///
/// Expected columns:
/// `particle,mass_gev,uncertainty_gev,source_file,source_page`
pub fn parse_pdg_mass_reference_csv(path: &Path) -> Result<Vec<PdgMassEntry>, FetchError> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|err| FetchError::Validation(format!("open {}: {err}", path.display())))?;

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result
            .map_err(|err| FetchError::Validation(format!("parse {}: {err}", path.display())))?;
        if record.len() < 5 {
            return Err(FetchError::Validation(format!(
                "expected 5 columns in {}, got {}",
                path.display(),
                record.len()
            )));
        }

        let particle = record[0].trim().to_string();
        let mass_gev = record[1].trim().parse::<f64>().map_err(|err| {
            FetchError::Validation(format!(
                "invalid mass_gev '{}' in {}: {err}",
                record[1].trim(),
                path.display()
            ))
        })?;
        let uncertainty_gev = record[2].trim().parse::<f64>().map_err(|err| {
            FetchError::Validation(format!(
                "invalid uncertainty_gev '{}' in {}: {err}",
                record[2].trim(),
                path.display()
            ))
        })?;
        let source_file = record[3].trim().to_string();
        let source_page = record[4].trim().parse::<usize>().map_err(|err| {
            FetchError::Validation(format!(
                "invalid source_page '{}' in {}: {err}",
                record[4].trim(),
                path.display()
            ))
        })?;

        rows.push(PdgMassEntry {
            particle,
            mass_gev,
            uncertainty_gev,
            source_file,
            source_page,
        });
    }

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parse_pdg_mass_reference_csv_reads_expected_rows() {
        let temp = std::env::temp_dir().join("pdg_mass_reference_test.csv");
        fs::write(
            &temp,
            "particle,mass_gev,uncertainty_gev,source_file,source_page\n\
             electron,5.1099895000e-4,1.5e-13,rpp2025-sum-leptons.pdf,1\n\
             W_boson,80.3692,0.0133,rpp2025-sum-gauge-higgs-bosons.pdf,1\n",
        )
        .unwrap();

        let rows = parse_pdg_mass_reference_csv(&temp).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].particle, "electron");
        assert!((rows[1].mass_gev - 80.3692).abs() < 1.0e-12);

        fs::remove_file(temp).ok();
    }
}
