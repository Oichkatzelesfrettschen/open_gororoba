//! Typed intake boundary for condition-bound gold optical datasets.
//!
//! The catalog records source candidates, not admitted measurements. Selection
//! requires dataset, state, and specimen identities; a chemical formula cannot
//! resolve among condition-dependent optical records.

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

use serde::Deserialize;

use crate::material_records::RecordId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldOpticalSelection {
    pub dataset_id: RecordId,
    pub state_id: RecordId,
    pub specimen_id: RecordId,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GoldOpticalCandidateCatalog {
    pub schema_version: u32,
    pub material_id: String,
    pub formula: String,
    pub database_id: String,
    pub database_commit: String,
    pub database_license: String,
    pub canonical_owner: String,
    pub admission_scope: String,
    pub dataset: Vec<GoldOpticalCandidate>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GoldOpticalCandidate {
    pub dataset_id: String,
    pub state_id: String,
    pub specimen_id: String,
    pub preferred_name: String,
    pub phase: String,
    pub temperature_label: Option<String>,
    pub specimen_preparation: Option<String>,
    pub geometry: Option<String>,
    pub measurement_method: Option<String>,
    pub measurement_uncertainty: Option<String>,
    pub underlying_doi: String,
    pub source_id: String,
    pub source_path: String,
    pub source_sha256: String,
    pub source_locator: String,
    pub wavelength_um: f64,
    pub n: f64,
    pub k: f64,
    pub direct_admission_ready: bool,
    pub admission_residual: String,
    pub missing_fields: Vec<String>,
}

impl GoldOpticalCandidateCatalog {
    pub fn load() -> Result<Self, String> {
        let catalog: Self =
            toml::from_str(materials_data::GOLD_OPTICAL_SPECIMEN_CANDIDATES_TOML)
                .map_err(|error| format!("parse gold optical candidate catalog: {error}"))?;
        catalog.validate()?;
        catalog.validate_retained_sources(&repo_root::resolve!())?;
        Ok(catalog)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported gold optical candidate schema {}",
                self.schema_version
            ));
        }
        require_nonempty("material_id", &self.material_id)?;
        require_nonempty("formula", &self.formula)?;
        require_nonempty("database_id", &self.database_id)?;
        require_nonempty("database_commit", &self.database_commit)?;
        require_nonempty("database_license", &self.database_license)?;
        require_nonempty("canonical_owner", &self.canonical_owner)?;
        require_nonempty("admission_scope", &self.admission_scope)?;
        if self.dataset.is_empty() {
            return Err("gold optical candidate catalog must contain datasets".to_owned());
        }

        let mut dataset_ids = BTreeSet::new();
        let mut state_ids = BTreeSet::new();
        let mut specimen_ids = BTreeSet::new();
        for candidate in &self.dataset {
            candidate.validate()?;
            insert_unique(&mut dataset_ids, "dataset", &candidate.dataset_id)?;
            insert_unique(&mut state_ids, "state", &candidate.state_id)?;
            insert_unique(&mut specimen_ids, "specimen", &candidate.specimen_id)?;
        }
        Ok(())
    }

    pub fn validate_retained_sources(&self, repository_root: &Path) -> Result<(), String> {
        for candidate in &self.dataset {
            candidate.validate_retained_source(repository_root)?;
        }
        Ok(())
    }

    pub fn select(
        &self,
        selection: &GoldOpticalSelection,
    ) -> Result<&GoldOpticalCandidate, String> {
        self.dataset
            .iter()
            .find(|candidate| {
                candidate.dataset_id == selection.dataset_id.0
                    && candidate.state_id == selection.state_id.0
                    && candidate.specimen_id == selection.specimen_id.0
            })
            .ok_or_else(|| {
                format!(
                    "no gold optical candidate matches dataset {}, state {}, specimen {}",
                    selection.dataset_id.0, selection.state_id.0, selection.specimen_id.0
                )
            })
    }
}

impl GoldOpticalCandidate {
    fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("dataset_id", self.dataset_id.as_str()),
            ("state_id", self.state_id.as_str()),
            ("specimen_id", self.specimen_id.as_str()),
            ("preferred_name", self.preferred_name.as_str()),
            ("phase", self.phase.as_str()),
            ("underlying_doi", self.underlying_doi.as_str()),
            ("source_id", self.source_id.as_str()),
            ("source_path", self.source_path.as_str()),
            ("source_locator", self.source_locator.as_str()),
            ("admission_residual", self.admission_residual.as_str()),
        ] {
            require_nonempty(label, value)?;
        }
        for identifier in [&self.dataset_id, &self.state_id, &self.specimen_id] {
            RecordId::new(identifier.clone())?;
        }
        if self.source_sha256.len() != 64
            || !self
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!(
                "candidate {} has invalid source_sha256",
                self.dataset_id
            ));
        }
        if !self.wavelength_um.is_finite() || self.wavelength_um <= 0.0 {
            return Err(format!(
                "candidate {} requires positive finite wavelength",
                self.dataset_id
            ));
        }
        if !self.n.is_finite() || self.n < 0.0 || !self.k.is_finite() || self.k < 0.0 {
            return Err(format!(
                "candidate {} requires finite nonnegative n and k",
                self.dataset_id
            ));
        }
        let unique_missing: BTreeSet<_> = self.missing_fields.iter().collect();
        if unique_missing.len() != self.missing_fields.len()
            || self
                .missing_fields
                .iter()
                .any(|field| field.trim().is_empty())
        {
            return Err(format!(
                "candidate {} has empty or duplicate missing fields",
                self.dataset_id
            ));
        }
        if self.direct_admission_ready && !self.missing_fields.is_empty() {
            return Err(format!(
                "candidate {} claims direct admission while required fields remain missing",
                self.dataset_id
            ));
        }
        Ok(())
    }

    fn validate_retained_source(&self, repository_root: &Path) -> Result<(), String> {
        let relative_path = Path::new(&self.source_path);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!(
                "candidate {} source path must be repository-relative without traversal",
                self.dataset_id
            ));
        }
        let canonical_repository_root = repository_root.canonicalize().map_err(|error| {
            format!(
                "resolve repository root {}: {error}",
                repository_root.display()
            )
        })?;
        let source_path = repository_root.join(relative_path);
        let canonical_source_path = source_path.canonicalize().map_err(|error| {
            format!("resolve retained source {}: {error}", source_path.display())
        })?;
        canonical_source_path
            .strip_prefix(&canonical_repository_root)
            .map_err(|_| {
                format!(
                    "candidate {} source path resolves outside the repository",
                    self.dataset_id
                )
            })?;
        let source_bytes = fs::read(&canonical_source_path).map_err(|error| {
            format!(
                "read retained source {}: {error}",
                canonical_source_path.display()
            )
        })?;
        let observed_sha256 = sha256_hex(&source_bytes);
        if observed_sha256 != self.source_sha256 {
            return Err(format!(
                "candidate {} source SHA-256 mismatch: expected {}, observed {}",
                self.dataset_id, self.source_sha256, observed_sha256
            ));
        }
        let source_text = std::str::from_utf8(&source_bytes).map_err(|error| {
            format!(
                "retained source {} is not UTF-8: {error}",
                canonical_source_path.display()
            )
        })?;
        let matching_rows: Vec<_> = source_text
            .lines()
            .enumerate()
            .filter_map(|(line_index, line)| {
                parse_nk_row(line).map(|row| (line_index + 1, line.trim(), row))
            })
            .filter(|(_, _, row)| row.0 == self.wavelength_um)
            .collect();
        if matching_rows.len() != 1 {
            return Err(format!(
                "candidate {} locator {} resolves to {} rows at wavelength {} um",
                self.dataset_id,
                self.source_locator,
                matching_rows.len(),
                self.wavelength_um
            ));
        }
        let (line_number, row_text, (_, observed_n, observed_k)) = matching_rows[0];
        let observed_locator = format!("line {line_number}: {row_text}");
        if self.source_locator != observed_locator {
            return Err(format!(
                "candidate {} source locator mismatch: expected {}, observed {}",
                self.dataset_id, self.source_locator, observed_locator
            ));
        }
        if observed_n != self.n || observed_k != self.k {
            return Err(format!(
                "candidate {} locator {} records n={}, k={}, expected n={}, k={}",
                self.dataset_id, self.source_locator, observed_n, observed_k, self.n, self.k
            ));
        }
        Ok(())
    }
}

fn parse_nk_row(line: &str) -> Option<(f64, f64, f64)> {
    let values: Vec<_> = line.split_ascii_whitespace().collect();
    if values.len() != 3 {
        return None;
    }
    Some((
        values[0].parse().ok()?,
        values[1].parse().ok()?,
        values[2].parse().ok()?,
    ))
}

fn sha256_hex(bytes: &[u8]) -> String {
    const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes(chunk[offset..offset + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let sigma0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let sigma1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(words[index - 7])
                .wrapping_add(sigma1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big_sigma1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(big_sigma1)
                .wrapping_add(choice)
                .wrapping_add(ROUND[index])
                .wrapping_add(words[index]);
            let big_sigma0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = big_sigma0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn require_nonempty(label: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be nonempty"))
    } else {
        Ok(())
    }
}

fn insert_unique<'a>(
    set: &mut BTreeSet<&'a str>,
    kind: &str,
    value: &'a str,
) -> Result<(), String> {
    if set.insert(value) {
        Ok(())
    } else {
        Err(format!("duplicate {kind} identifier {value}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_gold_candidates_are_valid_and_not_directly_admitted() {
        let catalog = GoldOpticalCandidateCatalog::load().unwrap();
        assert_eq!(catalog.material_id, "material:au");
        assert_eq!(catalog.formula, "Au");
        assert_eq!(catalog.dataset.len(), 3);
        assert!(
            catalog
                .dataset
                .iter()
                .all(|candidate| !candidate.direct_admission_ready)
        );
    }

    #[test]
    fn selection_requires_dataset_state_and_specimen_identity() {
        let catalog = GoldOpticalCandidateCatalog::load().unwrap();
        let candidate = &catalog.dataset[2];
        let exact = GoldOpticalSelection {
            dataset_id: RecordId::new(candidate.dataset_id.clone()).unwrap(),
            state_id: RecordId::new(candidate.state_id.clone()).unwrap(),
            specimen_id: RecordId::new(candidate.specimen_id.clone()).unwrap(),
        };
        assert_eq!(catalog.select(&exact).unwrap().n, 0.848_474_841);

        let formula_only_substitute = GoldOpticalSelection {
            dataset_id: RecordId::new("Au").unwrap(),
            state_id: RecordId::new("Au").unwrap(),
            specimen_id: RecordId::new("Au").unwrap(),
        };
        assert!(catalog.select(&formula_only_substitute).is_err());
    }

    #[test]
    fn admission_readiness_cannot_hide_missing_fields() {
        let mut catalog = GoldOpticalCandidateCatalog::load().unwrap();
        catalog.dataset[0].direct_admission_ready = true;
        assert!(catalog.validate().is_err());
    }

    #[test]
    fn retained_sources_bind_hashes_and_exact_nk_rows() {
        let catalog = GoldOpticalCandidateCatalog::load().unwrap();
        assert!(
            catalog
                .validate_retained_sources(&repo_root::resolve!())
                .is_ok()
        );
    }

    #[test]
    fn retained_source_mutations_are_rejected() {
        let mut catalog = GoldOpticalCandidateCatalog::load().unwrap();
        catalog.dataset[0].source_sha256 = "0".repeat(64);
        assert!(
            catalog
                .validate_retained_sources(&repo_root::resolve!())
                .is_err()
        );

        let mut catalog = GoldOpticalCandidateCatalog::load().unwrap();
        catalog.dataset[1].n += 0.01;
        assert!(
            catalog
                .validate_retained_sources(&repo_root::resolve!())
                .is_err()
        );

        let mut catalog = GoldOpticalCandidateCatalog::load().unwrap();
        catalog.dataset[1].source_locator = "tabulated nk row at 0.500 um".to_owned();
        assert!(
            catalog
                .validate_retained_sources(&repo_root::resolve!())
                .is_err()
        );

        let mut catalog = GoldOpticalCandidateCatalog::load().unwrap();
        catalog.dataset[2].source_path = "../outside.yml".to_owned();
        assert!(
            catalog
                .validate_retained_sources(&repo_root::resolve!())
                .is_err()
        );
    }

    #[test]
    fn sha256_matches_standard_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
