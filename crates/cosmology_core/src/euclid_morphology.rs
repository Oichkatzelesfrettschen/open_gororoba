//! Euclid Q1 physical measurements catalog reader (Apache Arrow parquet).
//!
//! Reads Sersic fit parameters, photo-z, and stellar mass from the Euclid Q1
//! `useful_physical_measurements.parquet` (Zenodo record 15106473).
//! Gated behind the `euclid-catalog` feature flag.
//!
//! # Data path
//!
//! Expected: `data/external/euclid/zenodo/15106473/useful_physical_measurements.parquet`
//! Download:
//! `cargo run -p gororoba_cli_data --bin euclid-fetch -- zenodo-download --catalog morphology`
//!
//! # Verified column names (2026-03-11)
//!
//! | Parquet column              | Struct field        | Type    |
//! |-----------------------------|---------------------|---------|
//! | `object_id`                 | object_id           | int64   |
//! | `right_ascension`           | ra_deg              | float64 |
//! | `declination`               | dec_deg             | float64 |
//! | `sersic_sersic_vis_index`   | n                   | float64 |
//! | `semimajor_axis`            | r_e_arcsec          | float64 |
//! | `position_angle`            | pa_deg              | float64 |
//! | `phz_pp_median_redshift`    | photo_z             | float64 |
//! | `phz_pp_median_stellarmass` | log_stellar_mass    | float64 |
//! | `phz_pp_median_luminosity`  | log_luminosity      | float64 |
//! | `phz_pp_median_sfr`         | log_sfr             | float64 |
//!
//! Ellipticity is NOT in this file; it lives in `morphology_catalogue.parquet`.
//! The reader sets ellipticity = 0 (circular approximation) by default.

use crate::distances;

/// Euclid VIS instrument plate scale: 0.1 arcsec per pixel.
///
/// The `semimajor_axis` column in the Q1 useful_physical_measurements catalog
/// is in pixel coordinates. Multiply by this constant to convert to arcseconds.
///
/// Reference: Euclid Collaboration, Cropper et al. (2024), A&A 697, A2.
pub const EUCLID_VIS_PLATE_SCALE: f64 = 0.1;

/// Physical measurements extracted from Euclid Q1 catalog.
#[derive(Debug, Clone)]
pub struct EuclidSersicParams {
    pub object_id: i64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Sersic index (VIS band).
    pub n: f64,
    /// Effective (half-light) radius in arcsec (semimajor_axis).
    pub r_e_arcsec: f64,
    /// Ellipticity e = 1 - b/a. Default 0.0 (circular approx).
    pub ellipticity: f64,
    /// Position angle in degrees.
    pub pa_deg: f64,
    /// Photometric redshift.
    pub photo_z: f64,
    /// log10(stellar mass / M_sun).
    pub log_stellar_mass: f64,
    /// log10(luminosity) (rest-frame, solar units).
    pub log_luminosity: f64,
    /// log10(star formation rate / M_sun yr^-1).
    pub log_sfr: f64,
}

/// Euclid Q1 visual morphology row from `morphology_catalogue.parquet`.
#[derive(Debug, Clone)]
pub struct EuclidMorphologyRecord {
    pub object_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub ellipticity: f64,
    pub featured_fraction: f32,
    pub spiral_fraction: f32,
    pub face_on_fraction: f32,
    pub non_merging_fraction: f32,
    pub bar_no_fraction: f32,
    pub clumps_yes_fraction: f32,
}

impl EuclidSersicParams {
    /// Convert angular effective radius to physical kpc at the object's photo-z.
    pub fn r_e_kpc(&self) -> f64 {
        arcsec_to_kpc(self.r_e_arcsec, self.photo_z)
    }

    /// Estimate halo virial mass M_200 (solar masses) from stellar mass
    /// via Moster+2013 SMHM relation at z ~ 0.
    ///
    /// Inverts M*(M_h) = 2 * N * M_h / ((M_h/M_1)^(-beta) + (M_h/M_1)^gamma)
    /// using bisection. Returns M_200 in solar masses (linear, not log).
    pub fn m200_solar(&self) -> f64 {
        let m_star = 10.0_f64.powf(self.log_stellar_mass);
        smhm_invert_moster2013(m_star)
    }

    /// Estimate surface brightness I_e from luminosity and effective radius.
    ///
    /// I_e = L / (2 * pi * R_e^2) in the circular approximation.
    /// Units: solar luminosity per kpc^2.
    pub fn i_e_solar_per_kpc2(&self) -> f64 {
        let lum = 10.0_f64.powf(self.log_luminosity);
        let r_e = self.r_e_kpc();
        if r_e > 0.0 {
            lum / (2.0 * std::f64::consts::PI * r_e * r_e)
        } else {
            0.0
        }
    }

    /// Mass-to-light ratio M*/L.
    pub fn ml_ratio(&self) -> f64 {
        let m_star = 10.0_f64.powf(self.log_stellar_mass);
        let lum = 10.0_f64.powf(self.log_luminosity);
        if lum > 0.0 { m_star / lum } else { 1.0 }
    }

    /// Morphological mass-to-light ratio based on Sersic index.
    ///
    /// The `log_luminosity` column in Euclid Q1 produces M/L ~ 8.4e9
    /// (nonsensical), so this provides a physically motivated workaround.
    /// n < 2 implies disk (late-type): M/L ~ 2.0 (Bell et al. 2003).
    /// n >= 2 implies elliptical (early-type): M/L ~ 5.0 (into et al. 2013).
    ///
    /// The original `ml_ratio()` is preserved for future audit when the
    /// column convention is resolved.
    pub fn ml_ratio_morphological(&self) -> f64 {
        if self.n < 2.0 { 2.0 } else { 5.0 }
    }

    /// Broad morphological type from Sersic index.
    ///
    /// n < 2: "disk" (exponential-dominated, late-type).
    /// n >= 2: "elliptical" (de Vaucouleurs-like, early-type).
    pub fn morphological_type(&self) -> &'static str {
        if self.n < 2.0 { "disk" } else { "elliptical" }
    }
}

/// Convert angular effective radius (arcsec) to physical (kpc) at redshift z.
///
/// Uses flat Lambda-CDM angular diameter distance with Planck 2018 parameters.
pub fn arcsec_to_kpc(arcsec: f64, z: f64) -> f64 {
    if z <= 0.0 || arcsec <= 0.0 {
        return 0.0;
    }
    let d_a_mpc = distances::angular_diameter_distance(z, 0.3111, 67.66);
    let kpc_per_arcsec = d_a_mpc * 1000.0 * std::f64::consts::PI / (180.0 * 3600.0);
    arcsec * kpc_per_arcsec
}

/// Moster+2013 stellar-to-halo mass relation parameters (z=0).
const MOSTER_LOG_M1: f64 = 11.59;
const MOSTER_N: f64 = 0.0351;
const MOSTER_BETA: f64 = 1.376;
const MOSTER_GAMMA: f64 = 0.608;

/// Forward SMHM: compute M*(M_h) from Moster+2013.
fn smhm_moster2013(m_halo: f64) -> f64 {
    let m1 = 10.0_f64.powf(MOSTER_LOG_M1);
    let ratio = m_halo / m1;
    2.0 * MOSTER_N * m_halo / (ratio.powf(-MOSTER_BETA) + ratio.powf(MOSTER_GAMMA))
}

/// Invert SMHM via bisection: given M*, find M_halo.
fn smhm_invert_moster2013(m_star: f64) -> f64 {
    if m_star <= 0.0 {
        return 0.0;
    }
    // Bisection in log space
    let mut lo = 9.0_f64; // log10(M_h) lower bound
    let mut hi = 16.0_f64; // log10(M_h) upper bound
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let m_h = 10.0_f64.powf(mid);
        let m_s_pred = smhm_moster2013(m_h);
        if m_s_pred < m_star {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    10.0_f64.powf(0.5 * (lo + hi))
}

/// Read physical measurements from Euclid Q1 parquet catalog.
///
/// Uses Apache Arrow `parquet` crate for pure-Rust zero-copy columnar reads.
/// Filters: n > 0, r_e > 0, photo_z > 0 (valid Sersic fits with redshift).
///
/// Returns sorted by object_id for deterministic output.
#[cfg(feature = "euclid-catalog")]
pub fn read_euclid_physical_measurements(
    parquet_path: &str,
) -> Result<Vec<EuclidSersicParams>, String> {
    use arrow_array::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::{fs::File, path::Path};

    if !Path::new(parquet_path).exists() {
        return Err(format!(
            "Parquet file not found: {parquet_path}\n\
             Download: cargo run -p gororoba_cli_data --bin euclid-fetch -- zenodo-download --catalog morphology"
        ));
    }

    let file = File::open(parquet_path).map_err(|e| format!("Failed to open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to read parquet metadata: {e}"))?;

    // Print schema for diagnostics
    let schema = builder.schema();
    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Verify required columns exist
    let required = [
        "object_id",
        "right_ascension",
        "declination",
        "sersic_sersic_vis_index",
        "semimajor_axis",
        "position_angle",
        "phz_pp_median_redshift",
        "phz_pp_median_stellarmass",
        "phz_pp_median_luminosity",
        "phz_pp_median_sfr",
    ];
    for &col in &required {
        if !col_names.contains(&col) {
            return Err(format!(
                "Required column '{col}' not found.\n\
                 Available columns: {col_names:?}\n\
                 Is this useful_physical_measurements.parquet?"
            ));
        }
    }

    let reader = builder
        .build()
        .map_err(|e| format!("Failed to build reader: {e}"))?;

    let mut result = Vec::new();

    for batch_result in reader {
        let batch: RecordBatch = batch_result.map_err(|e| format!("Failed to read batch: {e}"))?;

        let object_id = col_i64(&batch, "object_id")?;
        let ra = col_f64(&batch, "right_ascension")?;
        let dec = col_f64(&batch, "declination")?;
        let sersic_n = col_f64(&batch, "sersic_sersic_vis_index")?;
        let semi_a = col_f64(&batch, "semimajor_axis")?;
        let pa = col_f64(&batch, "position_angle")?;
        let photo_z = col_f64(&batch, "phz_pp_median_redshift")?;
        let stellar_mass = col_f64(&batch, "phz_pp_median_stellarmass")?;
        let luminosity = col_f64(&batch, "phz_pp_median_luminosity")?;
        let sfr = col_f64(&batch, "phz_pp_median_sfr")?;

        for i in 0..batch.num_rows() {
            let n = sersic_n.value(i);
            let r_e = semi_a.value(i);
            let z = photo_z.value(i);

            // Filter: valid Sersic fit with positive radius and redshift
            if n <= 0.0 || r_e <= 0.0 || z <= 0.0 || !n.is_finite() || !r_e.is_finite() {
                continue;
            }

            // semimajor_axis is in VIS pixels; convert to arcsec
            let r_e_arcsec = r_e * EUCLID_VIS_PLATE_SCALE;

            result.push(EuclidSersicParams {
                object_id: object_id.value(i),
                ra_deg: ra.value(i),
                dec_deg: dec.value(i),
                n,
                r_e_arcsec,
                ellipticity: 0.0, // circular approximation; not in this file
                pa_deg: pa.value(i),
                photo_z: z,
                log_stellar_mass: stellar_mass.value(i),
                log_luminosity: luminosity.value(i),
                log_sfr: sfr.value(i),
            });
        }
    }

    // Sort by object_id for deterministic output
    result.sort_by_key(|p| p.object_id);
    Ok(result)
}

/// Read Euclid Q1 visual morphology classifications from the Zenodo parquet.
#[cfg(feature = "euclid-catalog")]
pub fn read_euclid_visual_morphology(
    parquet_path: &str,
) -> Result<Vec<EuclidMorphologyRecord>, String> {
    use arrow_array::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::{fs::File, path::Path};

    if !Path::new(parquet_path).exists() {
        return Err(format!(
            "Parquet file not found: {parquet_path}\n\
             Download: cargo run -p gororoba_cli_data --bin euclid-fetch -- zenodo-download --catalog morphology"
        ));
    }

    let file = File::open(parquet_path).map_err(|e| format!("Failed to open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to read parquet metadata: {e}"))?;
    let schema = builder.schema();
    let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let required = [
        "object_id",
        "ra",
        "dec",
        "ellipticity",
        "smooth-or-featured_featured-or-disk_fraction",
        "has-spiral-arms_yes_fraction",
        "disk-edge-on_no_fraction",
        "merging_none_fraction",
        "bar_no_fraction",
        "clumps_yes_fraction",
    ];
    for &col in &required {
        if !col_names.contains(&col) {
            return Err(format!(
                "Required column '{col}' not found.\n\
                 Available columns: {col_names:?}\n\
                 Is this morphology_catalogue.parquet?"
            ));
        }
    }

    let reader = builder
        .build()
        .map_err(|e| format!("Failed to build reader: {e}"))?;
    let mut result = Vec::new();

    for batch_result in reader {
        let batch: RecordBatch = batch_result.map_err(|e| format!("Failed to read batch: {e}"))?;
        let object_id = col_string(&batch, "object_id")?;
        let ra = col_f64(&batch, "ra")?;
        let dec = col_f64(&batch, "dec")?;
        let ellipticity = col_f64(&batch, "ellipticity")?;
        let featured = col_f32(&batch, "smooth-or-featured_featured-or-disk_fraction")?;
        let spiral = col_f32(&batch, "has-spiral-arms_yes_fraction")?;
        let face_on = col_f32(&batch, "disk-edge-on_no_fraction")?;
        let non_merging = col_f32(&batch, "merging_none_fraction")?;
        let bar_no = col_f32(&batch, "bar_no_fraction")?;
        let clumps_yes = col_f32(&batch, "clumps_yes_fraction")?;

        for i in 0..batch.num_rows() {
            let ra_deg = ra.value(i);
            let dec_deg = dec.value(i);
            if !ra_deg.is_finite() || !dec_deg.is_finite() {
                continue;
            }
            result.push(EuclidMorphologyRecord {
                object_id: object_id.value(i).to_string(),
                ra_deg,
                dec_deg,
                ellipticity: ellipticity.value(i),
                featured_fraction: featured.value(i),
                spiral_fraction: spiral.value(i),
                face_on_fraction: face_on.value(i),
                non_merging_fraction: non_merging.value(i),
                bar_no_fraction: bar_no.value(i),
                clumps_yes_fraction: clumps_yes.value(i),
            });
        }
    }

    result.sort_by(|a, b| {
        a.object_id
            .cmp(&b.object_id)
            .then_with(|| a.ra_deg.partial_cmp(&b.ra_deg).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(result)
}

#[cfg(feature = "euclid-catalog")]
fn col_f64<'a>(
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::Float64Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("Column '{name}' not in batch"))?
        .as_any()
        .downcast_ref::<arrow_array::Float64Array>()
        .ok_or_else(|| format!("Column '{name}' is not Float64"))
}

#[cfg(feature = "euclid-catalog")]
fn col_i64<'a>(
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::Int64Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("Column '{name}' not in batch"))?
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .ok_or_else(|| format!("Column '{name}' is not Int64"))
}

#[cfg(feature = "euclid-catalog")]
fn col_f32<'a>(
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::Float32Array, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("Column '{name}' not in batch"))?
        .as_any()
        .downcast_ref::<arrow_array::Float32Array>()
        .ok_or_else(|| format!("Column '{name}' is not Float32"))
}

#[cfg(feature = "euclid-catalog")]
fn col_string<'a>(
    batch: &'a arrow_array::RecordBatch,
    name: &str,
) -> Result<&'a arrow_array::StringArray, String> {
    batch
        .column_by_name(name)
        .ok_or_else(|| format!("Column '{name}' not in batch"))?
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .ok_or_else(|| format!("Column '{name}' is not Utf8"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_arcsec_to_kpc_z03() {
        // At z=0.3, d_A ~ 930 Mpc -> 1" ~ 4.5 kpc
        let kpc = arcsec_to_kpc(1.0, 0.3);
        assert!(kpc > 3.0 && kpc < 6.0, "1 arcsec at z=0.3 = {kpc} kpc");
    }

    #[test]
    fn test_arcsec_to_kpc_edge_cases() {
        assert_eq!(arcsec_to_kpc(0.0, 0.3), 0.0);
        assert_eq!(arcsec_to_kpc(1.0, 0.0), 0.0);
        assert_eq!(arcsec_to_kpc(1.0, -1.0), 0.0);
    }

    #[test]
    fn test_arcsec_to_kpc_scales_linearly() {
        let k1 = arcsec_to_kpc(1.0, 0.5);
        let k2 = arcsec_to_kpc(2.0, 0.5);
        assert_relative_eq!(k2, 2.0 * k1, epsilon = 1e-10);
    }

    #[test]
    fn test_smhm_moster2013_milky_way() {
        // MW: M* ~ 5e10 Msun, M_halo ~ 1e12 Msun
        let m_h = smhm_invert_moster2013(5e10);
        let log_m_h = m_h.log10();
        // Should be in range 11.5 < log10(M_h) < 12.5
        assert!(
            log_m_h > 11.5 && log_m_h < 12.5,
            "MW-mass SMHM: log10(M_h)={log_m_h:.2}, expected ~12"
        );
    }

    #[test]
    fn test_smhm_moster2013_dwarf() {
        // Dwarf: M* ~ 1e8 Msun -> M_halo ~ 1e11 Msun
        let m_h = smhm_invert_moster2013(1e8);
        let log_m_h = m_h.log10();
        assert!(
            log_m_h > 10.0 && log_m_h < 12.0,
            "Dwarf SMHM: log10(M_h)={log_m_h:.2}, expected ~10.5-11"
        );
    }

    #[test]
    fn test_smhm_roundtrip() {
        // Forward-inverse roundtrip
        let m_halo = 1e12;
        let m_star = smhm_moster2013(m_halo);
        let m_halo_recovered = smhm_invert_moster2013(m_star);
        assert_relative_eq!(m_halo_recovered.log10(), m_halo.log10(), epsilon = 0.01);
    }

    #[test]
    fn test_params_derived_quantities() {
        let p = EuclidSersicParams {
            object_id: 1,
            ra_deg: 0.0,
            dec_deg: 0.0,
            n: 4.0,
            r_e_arcsec: 1.0,
            ellipticity: 0.0,
            pa_deg: 0.0,
            photo_z: 0.3,
            log_stellar_mass: 10.5,
            log_luminosity: 10.0,
            log_sfr: 0.0,
        };
        // r_e_kpc should be positive at z=0.3
        assert!(p.r_e_kpc() > 3.0 && p.r_e_kpc() < 6.0);
        // m200 should be ~ 10^12
        let log_m200 = p.m200_solar().log10();
        assert!(log_m200 > 11.0 && log_m200 < 13.0);
        // i_e should be positive
        assert!(p.i_e_solar_per_kpc2() > 0.0);
        // ml_ratio for M*=10^10.5, L=10^10 -> ~3.16
        assert_relative_eq!(p.ml_ratio(), 10.0_f64.powf(0.5), epsilon = 0.01);
    }

    #[test]
    fn test_ml_ratio_morphological_disk() {
        let p = EuclidSersicParams {
            object_id: 1,
            ra_deg: 0.0,
            dec_deg: 0.0,
            n: 1.0,
            r_e_arcsec: 1.0,
            ellipticity: 0.0,
            pa_deg: 0.0,
            photo_z: 0.3,
            log_stellar_mass: 10.0,
            log_luminosity: 10.0,
            log_sfr: 0.0,
        };
        assert_relative_eq!(p.ml_ratio_morphological(), 2.0, epsilon = 1e-10);
        assert_eq!(p.morphological_type(), "disk");
    }

    #[test]
    fn test_ml_ratio_morphological_elliptical() {
        let p = EuclidSersicParams {
            object_id: 2,
            ra_deg: 0.0,
            dec_deg: 0.0,
            n: 4.0,
            r_e_arcsec: 1.0,
            ellipticity: 0.0,
            pa_deg: 0.0,
            photo_z: 0.3,
            log_stellar_mass: 10.0,
            log_luminosity: 10.0,
            log_sfr: 0.0,
        };
        assert_relative_eq!(p.ml_ratio_morphological(), 5.0, epsilon = 1e-10);
        assert_eq!(p.morphological_type(), "elliptical");
    }

    #[test]
    fn test_ml_ratio_morphological_boundary() {
        // n=2.0 should be classified as elliptical
        let p = EuclidSersicParams {
            object_id: 3,
            ra_deg: 0.0,
            dec_deg: 0.0,
            n: 2.0,
            r_e_arcsec: 1.0,
            ellipticity: 0.0,
            pa_deg: 0.0,
            photo_z: 0.3,
            log_stellar_mass: 10.0,
            log_luminosity: 10.0,
            log_sfr: 0.0,
        };
        assert_relative_eq!(p.ml_ratio_morphological(), 5.0, epsilon = 1e-10);
        assert_eq!(p.morphological_type(), "elliptical");
    }
}
