// SPDX-License-Identifier: MIT
#![cfg(feature = "euclid-catalog")]

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use cosmology_core::euclid_morphology::read_euclid_physical_measurements_audited;
use parquet::arrow::ArrowWriter;
use std::{
    fs::{self, File},
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

struct CatalogFixture(PathBuf);
impl CatalogFixture {
    fn write(batch: &RecordBatch) -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "euclid-admission-{}-{}.parquet",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let file = File::create_new(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        Self(path)
    }
}
impl Drop for CatalogFixture {
    fn drop(&mut self) {
        fs::remove_file(&self.0).unwrap();
    }
}

fn batch(identities: Vec<Option<i64>>, invalid: bool) -> RecordBatch {
    let rows = identities.len();
    let mut columns: Vec<(&str, ArrayRef)> =
        vec![("object_id", Arc::new(Int64Array::from(identities)))];
    for name in [
        "right_ascension",
        "declination",
        "sersic_sersic_vis_index",
        "semimajor_axis",
        "position_angle",
        "phz_pp_median_redshift",
        "phz_pp_median_stellarmass",
        "phz_pp_median_luminosity",
        "phz_pp_median_sfr",
    ] {
        let mut values = vec![Some(1.0); rows];
        if invalid {
            match name {
                "phz_pp_median_stellarmass" => values[1] = None,
                "phz_pp_median_redshift" => values[2] = Some(f64::INFINITY),
                "position_angle" => values[3] = Some(f64::NAN),
                "sersic_sersic_vis_index" => {
                    values[4] = Some(-1.0);
                    values[7] = None;
                }
                "semimajor_axis" => values[8] = Some(-1.0),
                "phz_pp_median_sfr" | "right_ascension" | "declination" => values[6] = None,
                _ => {}
            }
        }
        columns.push((name, Arc::new(Float64Array::from(values))));
    }
    RecordBatch::try_from_iter(columns).unwrap()
}

#[test]
fn model_nulls_and_nonfinite_inputs_preserve_rejection_denominator() {
    let mut identities: Vec<_> = (10..19).map(Some).collect();
    identities[5] = None;
    let fixture = CatalogFixture::write(&batch(identities, true));
    let catalog = read_euclid_physical_measurements_audited(fixture.0.to_str().unwrap()).unwrap();
    assert_eq!(catalog.rows_read, 9);
    assert_eq!(
        catalog
            .entries
            .iter()
            .map(|entry| entry.object_id)
            .collect::<Vec<_>>(),
        [10, 16]
    );
    assert_eq!(
        catalog
            .rejected_rows
            .iter()
            .map(|row| row.source_row)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 7, 8]
    );
    assert_eq!(catalog.rejected_rows[0].reason, "null_model_input");
    assert_eq!(catalog.rejected_rows[1].reason, "nonfinite_model_input");
    assert_eq!(catalog.rejected_rows[4].object_id, None);
    assert!(catalog.entries[1].log_sfr.is_nan());
    assert!(catalog.entries[1].ra_deg.is_nan());
    assert_eq!(
        catalog.entries.len() + catalog.rejected_rows.len(),
        catalog.rows_read
    );
}

#[test]
fn duplicate_admitted_identity_blocks_ambiguous_pairing() {
    let fixture = CatalogFixture::write(&batch(vec![Some(10), Some(10)], false));
    let error = read_euclid_physical_measurements_audited(fixture.0.to_str().unwrap())
        .err()
        .unwrap();
    assert!(error.contains("duplicate admitted object_id 10 at source row 1"));
}

#[test]
#[ignore = "requires explicit retained catalog input and audit output directory"]
fn retained_catalog_admission() {
    let input = std::env::var("EUCLID_ADMISSION_INPUT").unwrap();
    let output = PathBuf::from(std::env::var("EUCLID_ADMISSION_OUTPUT").unwrap());
    let catalog = read_euclid_physical_measurements_audited(&input).unwrap();
    assert_eq!(
        catalog.entries.len() + catalog.rejected_rows.len(),
        catalog.rows_read
    );
    fs::create_dir_all(&output).unwrap();
    let mut rejected = File::create(output.join("euclid-catalog-rejected-rows.csv")).unwrap();
    writeln!(rejected, "source_row,object_id,reason,field").unwrap();
    for row in &catalog.rejected_rows {
        writeln!(
            rejected,
            "{},{},{},{}",
            row.source_row,
            row.object_id
                .map(|identity| identity.to_string())
                .unwrap_or_default(),
            row.reason,
            row.field
        )
        .unwrap();
    }
    let mut summary = File::create(output.join("euclid-catalog-admission.toml")).unwrap();
    writeln!(summary, "rows_read = {}\nrows_admitted = {}\nrows_rejected = {}\nsource_row_indexing = \"zero_based_physical_parquet_order\"", catalog.rows_read, catalog.entries.len(), catalog.rejected_rows.len()).unwrap();
    println!(
        "rows_read={} rows_admitted={} rows_rejected={}",
        catalog.rows_read,
        catalog.entries.len(),
        catalog.rejected_rows.len()
    );
}
