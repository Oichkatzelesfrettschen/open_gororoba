use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, TimeZone, Utc};
use data_core::{
    catalogs::{omni::OmniRecord, voyager::{VoyagerSpacecraft, parse_voyager_file, voyager_to_omni}},
    time_bounds::TimeBounds,
};
use std::path::Path;

use crate::voyager_arrow::TrajectoryFeeder;

#[derive(Clone, Debug)]
struct SpatialPoint {
    timestamp_ms: i64,
    r_au: f64,
    lat_deg: f64,
    lon_deg: f64,
}

#[derive(Clone, Debug)]
pub struct VoyagerSpatialSample {
    pub lower_index: usize,
    pub upper_index: usize,
    pub lower_time_ms: i64,
    pub upper_time_ms: i64,
    pub r_au: Option<f64>,
    pub lat_deg: Option<f64>,
    pub lon_deg: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct FusedEncounterState {
    pub timestamp_ms: i64,
    pub instrument_value: Option<f64>,
    pub r_au: Option<f64>,
    pub lat_deg: Option<f64>,
    pub lon_deg: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatticeIndex {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GridMappingConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub r_min_au: f64,
    pub r_max_au: f64,
    pub lat_min_deg: f64,
    pub lat_max_deg: f64,
    pub lon_min_deg: f64,
    pub lon_max_deg: f64,
    pub log_radius: bool,
}

pub struct VoyagerSpatialFeeder {
    points: Vec<SpatialPoint>,
}

impl VoyagerSpatialFeeder {
    pub fn from_omni_records(records: &[OmniRecord]) -> Result<Self> {
        let mut points = Vec::new();
        for record in records {
            let Some(timestamp_ms) = omni_timestamp_ms(record) else {
                continue;
            };
            points.push(SpatialPoint {
                timestamp_ms,
                r_au: record.r_au,
                lat_deg: record.lat_deg,
                lon_deg: record.lon_deg,
            });
        }
        points.sort_by_key(|point| point.timestamp_ms);
        points.dedup_by_key(|point| point.timestamp_ms);
        if points.is_empty() {
            bail!("no valid Voyager spatial points were available");
        }
        Ok(Self { points })
    }

    pub fn from_voyager_merged(path: &Path, spacecraft: VoyagerSpacecraft) -> Result<Self> {
        let raw = parse_voyager_file(path, spacecraft)
            .with_context(|| format!("failed to parse Voyager merged file {}", path.display()))?;
        let omni = voyager_to_omni(&raw);
        Self::from_omni_records(&omni)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn time_bounds(&self) -> Option<TimeBounds> {
        let timestamps: Vec<i64> = self.points.iter().map(|point| point.timestamp_ms).collect();
        TimeBounds::from_sorted_epoch_ms(&timestamps)
    }

    pub fn sample_linear(&self, target_ms: i64) -> Option<VoyagerSpatialSample> {
        if self.points.is_empty() {
            return None;
        }
        let upper = self
            .points
            .partition_point(|point| point.timestamp_ms < target_ms);
        let (lower_index, upper_index) = if upper == 0 {
            (0, 0)
        } else if upper >= self.points.len() {
            let last = self.points.len() - 1;
            (last, last)
        } else if self.points[upper].timestamp_ms == target_ms {
            (upper, upper)
        } else {
            (upper - 1, upper)
        };
        let lower = &self.points[lower_index];
        let upper = &self.points[upper_index];
        let r_au = interp_option(
            lower.r_au,
            upper.r_au,
            lower.timestamp_ms,
            upper.timestamp_ms,
            target_ms,
        );
        let lat_deg = interp_option(
            lower.lat_deg,
            upper.lat_deg,
            lower.timestamp_ms,
            upper.timestamp_ms,
            target_ms,
        );
        let lon_deg = interp_longitude_deg(
            lower.lon_deg,
            upper.lon_deg,
            lower.timestamp_ms,
            upper.timestamp_ms,
            target_ms,
        );
        Some(VoyagerSpatialSample {
            lower_index,
            upper_index,
            lower_time_ms: lower.timestamp_ms,
            upper_time_ms: upper.timestamp_ms,
            r_au,
            lat_deg,
            lon_deg,
        })
    }
}

pub struct FusedEncounterFeeder {
    telemetry: TrajectoryFeeder,
    spatial: VoyagerSpatialFeeder,
}

impl FusedEncounterFeeder {
    pub fn new(telemetry: TrajectoryFeeder, spatial: VoyagerSpatialFeeder) -> Self {
        Self { telemetry, spatial }
    }

    pub fn telemetry(&self) -> &TrajectoryFeeder {
        &self.telemetry
    }

    pub fn spatial(&self) -> &VoyagerSpatialFeeder {
        &self.spatial
    }

    pub fn sample(&self, target_ms: i64) -> Result<Option<FusedEncounterState>> {
        let telemetry = self.telemetry.sample_linear(target_ms)?;
        let spatial = self.spatial.sample_linear(target_ms);
        match (telemetry, spatial) {
            (None, None) => Ok(None),
            (telemetry_sample, spatial_sample) => Ok(Some(FusedEncounterState {
                timestamp_ms: target_ms,
                instrument_value: telemetry_sample.and_then(|sample| sample.interpolated_value),
                r_au: spatial_sample.as_ref().and_then(|sample| sample.r_au),
                lat_deg: spatial_sample.as_ref().and_then(|sample| sample.lat_deg),
                lon_deg: spatial_sample.as_ref().and_then(|sample| sample.lon_deg),
            })),
        }
    }
}

pub fn heliocentric_to_lattice(
    r_au: f64,
    lat_deg: f64,
    lon_deg: f64,
    config: GridMappingConfig,
) -> Option<LatticeIndex> {
    if !(r_au.is_finite() && lat_deg.is_finite() && lon_deg.is_finite()) {
        return None;
    }
    if r_au < config.r_min_au
        || r_au > config.r_max_au
        || lat_deg < config.lat_min_deg
        || lat_deg > config.lat_max_deg
        || lon_deg < config.lon_min_deg
        || lon_deg > config.lon_max_deg
    {
        return None;
    }

    let x_frac = if config.log_radius {
        let ln_min = config.r_min_au.ln();
        let ln_max = config.r_max_au.ln();
        let denom = (ln_max - ln_min).abs();
        if denom <= f64::EPSILON {
            0.0
        } else {
            (r_au.ln() - ln_min) / (ln_max - ln_min)
        }
    } else {
        let denom = (config.r_max_au - config.r_min_au).abs();
        if denom <= f64::EPSILON {
            0.0
        } else {
            (r_au - config.r_min_au) / (config.r_max_au - config.r_min_au)
        }
    };
    let y_frac = frac_in_range(lon_deg, config.lon_min_deg, config.lon_max_deg);
    let z_frac = frac_in_range(lat_deg, config.lat_min_deg, config.lat_max_deg);

    Some(LatticeIndex {
        x: frac_to_index(x_frac, config.nx),
        y: frac_to_index(y_frac, config.ny),
        z: frac_to_index(z_frac, config.nz),
    })
}

fn frac_in_range(value: f64, min_value: f64, max_value: f64) -> f64 {
    let denom = (max_value - min_value).abs();
    if denom <= f64::EPSILON {
        0.0
    } else {
        (value - min_value) / (max_value - min_value)
    }
}

fn frac_to_index(frac: f64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let clamped = frac.clamp(0.0, 1.0);
    let idx = (clamped * (n as f64 - 1.0)).round() as usize;
    idx.min(n - 1)
}

fn omni_timestamp_ms(record: &OmniRecord) -> Option<i64> {
    let date = NaiveDate::from_yo_opt(record.year as i32, record.doy as u32)?;
    let datetime = date.and_hms_opt(record.hour as u32, 0, 0)?;
    Some(Utc.from_utc_datetime(&datetime).timestamp_millis())
}

fn interp_option(
    lower: f64,
    upper: f64,
    lower_time_ms: i64,
    upper_time_ms: i64,
    target_ms: i64,
) -> Option<f64> {
    if !(lower.is_finite() && upper.is_finite()) {
        return None;
    }
    if lower_time_ms == upper_time_ms {
        return Some(lower);
    }
    let frac = (target_ms - lower_time_ms) as f64 / (upper_time_ms - lower_time_ms) as f64;
    Some(lower * (1.0 - frac) + upper * frac)
}

fn interp_longitude_deg(
    lower: f64,
    upper: f64,
    lower_time_ms: i64,
    upper_time_ms: i64,
    target_ms: i64,
) -> Option<f64> {
    if !(lower.is_finite() && upper.is_finite()) {
        return None;
    }
    if lower_time_ms == upper_time_ms {
        return Some(lower);
    }
    let mut delta = upper - lower;
    if delta > 180.0 {
        delta -= 360.0;
    } else if delta < -180.0 {
        delta += 360.0;
    }
    let frac = (target_ms - lower_time_ms) as f64 / (upper_time_ms - lower_time_ms) as f64;
    let value = lower + frac * delta;
    Some(normalize_longitude_deg(value))
}

fn normalize_longitude_deg(value: f64) -> f64 {
    let mut out = value % 360.0;
    if out < 0.0 {
        out += 360.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, RecordBatch,
        builder::{Float64Builder, TimestampMillisecondBuilder},
    };
    use arrow_ipc::writer::FileWriter;
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use std::{
        fs::File,
        path::PathBuf,
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_arrow_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{nanos}.arrow"))
    }

    fn write_sample_arrow(path: &Path) {
        let mut time_builder = TimestampMillisecondBuilder::new();
        let mut value_builder = Float64Builder::new();
        let base_time = Utc
            .with_ymd_and_hms(1979, 7, 3, 0, 0, 0)
            .single()
            .expect("base time")
            .timestamp_millis();
        for (time_ms, value) in [(base_time, 10.0_f64), (base_time + 3_600_000, 20.0)] {
            time_builder.append_value(time_ms);
            value_builder.append_value(value);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "TIME",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new("LD1 RATE", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(time_builder.finish()) as ArrayRef,
                Arc::new(value_builder.finish()) as ArrayRef,
            ],
        )
        .expect("batch");
        let file = File::create(path).expect("create arrow");
        let mut writer = FileWriter::try_new(file, &schema).expect("writer");
        writer.write(&batch).expect("write");
        writer.finish().expect("finish");
    }

    fn sample_omni(hour: u8, r_au: f64, lat_deg: f64, lon_deg: f64) -> OmniRecord {
        OmniRecord {
            year: 1979,
            doy: 184,
            hour,
            b_magnitude: f64::NAN,
            bx_gse: f64::NAN,
            by_gse: f64::NAN,
            bz_gse: f64::NAN,
            proton_temperature: f64::NAN,
            proton_density: f64::NAN,
            bulk_speed: f64::NAN,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au,
            lat_deg,
            lon_deg,
        }
    }

    #[test]
    fn test_spatial_feeder_interpolates_position() {
        let first = sample_omni(0, 5.0, -2.0, 100.0);
        let second = sample_omni(1, 5.5, 2.0, 104.0);
        let target_ms = omni_timestamp_ms(&first).expect("t0") + 30_i64 * 60_i64 * 1000_i64;
        let feeder =
            VoyagerSpatialFeeder::from_omni_records(&[first, second]).expect("spatial feeder");
        let sample = feeder.sample_linear(target_ms).expect("sample");
        assert_eq!(sample.lower_index, 0);
        assert_eq!(sample.upper_index, 1);
        assert!((sample.r_au.expect("r") - 5.25).abs() < 1e-12);
        assert!((sample.lat_deg.expect("lat") - 0.0).abs() < 1e-12);
        assert!((sample.lon_deg.expect("lon") - 102.0).abs() < 1e-12);
    }

    #[test]
    fn test_fused_feeder_joins_telemetry_and_position() {
        let arrow_path = unique_arrow_path("voyager_fused");
        write_sample_arrow(&arrow_path);
        let telemetry = TrajectoryFeeder::open_input(&arrow_path, "LD1 RATE").expect("telemetry");
        let first = sample_omni(0, 5.0, -2.0, 100.0);
        let second = sample_omni(1, 5.5, 2.0, 104.0);
        let target_ms = omni_timestamp_ms(&first).expect("t0") + 30_i64 * 60_i64 * 1000_i64;
        let spatial = VoyagerSpatialFeeder::from_omni_records(&[first, second]).expect("spatial");
        let fused = FusedEncounterFeeder::new(telemetry, spatial);
        let sample = fused.sample(target_ms).expect("sample").expect("non-empty");
        assert!((sample.instrument_value.expect("value") - 15.0).abs() < 1e-12);
        assert!((sample.r_au.expect("r") - 5.25).abs() < 1e-12);
        assert!((sample.lon_deg.expect("lon") - 102.0).abs() < 1e-12);
        let _ = std::fs::remove_file(arrow_path);
    }

    #[test]
    fn test_heliocentric_to_lattice_maps_log_radius_and_angles() {
        let config = GridMappingConfig {
            nx: 5,
            ny: 5,
            nz: 5,
            r_min_au: 1.0,
            r_max_au: 100.0,
            lat_min_deg: -40.0,
            lat_max_deg: 40.0,
            lon_min_deg: 80.0,
            lon_max_deg: 120.0,
            log_radius: true,
        };
        let idx = heliocentric_to_lattice(10.0, 0.0, 100.0, config).expect("idx");
        assert_eq!(idx.x, 2);
        assert_eq!(idx.y, 2);
        assert_eq!(idx.z, 2);
        assert!(heliocentric_to_lattice(200.0, 0.0, 100.0, config).is_none());
    }
}
