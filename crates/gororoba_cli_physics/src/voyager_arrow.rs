use anyhow::{Context, Result, bail};
use arrow_array::{Array, Float64Array, RecordBatch, TimestampMillisecondArray};
use arrow_ipc::reader::FileReader;
use arrow_schema::{DataType, TimeUnit};
use data_core::time_bounds::TimeBounds;
use memmap2::Mmap;
use std::{
    fs,
    fs::File,
    io::Cursor,
    path::{Path, PathBuf},
};

pub struct VoyagerArrowDataset {
    path: PathBuf,
    mmap: Mmap,
    batch: RecordBatch,
    time_column_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissionPhase {
    JupiterEncounter,
}

impl MissionPhase {
    pub fn as_path_component(self) -> &'static str {
        match self {
            MissionPhase::JupiterEncounter => "jupiter_encounter",
        }
    }
}

impl VoyagerArrowDataset {
    pub fn open(path: impl AsRef<Path>, required_columns: &[RequiredColumn]) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)
            .with_context(|| format!("failed to open Arrow file {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap Arrow file {}", path.display()))?;
        let cursor = Cursor::new(&mmap[..]);
        let mut reader = FileReader::try_new(cursor, None)
            .with_context(|| format!("failed to decode Arrow file {}", path.display()))?;

        let mut batches = Vec::new();
        for batch in &mut reader {
            batches.push(
                batch.with_context(|| format!("failed to read batch from {}", path.display()))?,
            );
        }
        if batches.len() != 1 {
            bail!(
                "expected exactly one record batch in {}, found {}",
                path.display(),
                batches.len()
            );
        }
        let batch = batches.remove(0);
        let schema = batch.schema();

        let mut time_column_index = None;
        for required in required_columns {
            let (index, field) = schema
                .fields()
                .iter()
                .enumerate()
                .find(|(_, field)| field.name().as_str() == required.name.as_str())
                .with_context(|| {
                    format!(
                        "required column {} missing from {}",
                        required.name,
                        path.display()
                    )
                })?;
            if !required.matches(field.data_type()) {
                bail!(
                    "column {} in {} had type {:?}, expected {:?}",
                    required.name,
                    path.display(),
                    field.data_type(),
                    required.expected
                );
            }
            if matches!(required.expected, RequiredType::TimestampMillis) {
                time_column_index = Some(index);
            }
        }

        Ok(Self {
            path,
            mmap,
            batch,
            time_column_index: time_column_index.context("missing required timestamp column")?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_len_bytes(&self) -> usize {
        self.mmap.len()
    }

    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn timestamp_array(&self) -> Result<&TimestampMillisecondArray> {
        self.batch
            .column(self.time_column_index)
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .context("timestamp column was not a TimestampMillisecondArray")
    }

    pub fn timestamp_values(&self) -> Result<&[i64]> {
        Ok(self.timestamp_array()?.values())
    }

    pub fn assert_monotonic_time(&self) -> Result<()> {
        let times = self.timestamp_values()?;
        if times.is_empty() {
            bail!("{} has no timestamps", self.path.display());
        }
        for window in times.windows(2) {
            if window[1] < window[0] {
                bail!(
                    "{} is not monotonic at {} -> {}",
                    self.path.display(),
                    window[0],
                    window[1]
                );
            }
        }
        Ok(())
    }

    pub fn time_bounds_for(&self, target_ms: i64) -> Result<Option<(usize, usize)>> {
        let times = self.timestamp_values()?;
        if times.is_empty() {
            return Ok(None);
        }
        let upper = times.partition_point(|value| *value < target_ms);
        if upper == 0 {
            return Ok(Some((0, 0)));
        }
        if upper >= times.len() {
            let last = times.len() - 1;
            return Ok(Some((last, last)));
        }
        if times[upper] == target_ms {
            return Ok(Some((upper, upper)));
        }
        Ok(Some((upper - 1, upper)))
    }

    pub fn float64_column(&self, name: &str) -> Result<&Float64Array> {
        let index = self
            .batch
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == name)
            .with_context(|| format!("column {} missing from {}", name, self.path.display()))?;
        self.batch
            .column(index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .with_context(|| format!("column {} was not Float64 in {}", name, self.path.display()))
    }

    pub fn is_valid(&self, column: &str, index: usize) -> Result<bool> {
        let arr = self.float64_column(column)?;
        if index >= arr.len() {
            bail!(
                "index {} out of bounds for column {} in {}",
                index,
                column,
                self.path.display()
            );
        }
        Ok(!arr.is_null(index))
    }

    pub fn float64_value(&self, column: &str, index: usize) -> Result<Option<f64>> {
        let arr = self.float64_column(column)?;
        if index >= arr.len() {
            bail!(
                "index {} out of bounds for column {} in {}",
                index,
                column,
                self.path.display()
            );
        }
        if arr.is_null(index) {
            return Ok(None);
        }
        Ok(Some(arr.value(index)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequiredType {
    TimestampMillis,
    Float64,
}

#[derive(Clone, Debug)]
pub struct RequiredColumn {
    pub name: String,
    pub expected: RequiredType,
}

impl RequiredColumn {
    pub fn timestamp_ms(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected: RequiredType::TimestampMillis,
        }
    }

    pub fn float64(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected: RequiredType::Float64,
        }
    }

    fn matches(&self, actual: &DataType) -> bool {
        match self.expected {
            RequiredType::TimestampMillis => {
                matches!(actual, DataType::Timestamp(TimeUnit::Millisecond, None))
            }
            RequiredType::Float64 => matches!(actual, DataType::Float64),
        }
    }
}

#[derive(Clone, Debug)]
pub struct InterpolatedSample {
    pub lower_index: usize,
    pub upper_index: usize,
    pub lower_time_ms: i64,
    pub upper_time_ms: i64,
    pub lower_value: Option<f64>,
    pub upper_value: Option<f64>,
    pub interpolated_value: Option<f64>,
}

pub struct TrajectoryFeeder {
    dataset: VoyagerArrowDataset,
    value_column: String,
}

impl TrajectoryFeeder {
    pub fn open_input(path: impl AsRef<Path>, value_column: impl Into<String>) -> Result<Self> {
        let value_column = value_column.into();
        let dataset = VoyagerArrowDataset::open(
            path,
            &[
                RequiredColumn::timestamp_ms("TIME"),
                RequiredColumn::float64(value_column.clone()),
            ],
        )?;
        dataset.assert_monotonic_time()?;
        Ok(Self {
            dataset,
            value_column,
        })
    }

    pub fn open_mission_phase(
        repo_root: impl AsRef<Path>,
        phase: MissionPhase,
        spacecraft: u8,
        product_id: &str,
        value_column: impl Into<String>,
    ) -> Result<Self> {
        let path = discover_promoted_arrow(repo_root, phase, spacecraft, product_id)?;
        Self::open_input(path, value_column)
    }

    pub fn dataset(&self) -> &VoyagerArrowDataset {
        &self.dataset
    }

    pub fn value_column(&self) -> &str {
        &self.value_column
    }

    pub fn sample_linear(&self, target_ms: i64) -> Result<Option<InterpolatedSample>> {
        let Some((lower, upper)) = self.dataset.time_bounds_for(target_ms)? else {
            return Ok(None);
        };
        let times = self.dataset.timestamp_values()?;
        let lower_time_ms = times[lower];
        let upper_time_ms = times[upper];
        let lower_value = self.dataset.float64_value(&self.value_column, lower)?;
        let upper_value = self.dataset.float64_value(&self.value_column, upper)?;
        let interpolated_value = if lower == upper || lower_time_ms == upper_time_ms {
            lower_value
        } else if let (Some(a), Some(b)) = (lower_value, upper_value) {
            let frac = (target_ms - lower_time_ms) as f64 / (upper_time_ms - lower_time_ms) as f64;
            Some(a * (1.0 - frac) + b * frac)
        } else {
            None
        };
        Ok(Some(InterpolatedSample {
            lower_index: lower,
            upper_index: upper,
            lower_time_ms,
            upper_time_ms,
            lower_value,
            upper_value,
            interpolated_value,
        }))
    }

    pub fn window_bounds(&self, start_ms: i64, end_ms: i64) -> Result<Option<(usize, usize)>> {
        if end_ms < start_ms {
            bail!("window end {} precedes start {}", end_ms, start_ms);
        }
        let times = self.dataset.timestamp_values()?;
        if times.is_empty() {
            return Ok(None);
        }
        let start = times.partition_point(|value| *value < start_ms);
        let end_exclusive = times.partition_point(|value| *value <= end_ms);
        if start >= end_exclusive {
            return Ok(None);
        }
        Ok(Some((start, end_exclusive - 1)))
    }

    pub fn time_bounds(&self) -> Result<TimeBounds> {
        let times = self.dataset.timestamp_values()?;
        TimeBounds::from_sorted_epoch_ms(times)
            .with_context(|| format!("{} had no time bounds", self.dataset.path().display()))
    }
}

pub fn default_repo_root() -> Result<PathBuf> {
    std::env::current_dir().context("failed to resolve current working directory")
}

pub fn promoted_phase_root(
    repo_root: impl AsRef<Path>,
    phase: MissionPhase,
    spacecraft: u8,
) -> PathBuf {
    repo_root
        .as_ref()
        .join("data/output/heliosphere/voyager")
        .join(phase.as_path_component())
        .join(format!("voyager{spacecraft}"))
}

pub fn discover_promoted_arrow(
    repo_root: impl AsRef<Path>,
    phase: MissionPhase,
    spacecraft: u8,
    product_id: &str,
) -> Result<PathBuf> {
    let root = promoted_phase_root(repo_root, phase, spacecraft);
    let wanted = normalize_product_key(product_id);
    let mut matches = Vec::new();
    for entry in fs::read_dir(&root)
        .with_context(|| format!("failed to read promoted phase directory {}", root.display()))?
    {
        let entry =
            entry.with_context(|| format!("bad directory entry under {}", root.display()))?;
        let path = entry.path();
        if !path.is_file() || path.extension().is_none_or(|ext| ext != "arrow") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if normalize_product_key(stem).contains(&wanted) {
            matches.push(path);
        }
    }
    matches.sort();
    match matches.len() {
        0 => bail!(
            "no promoted Arrow artifact for product {} under {}",
            product_id,
            root.display()
        ),
        1 => Ok(matches.remove(0)),
        _ => bail!(
            "multiple promoted Arrow artifacts matched product {} under {}: {:?}",
            product_id,
            root.display(),
            matches
        ),
    }
}

fn normalize_product_key(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| {
            ch.to_ascii_lowercase()
                .to_string()
                .chars()
                .collect::<Vec<_>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, RecordBatch,
        builder::{Float64Builder, TimestampMillisecondBuilder},
    };
    use arrow_ipc::writer::FileWriter;
    use arrow_schema::{Field, Schema};
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_test_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{nanos}.arrow"))
    }

    fn write_sample_arrow(path: &Path) -> Result<()> {
        let mut time_builder = TimestampMillisecondBuilder::new();
        time_builder.append_value(1_000);
        time_builder.append_value(2_000);
        time_builder.append_value(4_000);
        let time_array = time_builder.finish();

        let mut flux_builder = Float64Builder::new();
        flux_builder.append_value(1.5);
        flux_builder.append_null();
        flux_builder.append_value(3.5);
        let flux_array = flux_builder.finish();

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "TIME",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
            Field::new("ELECTRON_FLUX_A", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(time_array) as ArrayRef,
                Arc::new(flux_array) as ArrayRef,
            ],
        )
        .context("build sample batch")?;

        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut writer = FileWriter::try_new(file, &schema)
            .with_context(|| format!("writer {}", path.display()))?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(())
    }

    #[test]
    fn test_loader_asserts_schema_and_binary_searches_time() {
        let path = unique_test_path("voyager_arrow_loader");
        write_sample_arrow(&path).expect("write sample arrow");

        let dataset = VoyagerArrowDataset::open(
            &path,
            &[
                RequiredColumn::timestamp_ms("TIME"),
                RequiredColumn::float64("ELECTRON_FLUX_A"),
            ],
        )
        .expect("open dataset");

        assert_eq!(dataset.num_rows(), 3);
        assert!(dataset.file_len_bytes() > 0);
        dataset.assert_monotonic_time().expect("monotonic");
        assert_eq!(dataset.time_bounds_for(500).expect("bounds"), Some((0, 0)));
        assert_eq!(
            dataset.time_bounds_for(1_000).expect("bounds"),
            Some((0, 0))
        );
        assert_eq!(
            dataset.time_bounds_for(1_500).expect("bounds"),
            Some((0, 1))
        );
        assert_eq!(
            dataset.time_bounds_for(5_000).expect("bounds"),
            Some((2, 2))
        );
        assert!(!dataset.is_valid("ELECTRON_FLUX_A", 1).expect("validity"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_trajectory_feeder_interpolates_and_windows() {
        let path = unique_test_path("voyager_arrow_feeder");
        write_sample_arrow(&path).expect("write sample arrow");
        let feeder = TrajectoryFeeder::open_input(&path, "ELECTRON_FLUX_A").expect("open feeder");
        let sample = feeder
            .sample_linear(1_500)
            .expect("sample")
            .expect("non-empty");
        assert_eq!(sample.lower_index, 0);
        assert_eq!(sample.upper_index, 1);
        assert_eq!(sample.lower_value, Some(1.5));
        assert_eq!(sample.upper_value, None);
        assert_eq!(sample.interpolated_value, None);
        assert_eq!(
            feeder.window_bounds(900, 2_100).expect("window"),
            Some((0, 1))
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_discover_promoted_arrow_matches_product_ids() {
        let root = std::env::temp_dir().join(format!(
            "voyager_arrow_discovery_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let phase_dir = promoted_phase_root(&root, MissionPhase::JupiterEncounter, 2);
        std::fs::create_dir_all(&phase_dir).expect("create phase dir");
        let arrow_path = phase_dir.join("VG2-J-CRS-5-SUMM-FLUX-V1.0_ld1_rate.tab.arrow");
        std::fs::write(&arrow_path, b"dummy").expect("write dummy arrow");
        let discovered =
            discover_promoted_arrow(&root, MissionPhase::JupiterEncounter, 2, "LD1_RATE")
                .expect("discover");
        assert_eq!(discovered, arrow_path);
        let _ = std::fs::remove_dir_all(root);
    }
}
