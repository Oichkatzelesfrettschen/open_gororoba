use anyhow::{Result, bail};
use data_core::{
    catalogs::{omni::OmniRecord, soho_celias::SohoCeliasRecord},
    time_bounds::{
        TimeBounds, bounds_from_omni, bounds_from_soho_celias, epoch_from_unix_milliseconds,
        epoch_to_unix_milliseconds, format_epoch, omni_timestamp_ms, soho_celias_timestamp_ms,
    },
};
use hifitime::Epoch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorFrame {
    Gse,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct HeliosphericPlasmaState {
    pub timestamp_ms: i64,
    pub epoch: Epoch,
    pub epoch_seconds_j2000_et: f64,
    pub proton_density: Option<f64>,
    pub bulk_speed_kms: Option<f64>,
    pub proton_temperature_k: Option<f64>,
    pub b_gse_nt: [Option<f64>; 3],
    pub r_au: Option<f64>,
    pub lat_deg: Option<f64>,
    pub lon_deg: Option<f64>,
    pub vector_frame: VectorFrame,
    pub distance_measured: bool,
    pub vector_mapped: bool,
    pub temporal_gap: bool,
    pub source_label: String,
}

pub trait PlasmaBoundaryProvider {
    fn label(&self) -> &str;
    fn time_bounds(&self) -> TimeBounds;
    fn sample_at_epoch(&self, target_epoch: Epoch) -> Result<Option<HeliosphericPlasmaState>>;

    fn sample_at_unix_milliseconds(
        &self,
        target_ms: i64,
    ) -> Result<Option<HeliosphericPlasmaState>> {
        self.sample_at_epoch(epoch_from_unix_milliseconds(target_ms))
    }
}

pub struct GapTolerantBoundaryProvider {
    label: String,
    inner: Box<dyn PlasmaBoundaryProvider>,
    native_bounds: TimeBounds,
    declared_bounds: TimeBounds,
}

impl GapTolerantBoundaryProvider {
    pub fn new(
        label: impl Into<String>,
        inner: Box<dyn PlasmaBoundaryProvider>,
        declared_bounds: TimeBounds,
    ) -> Result<Self> {
        if declared_bounds.end_ms < declared_bounds.start_ms {
            bail!("GapTolerantBoundaryProvider requires start <= end");
        }
        let native_bounds = inner.time_bounds();
        Ok(Self {
            label: label.into(),
            inner,
            native_bounds,
            declared_bounds,
        })
    }

    pub fn native_bounds(&self) -> &TimeBounds {
        &self.native_bounds
    }

    pub fn declared_bounds(&self) -> &TimeBounds {
        &self.declared_bounds
    }
}

impl PlasmaBoundaryProvider for GapTolerantBoundaryProvider {
    fn label(&self) -> &str {
        &self.label
    }

    fn time_bounds(&self) -> TimeBounds {
        self.declared_bounds.clone()
    }

    fn sample_at_epoch(&self, target_epoch: Epoch) -> Result<Option<HeliosphericPlasmaState>> {
        if !self.declared_bounds.contains_epoch_window(target_epoch, target_epoch) {
            return Ok(None);
        }
        if !self.native_bounds.contains_epoch_window(target_epoch, target_epoch) {
            return Ok(Some(missing_plasma_state(target_epoch, self.label.clone())));
        }
        match self.inner.sample_at_epoch(target_epoch)? {
            Some(mut state) => {
                state.source_label = self.label.clone();
                Ok(Some(state))
            }
            None => Ok(Some(missing_plasma_state(target_epoch, self.label.clone()))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DatasetBounds {
    pub label: String,
    pub bounds: TimeBounds,
}

pub struct ChronologyGate {
    datasets: Vec<DatasetBounds>,
    overlap: TimeBounds,
}

impl ChronologyGate {
    pub fn from_providers(providers: &[&dyn PlasmaBoundaryProvider]) -> Result<Self> {
        let datasets: Vec<DatasetBounds> = providers
            .iter()
            .map(|provider| DatasetBounds {
                label: provider.label().to_string(),
                bounds: provider.time_bounds(),
            })
            .collect();
        Self::from_dataset_bounds(datasets)
    }

    pub fn from_dataset_bounds(datasets: Vec<DatasetBounds>) -> Result<Self> {
        if datasets.is_empty() {
            bail!("ChronologyGate requires at least one dataset");
        }
        let bounds: Vec<TimeBounds> = datasets.iter().map(|dataset| dataset.bounds.clone()).collect();
        let overlap = TimeBounds::intersect_all(&bounds).ok_or_else(|| {
            anyhow::anyhow!(
                "ChronologyViolation: datasets have no overlapping window: {}",
                datasets
                    .iter()
                    .map(|dataset| format!(
                        "{}=[{}, {}]",
                        dataset.label,
                        format_epoch(dataset.bounds.start_epoch()),
                        format_epoch(dataset.bounds.end_epoch())
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        Ok(Self { datasets, overlap })
    }

    pub fn overlap(&self) -> &TimeBounds {
        &self.overlap
    }

    pub fn datasets(&self) -> &[DatasetBounds] {
        &self.datasets
    }

    pub fn require_window(&self, start_ms: i64, end_ms: i64) -> Result<()> {
        self.require_epoch_window(
            epoch_from_unix_milliseconds(start_ms),
            epoch_from_unix_milliseconds(end_ms),
        )
    }

    pub fn require_epoch_window(&self, start_epoch: Epoch, end_epoch: Epoch) -> Result<()> {
        let start_ms = epoch_to_unix_milliseconds(start_epoch);
        let end_ms = epoch_to_unix_milliseconds(end_epoch);
        if end_ms < start_ms {
            bail!(
                "ChronologyViolation: requested end {} precedes start {}",
                format_epoch(end_epoch),
                format_epoch(start_epoch)
            );
        }
        if !self.overlap.contains_epoch_window(start_epoch, end_epoch) {
            bail!(
                "ChronologyViolation: requested window [{}, {}] falls outside global overlap [{}, {}]",
                format_epoch(start_epoch),
                format_epoch(end_epoch),
                format_epoch(self.overlap.start_epoch()),
                format_epoch(self.overlap.end_epoch())
            );
        }
        Ok(())
    }
}

pub struct OmniBoundaryProvider {
    label: String,
    records: Vec<OmniRecord>,
    timestamps_ms: Vec<i64>,
    bounds: TimeBounds,
    distance_measured: bool,
    vector_mapped: bool,
}

impl OmniBoundaryProvider {
    pub fn new(
        label: impl Into<String>,
        records: Vec<OmniRecord>,
        distance_measured: bool,
        vector_mapped: bool,
    ) -> Result<Self> {
        let label = label.into();
        let timestamps_ms: Vec<i64> = records.iter().filter_map(omni_timestamp_ms).collect();
        let bounds = bounds_from_omni(&records)
            .ok_or_else(|| anyhow::anyhow!("{label} had no valid temporal bounds"))?;
        Ok(Self {
            label,
            records,
            timestamps_ms,
            bounds,
            distance_measured,
            vector_mapped,
        })
    }
}

impl PlasmaBoundaryProvider for OmniBoundaryProvider {
    fn label(&self) -> &str {
        &self.label
    }

    fn time_bounds(&self) -> TimeBounds {
        self.bounds.clone()
    }

    fn sample_at_epoch(&self, target_epoch: Epoch) -> Result<Option<HeliosphericPlasmaState>> {
        let target_ms = epoch_to_unix_milliseconds(target_epoch);
        let Some((lower, upper)) = time_bounds_for(&self.timestamps_ms, target_ms) else {
            return Ok(None);
        };
        let lower_record = &self.records[lower];
        let upper_record = &self.records[upper];
        let lower_time_ms = self.timestamps_ms[lower];
        let upper_time_ms = self.timestamps_ms[upper];
        Ok(Some(HeliosphericPlasmaState {
            timestamp_ms: target_ms,
            epoch: target_epoch,
            epoch_seconds_j2000_et: target_epoch.to_et_seconds(),
            proton_density: interp_value(
                lower_record.proton_density,
                upper_record.proton_density,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            bulk_speed_kms: interp_value(
                lower_record.bulk_speed,
                upper_record.bulk_speed,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            proton_temperature_k: interp_value(
                lower_record.proton_temperature,
                upper_record.proton_temperature,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            b_gse_nt: [
                interp_value(
                    lower_record.bx_gse,
                    upper_record.bx_gse,
                    lower_time_ms,
                    upper_time_ms,
                    target_ms,
                ),
                interp_value(
                    lower_record.by_gse,
                    upper_record.by_gse,
                    lower_time_ms,
                    upper_time_ms,
                    target_ms,
                ),
                interp_value(
                    lower_record.bz_gse,
                    upper_record.bz_gse,
                    lower_time_ms,
                    upper_time_ms,
                    target_ms,
                ),
            ],
            r_au: interp_value(
                lower_record.r_au,
                upper_record.r_au,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            lat_deg: interp_value(
                lower_record.lat_deg,
                upper_record.lat_deg,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            lon_deg: interp_longitude(
                lower_record.lon_deg,
                upper_record.lon_deg,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            vector_frame: VectorFrame::Gse,
            distance_measured: self.distance_measured,
            vector_mapped: self.vector_mapped,
            temporal_gap: false,
            source_label: self.label.clone(),
        }))
    }
}

pub struct SohoCeliasBoundaryProvider {
    label: String,
    records: Vec<SohoCeliasRecord>,
    timestamps_ms: Vec<i64>,
    bounds: TimeBounds,
}

impl SohoCeliasBoundaryProvider {
    pub fn new(label: impl Into<String>, records: Vec<SohoCeliasRecord>) -> Result<Self> {
        let label = label.into();
        let timestamps_ms: Vec<i64> = records.iter().filter_map(soho_celias_timestamp_ms).collect();
        let bounds = bounds_from_soho_celias(&records)
            .ok_or_else(|| anyhow::anyhow!("{label} had no valid temporal bounds"))?;
        Ok(Self {
            label,
            records,
            timestamps_ms,
            bounds,
        })
    }
}

impl PlasmaBoundaryProvider for SohoCeliasBoundaryProvider {
    fn label(&self) -> &str {
        &self.label
    }

    fn time_bounds(&self) -> TimeBounds {
        self.bounds.clone()
    }

    fn sample_at_epoch(&self, target_epoch: Epoch) -> Result<Option<HeliosphericPlasmaState>> {
        let target_ms = epoch_to_unix_milliseconds(target_epoch);
        let Some((lower, upper)) = time_bounds_for(&self.timestamps_ms, target_ms) else {
            return Ok(None);
        };
        let lower_record = &self.records[lower];
        let upper_record = &self.records[upper];
        let lower_time_ms = self.timestamps_ms[lower];
        let upper_time_ms = self.timestamps_ms[upper];
        Ok(Some(HeliosphericPlasmaState {
            timestamp_ms: target_ms,
            epoch: target_epoch,
            epoch_seconds_j2000_et: target_epoch.to_et_seconds(),
            proton_density: interp_value(
                lower_record.proton_density,
                upper_record.proton_density,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            bulk_speed_kms: interp_value(
                lower_record.bulk_speed,
                upper_record.bulk_speed,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            proton_temperature_k: interp_value(
                lower_record.proton_temperature,
                upper_record.proton_temperature,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            b_gse_nt: [None, None, None],
            r_au: interp_value(
                lower_record.r_au,
                upper_record.r_au,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            lat_deg: interp_value(
                lower_record.lat_deg,
                upper_record.lat_deg,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            lon_deg: interp_longitude(
                lower_record.lon_deg,
                upper_record.lon_deg,
                lower_time_ms,
                upper_time_ms,
                target_ms,
            ),
            vector_frame: VectorFrame::Unknown,
            distance_measured: true,
            vector_mapped: false,
            temporal_gap: false,
            source_label: self.label.clone(),
        }))
    }
}

fn missing_plasma_state(target_epoch: Epoch, source_label: String) -> HeliosphericPlasmaState {
    let target_ms = epoch_to_unix_milliseconds(target_epoch);
    HeliosphericPlasmaState {
        timestamp_ms: target_ms,
        epoch: target_epoch,
        epoch_seconds_j2000_et: target_epoch.to_et_seconds(),
        proton_density: None,
        bulk_speed_kms: None,
        proton_temperature_k: None,
        b_gse_nt: [None, None, None],
        r_au: None,
        lat_deg: None,
        lon_deg: None,
        vector_frame: VectorFrame::Unknown,
        distance_measured: false,
        vector_mapped: false,
        temporal_gap: true,
        source_label,
    }
}

fn time_bounds_for(values: &[i64], target_ms: i64) -> Option<(usize, usize)> {
    if values.is_empty() {
        return None;
    }
    let upper = values.partition_point(|value| *value < target_ms);
    if upper == 0 {
        return Some((0, 0));
    }
    if upper >= values.len() {
        let last = values.len() - 1;
        return Some((last, last));
    }
    if values[upper] == target_ms {
        return Some((upper, upper));
    }
    Some((upper - 1, upper))
}

fn interp_value(
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

fn interp_longitude(
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
        return Some(lower.rem_euclid(360.0));
    }
    let frac = (target_ms - lower_time_ms) as f64 / (upper_time_ms - lower_time_ms) as f64;
    let mut delta = upper - lower;
    if delta > 180.0 {
        delta -= 360.0;
    } else if delta < -180.0 {
        delta += 360.0;
    }
    Some((lower + frac * delta).rem_euclid(360.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_omni(year: u16, doy: u16, hour: u8, density: f64) -> OmniRecord {
        OmniRecord {
            year,
            doy,
            hour,
            b_magnitude: f64::NAN,
            bx_gse: 1.0,
            by_gse: 2.0,
            bz_gse: 3.0,
            proton_temperature: 10.0,
            proton_density: density,
            bulk_speed: 400.0,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: 1.0,
            lat_deg: 0.0,
            lon_deg: 10.0,
        }
    }

    #[test]
    fn chronology_gate_intersects_windows() {
        let a = OmniBoundaryProvider::new(
            "a",
            vec![make_omni(2024, 1, 0, 5.0), make_omni(2024, 1, 2, 7.0)],
            true,
            false,
        )
        .expect("provider a");
        let b = OmniBoundaryProvider::new(
            "b",
            vec![make_omni(2024, 1, 1, 6.0), make_omni(2024, 1, 3, 8.0)],
            true,
            false,
        )
        .expect("provider b");
        let gate = ChronologyGate::from_providers(&[&a, &b]).expect("gate");
        assert_eq!(
            gate.overlap().start_ms,
            omni_timestamp_ms(&make_omni(2024, 1, 1, 6.0)).unwrap_or_default()
        );
        assert_eq!(
            gate.overlap().end_ms,
            omni_timestamp_ms(&make_omni(2024, 1, 2, 7.0)).unwrap_or_default()
        );
        assert!(gate.overlap().start_et_seconds().is_finite());
    }

    #[test]
    fn chronology_gate_rejects_disjoint_windows() {
        let a = OmniBoundaryProvider::new("a", vec![make_omni(2024, 1, 0, 5.0)], true, false)
            .expect("provider a");
        let b = OmniBoundaryProvider::new("b", vec![make_omni(2025, 1, 0, 6.0)], true, false)
            .expect("provider b");
        assert!(ChronologyGate::from_providers(&[&a, &b]).is_err());
    }

    #[test]
    fn sample_includes_epoch_metadata() {
        let provider = OmniBoundaryProvider::new(
            "a",
            vec![make_omni(2024, 1, 0, 5.0), make_omni(2024, 1, 1, 7.0)],
            true,
            false,
        )
        .expect("provider");
        let sample = provider
            .sample_at_unix_milliseconds(omni_timestamp_ms(&make_omni(2024, 1, 0, 5.0)).unwrap())
            .expect("sample")
            .expect("state");
        assert_eq!(sample.timestamp_ms, epoch_to_unix_milliseconds(sample.epoch));
        assert!(sample.epoch_seconds_j2000_et.is_finite());
    }

    #[test]
    fn gap_tolerant_provider_returns_missing_state_outside_native_coverage() {
        let native = OmniBoundaryProvider::new(
            "native",
            vec![make_omni(2024, 10, 0, 5.0), make_omni(2024, 10, 1, 7.0)],
            true,
            false,
        )
        .expect("native provider");
        let declared_bounds = TimeBounds {
            start_ms: omni_timestamp_ms(&make_omni(2024, 1, 0, 5.0)).expect("declared start"),
            end_ms: omni_timestamp_ms(&make_omni(2024, 10, 1, 7.0)).expect("declared end"),
            cadence_seconds: Some(3600.0),
        };
        let gap = GapTolerantBoundaryProvider::new("gap", Box::new(native), declared_bounds)
            .expect("gap provider");
        let sample = gap
            .sample_at_unix_milliseconds(
                omni_timestamp_ms(&make_omni(2024, 1, 12, 5.0)).expect("target"),
            )
            .expect("sample")
            .expect("state");
        assert!(sample.temporal_gap);
        assert!(sample.proton_density.is_none());
    }

    #[test]
    fn chronology_gate_accepts_gap_tolerant_declared_window() {
        let full = OmniBoundaryProvider::new(
            "full",
            vec![make_omni(2024, 1, 0, 1.0), make_omni(2024, 10, 1, 2.0)],
            true,
            false,
        )
        .expect("full");
        let native = OmniBoundaryProvider::new(
            "native",
            vec![make_omni(2024, 10, 0, 5.0), make_omni(2024, 10, 1, 7.0)],
            true,
            false,
        )
        .expect("native");
        let declared_bounds = TimeBounds {
            start_ms: omni_timestamp_ms(&make_omni(2024, 1, 0, 1.0)).expect("declared start"),
            end_ms: omni_timestamp_ms(&make_omni(2024, 10, 1, 7.0)).expect("declared end"),
            cadence_seconds: Some(3600.0),
        };
        let gap = GapTolerantBoundaryProvider::new("gap", Box::new(native), declared_bounds)
            .expect("gap");
        let gate = ChronologyGate::from_providers(&[&full, &gap]).expect("gate");
        assert_eq!(
            gate.overlap().start_ms,
            omni_timestamp_ms(&make_omni(2024, 1, 0, 1.0)).expect("expected start")
        );
    }
}
