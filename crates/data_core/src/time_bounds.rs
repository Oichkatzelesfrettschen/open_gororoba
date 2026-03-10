use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use hifitime::Epoch;

use crate::catalogs::{omni::OmniRecord, soho_celias::SohoCeliasRecord};

#[derive(Clone, Debug, PartialEq)]
pub struct TimeBounds {
    pub start_ms: i64,
    pub end_ms: i64,
    pub cadence_seconds: Option<f64>,
}

impl TimeBounds {
    pub fn from_sorted_epoch_ms(values: &[i64]) -> Option<Self> {
        let (&start_ms, &end_ms) = (values.first()?, values.last()?);
        let cadence_seconds = values
            .windows(2)
            .map(|window| window[1] - window[0])
            .find(|delta_ms| *delta_ms > 0)
            .map(|delta_ms| delta_ms as f64 / 1000.0);
        Some(Self {
            start_ms,
            end_ms,
            cadence_seconds,
        })
    }

    pub fn from_sorted_epochs(values: &[Epoch]) -> Option<Self> {
        let millis: Vec<i64> = values
            .iter()
            .map(|epoch| epoch_to_unix_milliseconds(*epoch))
            .collect();
        Self::from_sorted_epoch_ms(&millis)
    }

    pub fn intersect_all(bounds: &[Self]) -> Option<Self> {
        let start_ms = bounds.iter().map(|b| b.start_ms).max()?;
        let end_ms = bounds.iter().map(|b| b.end_ms).min()?;
        if start_ms > end_ms {
            return None;
        }
        Some(Self {
            start_ms,
            end_ms,
            cadence_seconds: None,
        })
    }

    pub fn contains_window(&self, start_ms: i64, end_ms: i64) -> bool {
        start_ms >= self.start_ms && end_ms <= self.end_ms
    }

    pub fn contains_epoch_window(&self, start_epoch: Epoch, end_epoch: Epoch) -> bool {
        self.contains_window(
            epoch_to_unix_milliseconds(start_epoch),
            epoch_to_unix_milliseconds(end_epoch),
        )
    }

    pub fn start_epoch(&self) -> Epoch {
        epoch_from_unix_milliseconds(self.start_ms)
    }

    pub fn end_epoch(&self) -> Epoch {
        epoch_from_unix_milliseconds(self.end_ms)
    }

    pub fn start_et_seconds(&self) -> f64 {
        self.start_epoch().to_et_seconds()
    }

    pub fn end_et_seconds(&self) -> f64 {
        self.end_epoch().to_et_seconds()
    }
}

pub fn epoch_from_unix_milliseconds(timestamp_ms: i64) -> Epoch {
    Epoch::from_unix_milliseconds(timestamp_ms as f64)
}

pub fn epoch_to_unix_milliseconds(epoch: Epoch) -> i64 {
    epoch.to_unix_milliseconds().round() as i64
}

pub fn format_timestamp_ms(timestamp_ms: i64) -> String {
    epoch_from_unix_milliseconds(timestamp_ms).to_rfc3339()
}

pub fn format_epoch(epoch: Epoch) -> String {
    epoch.to_rfc3339()
}

pub fn epoch_ms_to_datetime(timestamp_ms: i64) -> Option<DateTime<Utc>> {
    Utc.timestamp_millis_opt(timestamp_ms).single()
}

pub fn omni_timestamp_ms(record: &OmniRecord) -> Option<i64> {
    let date = NaiveDate::from_yo_opt(i32::from(record.year), u32::from(record.doy))?;
    let datetime = date.and_hms_opt(u32::from(record.hour), 0, 0)?;
    Some(Utc.from_utc_datetime(&datetime).timestamp_millis())
}

pub fn omni_epoch(record: &OmniRecord) -> Option<Epoch> {
    omni_timestamp_ms(record).map(epoch_from_unix_milliseconds)
}

pub fn soho_celias_timestamp_ms(record: &SohoCeliasRecord) -> Option<i64> {
    let date = NaiveDate::from_yo_opt(i32::from(record.year), u32::from(record.doy))?;
    let datetime = date.and_hms_opt(
        u32::from(record.hour),
        u32::from(record.minute),
        u32::from(record.second),
    )?;
    Some(Utc.from_utc_datetime(&datetime).timestamp_millis())
}

pub fn soho_celias_epoch(record: &SohoCeliasRecord) -> Option<Epoch> {
    soho_celias_timestamp_ms(record).map(epoch_from_unix_milliseconds)
}

pub fn bounds_from_omni(records: &[OmniRecord]) -> Option<TimeBounds> {
    let timestamps: Vec<i64> = records.iter().filter_map(omni_timestamp_ms).collect();
    TimeBounds::from_sorted_epoch_ms(&timestamps)
}

pub fn bounds_from_soho_celias(records: &[SohoCeliasRecord]) -> Option<TimeBounds> {
    let timestamps: Vec<i64> = records.iter().filter_map(soho_celias_timestamp_ms).collect();
    TimeBounds::from_sorted_epoch_ms(&timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect_all() {
        let a = TimeBounds {
            start_ms: 10,
            end_ms: 30,
            cadence_seconds: Some(3600.0),
        };
        let b = TimeBounds {
            start_ms: 20,
            end_ms: 40,
            cadence_seconds: Some(300.0),
        };
        let overlap = TimeBounds::intersect_all(&[a, b]).expect("overlap");
        assert_eq!(overlap.start_ms, 20);
        assert_eq!(overlap.end_ms, 30);
    }

    #[test]
    fn test_from_sorted_epoch_ms() {
        let bounds = TimeBounds::from_sorted_epoch_ms(&[1000, 2000, 3000]).expect("bounds");
        assert_eq!(bounds.start_ms, 1000);
        assert_eq!(bounds.end_ms, 3000);
        assert_eq!(bounds.cadence_seconds, Some(1.0));
    }

    #[test]
    fn test_epoch_round_trip() {
        let epoch = epoch_from_unix_milliseconds(1_720_000_000_000);
        assert_eq!(epoch_to_unix_milliseconds(epoch), 1_720_000_000_000);
        assert!(epoch.to_et_seconds().is_finite());
    }
}
