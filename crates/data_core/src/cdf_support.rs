use crate::fetcher::FetchError;
use cdf::{
    cdf::Cdf,
    record::{
        vxr::{VariableIndexRecord, VariableIndexRecordChild},
        zvdr::ZVariableDescriptorRecord,
    },
    types::CdfType,
};
use chrono::{Datelike, NaiveDate, TimeZone, Timelike, Utc};
use hifitime::{Duration, Epoch};
use std::path::Path;

const TT2000_REF_TAI: (i32, u8, u8, u8, u8, u8, u32) = (2000, 1, 1, 11, 59, 27, 816_000_000);

pub fn read_cdf_file(path: &Path) -> Result<Cdf, FetchError> {
    Cdf::read_cdf_file(path).map_err(|err| FetchError::Validation(format!("CDF parse error: {err}")))
}

fn zvariable_by_name<'a>(cdf: &'a Cdf, name: &str) -> Option<&'a ZVariableDescriptorRecord> {
    cdf.cdr
        .gdr
        .zvdr_vec
        .iter()
        .find(|zvdr| zvdr.name.as_ref() == name)
}

fn collect_vxr_rows(
    vxr: &VariableIndexRecord,
    rows: &mut Vec<Vec<CdfType>>,
) -> Result<(), FetchError> {
    for child in &vxr.children {
        let Some(child) = child else {
            continue;
        };
        match child {
            VariableIndexRecordChild::VVR(vvr) => {
                for record in &vvr.records {
                    rows.push(record.data.clone());
                }
            }
            VariableIndexRecordChild::VXR(inner) => collect_vxr_rows(inner, rows)?,
            VariableIndexRecordChild::CVVR(_) => {
                return Err(FetchError::Validation(
                    "compressed CDF variable blocks are not yet supported in the Rust CDF helper"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

pub fn cdf_variable_rows(cdf: &Cdf, name: &str) -> Result<Vec<Vec<CdfType>>, FetchError> {
    let zvdr = zvariable_by_name(cdf, name).ok_or_else(|| {
        FetchError::Validation(format!("missing CDF zVariable {name}"))
    })?;
    let mut rows = Vec::new();
    for vxr in &zvdr.vxr_vec {
        collect_vxr_rows(vxr, &mut rows)?;
    }
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "CDF zVariable {name} has no readable rows"
        )));
    }
    Ok(rows)
}

pub fn cdf_scalar_f64_rows(cdf: &Cdf, name: &str) -> Result<Vec<f64>, FetchError> {
    let rows = cdf_variable_rows(cdf, name)?;
    rows.into_iter()
        .map(|row| {
            let Some(value) = row.first() else {
                return Err(FetchError::Validation(format!(
                    "CDF zVariable {name} contained an empty row"
                )));
            };
            cdf_type_to_f64(value).ok_or_else(|| {
                FetchError::Validation(format!("CDF zVariable {name} is not numeric"))
            })
        })
        .collect()
}

pub fn cdf_vector_f64_rows(cdf: &Cdf, name: &str) -> Result<Vec<Vec<f64>>, FetchError> {
    let rows = cdf_variable_rows(cdf, name)?;
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| {
                    cdf_type_to_f64(&value).ok_or_else(|| {
                        FetchError::Validation(format!("CDF zVariable {name} contains non-numeric data"))
                    })
                })
                .collect()
        })
        .collect()
}

pub fn cdf_type_to_f64(value: &CdfType) -> Option<f64> {
    match value {
        CdfType::Int1(v) => Some(f64::from(i8::from(v.clone()))),
        CdfType::Int2(v) => Some(f64::from(i16::from(v.clone()))),
        CdfType::Int4(v) => Some(f64::from(i32::from(v.clone()))),
        CdfType::Int8(v) => Some(i64::from(v.clone()) as f64),
        CdfType::Uint1(v) => Some(f64::from(u8::from(v.clone()))),
        CdfType::Uint2(v) => Some(f64::from(u16::from(v.clone()))),
        CdfType::Uint4(v) => Some(f64::from(u32::from(v.clone()))),
        CdfType::Real4(v) => Some(f64::from(f32::from(v.clone()))),
        CdfType::Real8(v) => Some(f64::from(v.clone())),
        CdfType::Epoch(v) => Some(f64::from(v.clone())),
        CdfType::TimeTt2000(v) => Some(i64::from(v.clone()) as f64),
        _ => None,
    }
}

pub fn cdf_type_to_ydh(value: &CdfType) -> Option<(u16, u16, u8)> {
    let timestamp_ms = cdf_type_to_unix_ms(value)?;
    let dt = Utc.timestamp_millis_opt(timestamp_ms).single()?;
    Some((dt.year() as u16, dt.ordinal() as u16, dt.hour() as u8))
}

pub fn cdf_type_to_unix_ms(value: &CdfType) -> Option<i64> {
    match value {
        CdfType::Epoch(v) => cdf_epoch_ms_to_unix_ms(f64::from(v.clone())),
        CdfType::TimeTt2000(v) => tt2000_to_unix_ms(i64::from(v.clone())),
        _ => None,
    }
}

fn cdf_epoch_ms_to_unix_ms(epoch_ms: f64) -> Option<i64> {
    if !epoch_ms.is_finite() {
        return None;
    }
    let base = NaiveDate::from_ymd_opt(0, 1, 1)?.and_hms_milli_opt(0, 0, 0, 0)?;
    let delta = chrono::Duration::milliseconds(epoch_ms.round() as i64);
    let dt = base.checked_add_signed(delta)?;
    Some(Utc.from_utc_datetime(&dt).timestamp_millis())
}

fn tt2000_to_unix_ms(tt2000_ns: i64) -> Option<i64> {
    let base = Epoch::from_gregorian_tai(
        TT2000_REF_TAI.0,
        TT2000_REF_TAI.1,
        TT2000_REF_TAI.2,
        TT2000_REF_TAI.3,
        TT2000_REF_TAI.4,
        TT2000_REF_TAI.5,
        TT2000_REF_TAI.6,
    );
    let epoch = base + Duration::from_total_nanoseconds(i128::from(tt2000_ns));
    Some(epoch.to_unix_milliseconds().round() as i64)
}

pub fn filename_date_yyyymmdd(path: &Path) -> Option<(u16, u8, u8)> {
    let stem = path.file_name()?.to_string_lossy();
    let bytes = stem.as_bytes();
    for idx in 0..bytes.len().saturating_sub(7) {
        if bytes[idx..idx + 8].iter().all(|byte| byte.is_ascii_digit()) {
            let text = &stem[idx..idx + 8];
            let year = text[0..4].parse::<u16>().ok()?;
            let month = text[4..6].parse::<u8>().ok()?;
            let day = text[6..8].parse::<u8>().ok()?;
            if NaiveDate::from_ymd_opt(i32::from(year), u32::from(month), u32::from(day))
                .is_some()
            {
                return Some((year, month, day));
            }
        }
    }
    None
}

pub fn ymd_to_doy(year: u16, month: u8, day: u8) -> Option<u16> {
    NaiveDate::from_ymd_opt(i32::from(year), u32::from(month), u32::from(day))
        .map(|date| date.ordinal() as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filename_date_yyyymmdd() {
        let path = Path::new("imap_hi_l2_h90-ena-h-sf-nsp-full-4deg-3mo_20260101_v001.cdf");
        assert_eq!(filename_date_yyyymmdd(path), Some((2026, 1, 1)));
    }

    #[test]
    fn test_ymd_to_doy() {
        assert_eq!(ymd_to_doy(2020, 1, 1), Some(1));
        assert_eq!(ymd_to_doy(2020, 2, 29), Some(60));
    }

    #[test]
    fn test_tt2000_zero_maps_to_j2000_utc() {
        let value = CdfType::TimeTt2000(0_i64.into());
        let (year, doy, hour) = cdf_type_to_ydh(&value).expect("tt2000");
        assert_eq!((year, doy, hour), (2000, 1, 11));
    }
}
