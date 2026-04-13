use anyhow::{Context, Result};
use crate::spatial::{ecliptic_to_equatorial_vector, parse_dms_radians, parse_hms_radians};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default)]
pub struct TimingModelMetadata {
    pub file_count: usize,
    pub start_mjd: Option<f64>,
    pub finish_mjd: Option<f64>,
    pub dm: Option<f64>,
    pub px_mas: Option<f64>,
    pub elong_deg: Option<f64>,
    pub elat_deg: Option<f64>,
    pub raj: Option<String>,
    pub decj: Option<String>,
}

impl TimingModelMetadata {
    fn merge_from(&mut self, other: TimingModelMetadata) {
        self.file_count += other.file_count;
        update_min(&mut self.start_mjd, other.start_mjd);
        update_max(&mut self.finish_mjd, other.finish_mjd);
        set_if_none(&mut self.dm, other.dm);
        set_if_none(&mut self.px_mas, other.px_mas);
        set_if_none(&mut self.elong_deg, other.elong_deg);
        set_if_none(&mut self.elat_deg, other.elat_deg);
        set_string_if_none(&mut self.raj, other.raj);
        set_string_if_none(&mut self.decj, other.decj);
    }
}

#[derive(Debug, Clone)]
pub struct WidebandDmPoint {
    pub mjd: f64,
    pub frequency_mhz: f64,
    pub uncertainty_us: f64,
    pub pp_dm: f64,
    pub pp_dme: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct DmxPoint {
    pub epoch_mjd: f64,
    pub dmx_value: f64,
    pub dmx_error: Option<f64>,
    pub window_start_mjd: Option<f64>,
    pub window_end_mjd: Option<f64>,
    pub bin_label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResidualPoint {
    pub mjd: f64,
    pub frequency_mhz: Option<f64>,
    pub residual_us: f64,
    pub white_residual_us: Option<f64>,
    pub uncertainty_us: Option<f64>,
    pub chi2: Option<f64>,
    pub flag: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct PulsarTimingData {
    pub pulsar: String,
    pub narrowband_par: TimingModelMetadata,
    pub wideband_par: TimingModelMetadata,
    pub wideband_dm: Vec<WidebandDmPoint>,
    pub dmx: Vec<DmxPoint>,
    pub avg_residuals: Vec<ResidualPoint>,
    pub full_residuals: Vec<ResidualPoint>,
}

impl PulsarTimingData {
    fn new(pulsar: String) -> Self {
        Self {
            pulsar,
            ..Self::default()
        }
    }

    pub fn preferred_metadata(&self) -> &TimingModelMetadata {
        if self.wideband_par.elong_deg.is_some()
            || self.wideband_par.elat_deg.is_some()
            || self.wideband_par.raj.is_some()
            || self.wideband_par.decj.is_some()
        {
            &self.wideband_par
        } else {
            &self.narrowband_par
        }
    }

    pub fn sky_vector(&self) -> Option<[f64; 3]> {
        let meta = self.preferred_metadata();
        if let (Some(elong_deg), Some(elat_deg)) = (meta.elong_deg, meta.elat_deg) {
            return Some(ecliptic_to_equatorial_vector(
                elong_deg.to_radians(),
                elat_deg.to_radians(),
            ));
        }
        let raj = meta.raj.as_deref()?;
        let decj = meta.decj.as_deref()?;
        let ra_rad = parse_hms_radians(raj)?;
        let dec_rad = parse_dms_radians(decj)?;
        Some([
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        ])
    }
}

pub fn load_release(root: &Path) -> Result<BTreeMap<String, PulsarTimingData>> {
    let mut pulsars = BTreeMap::new();
    walk_release(root, root, &mut pulsars)?;
    for data in pulsars.values_mut() {
        data.wideband_dm.sort_by(|a, b| a.mjd.total_cmp(&b.mjd));
        data.dmx.sort_by(|a, b| a.epoch_mjd.total_cmp(&b.epoch_mjd));
        data.avg_residuals.sort_by(|a, b| a.mjd.total_cmp(&b.mjd));
        data.full_residuals.sort_by(|a, b| a.mjd.total_cmp(&b.mjd));
    }
    Ok(pulsars)
}

fn walk_release(
    release_root: &Path,
    scan_dir: &Path,
    pulsars: &mut BTreeMap<String, PulsarTimingData>,
) -> Result<()> {
    for entry in
        fs::read_dir(scan_dir).with_context(|| format!("read_dir {}", scan_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_release(release_root, &path, pulsars)?;
            continue;
        }
        inventory_file(release_root, &path, pulsars)?;
    }
    Ok(())
}

fn inventory_file(
    root: &Path,
    path: &Path,
    pulsars: &mut BTreeMap<String, PulsarTimingData>,
) -> Result<()> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("strip_prefix {} from {}", root.display(), path.display()))?;
    let rel = relative.to_string_lossy();

    if rel.contains("wideband/tim/") && rel.ends_with(".tim") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let data = ensure_pulsar(pulsars, &pulsar);
        data.wideband_dm.extend(parse_wideband_tim(path)?);
        return Ok(());
    }
    if rel.contains("narrowband/par/") && rel.ends_with(".par") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let data = ensure_pulsar(pulsars, &pulsar);
        data.narrowband_par.merge_from(parse_par_file(path)?);
        return Ok(());
    }
    if rel.contains("wideband/par/") && rel.ends_with(".par") {
        let pulsar = canonical_pulsar_name(path_stem_string(path)?);
        let data = ensure_pulsar(pulsars, &pulsar);
        data.wideband_par.merge_from(parse_par_file(path)?);
        return Ok(());
    }
    if rel.contains("wideband/dmx/") && rel.ends_with(".out") {
        let pulsar = canonical_pulsar_name(file_name_string(path)?);
        let data = ensure_pulsar(pulsars, &pulsar);
        data.dmx.extend(parse_dmx_file(path)?);
        return Ok(());
    }
    if rel.contains("residuals/") && rel.ends_with(".res") {
        let pulsar = canonical_pulsar_name(file_name_string(path)?);
        let data = ensure_pulsar(pulsars, &pulsar);
        let residuals = parse_residual_file(path)?;
        if file_name_string(path)?.contains(".avg.") {
            data.avg_residuals.extend(residuals);
        } else {
            data.full_residuals.extend(residuals);
        }
    }
    Ok(())
}

fn ensure_pulsar<'a>(
    pulsars: &'a mut BTreeMap<String, PulsarTimingData>,
    pulsar: &str,
) -> &'a mut PulsarTimingData {
    pulsars
        .entry(pulsar.to_string())
        .or_insert_with(|| PulsarTimingData::new(pulsar.to_string()))
}

fn path_stem_string(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing UTF-8 file stem for {}", path.display()))
}

fn file_name_string(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing UTF-8 file name for {}", path.display()))
}

pub fn canonical_pulsar_name(raw: String) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if out.is_empty() {
            if ch == 'J' || ch == 'B' {
                out.push(ch);
            } else if ch.is_ascii_alphanumeric() {
                continue;
            } else {
                break;
            }
            continue;
        }
        if ch.is_ascii_digit() || ch == '+' || ch == '-' {
            out.push(ch);
        } else {
            break;
        }
    }
    if out.is_empty() { raw } else { out }
}

fn parse_wideband_tim(path: &Path) -> Result<Vec<WidebandDmPoint>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut points = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('C') || trimmed.starts_with("FORMAT") {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }
        let (Some(frequency_mhz), Some(mjd), Some(uncertainty_us), Some(pp_dm)) = (
            parse_f64(fields[1]),
            parse_f64(fields[2]),
            parse_f64(fields[3]),
            collect_numeric_flag_value(&fields, "-pp_dm"),
        ) else {
            continue;
        };
        let pp_dme = collect_numeric_flag_value(&fields, "-pp_dme");
        points.push(WidebandDmPoint {
            mjd,
            frequency_mhz,
            uncertainty_us,
            pp_dm,
            pp_dme,
        });
    }
    Ok(points)
}

fn collect_numeric_flag_value(fields: &[&str], flag: &str) -> Option<f64> {
    for window in fields.windows(2) {
        if window[0] == flag {
            return parse_f64(window[1]);
        }
    }
    None
}

fn parse_par_file(path: &Path) -> Result<TimingModelMetadata> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut meta = TimingModelMetadata {
        file_count: 1,
        ..TimingModelMetadata::default()
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 2 {
            continue;
        }
        match fields[0] {
            "START" => update_min(&mut meta.start_mjd, parse_f64(fields[1])),
            "FINISH" => update_max(&mut meta.finish_mjd, parse_f64(fields[1])),
            "DM" if fields.get(1).copied() != Some("DMDATA") => {
                set_if_none(&mut meta.dm, parse_f64(fields[1]));
            }
            "PX" => set_if_none(&mut meta.px_mas, parse_f64(fields[1])),
            "ELONG" => set_if_none(&mut meta.elong_deg, parse_f64(fields[1])),
            "ELAT" => set_if_none(&mut meta.elat_deg, parse_f64(fields[1])),
            "RAJ" => set_string_if_none(&mut meta.raj, Some(fields[1].to_string())),
            "DECJ" => set_string_if_none(&mut meta.decj, Some(fields[1].to_string())),
            _ => {}
        }
    }
    Ok(meta)
}

fn parse_dmx_file(path: &Path) -> Result<Vec<DmxPoint>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut points = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        let (Some(epoch_mjd), Some(dmx_value)) = (
            fields.first().and_then(|v| parse_f64(v)),
            fields.get(1).and_then(|v| parse_f64(v)),
        ) else {
            continue;
        };
        points.push(DmxPoint {
            epoch_mjd,
            dmx_value,
            dmx_error: fields.get(2).and_then(|v| parse_f64(v)),
            window_start_mjd: fields.get(3).and_then(|v| parse_f64(v)),
            window_end_mjd: fields.get(4).and_then(|v| parse_f64(v)),
            bin_label: fields.get(5).map(|value| (*value).to_string()),
        });
    }
    Ok(points)
}

fn parse_residual_file(path: &Path) -> Result<Vec<ResidualPoint>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let is_full = file_name_string(path)?.contains(".full.");
    let mut points = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        let (Some(mjd), Some(residual_us)) = (
            fields.first().and_then(|v| parse_f64(v)),
            fields.get(2).and_then(|v| parse_f64(v)),
        ) else {
            continue;
        };
        let frequency_mhz = fields.get(1).and_then(|v| parse_f64(v));
        let white_residual_us = if is_full {
            if fields.len() >= 7 {
                fields.get(3).and_then(|v| parse_f64(v))
            } else {
                None
            }
        } else if fields.len() >= 5 {
            fields.get(3).and_then(|v| parse_f64(v))
        } else {
            None
        };
        let uncertainty_us = if is_full {
            if fields.len() >= 7 {
                fields.get(4).and_then(|v| parse_f64(v))
            } else {
                fields.get(3).and_then(|v| parse_f64(v))
            }
        } else if fields.len() >= 5 {
            fields.get(4).and_then(|v| parse_f64(v))
        } else {
            fields.get(3).and_then(|v| parse_f64(v))
        };
        let chi2 = if is_full {
            if fields.len() >= 7 {
                fields.get(5).and_then(|v| parse_f64(v))
            } else {
                None
            }
        } else {
            None
        };
        let flag = if is_full {
            if fields.len() >= 7 {
                fields.get(6).map(|value| (*value).to_string())
            } else {
                None
            }
        } else {
            None
        };
        points.push(ResidualPoint {
            mjd,
            frequency_mhz,
            residual_us,
            white_residual_us,
            uncertainty_us,
            chi2,
            flag,
        });
    }
    Ok(points)
}

fn parse_f64(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|parsed| parsed.is_finite())
}

fn set_if_none(slot: &mut Option<f64>, value: Option<f64>) {
    if slot.is_none() {
        *slot = value;
    }
}

fn set_string_if_none(slot: &mut Option<String>, value: Option<String>) {
    if slot.is_none() {
        *slot = value;
    }
}

fn update_min(slot: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        match slot {
            Some(current) if value >= *current => {}
            _ => *slot = Some(value),
        }
    }
}

fn update_max(slot: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        match slot {
            Some(current) if value <= *current => {}
            _ => *slot = Some(value),
        }
    }
}

#[allow(dead_code)]
fn _debug_path(_path: PathBuf) {}

#[cfg(test)]
mod tests {
    use super::canonical_pulsar_name;

    #[test]
    fn canonicalizes_release_file_names() {
        assert_eq!(
            canonical_pulsar_name("J1713+0747_PINT_20230131.wb".to_string()),
            "J1713+0747"
        );
        assert_eq!(
            canonical_pulsar_name("J1713+0747_NG15yr_nb.full.res".to_string()),
            "J1713+0747"
        );
        assert_eq!(
            canonical_pulsar_name("J1903+0327ao_PINT_20230131.wb".to_string()),
            "J1903+0327"
        );
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PairAuditRow {
    pub pulsar_a: String,
    pub pulsar_b: String,
    pub hellings_downs: f64,
    pub overlap_bins: usize,
    pub avg_residual_pearson: f64,
}
