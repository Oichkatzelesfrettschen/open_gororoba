use super::timing::canonical_pulsar_name;
use anyhow::{Context, Result};
use regex::Regex;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ReleaseBand {
    Narrowband,
    Wideband,
}

impl ReleaseBand {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Narrowband => "narrowband",
            Self::Wideband => "wideband",
        }
    }

    fn par_dir_name(self) -> &'static str {
        match self {
            Self::Narrowband => "narrowband/par",
            Self::Wideband => "wideband/par",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum BinaryFamily {
    Bt,
    Dd,
    Ddk,
    Ell1,
    Ell1h,
    Other(String),
}

impl BinaryFamily {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Bt => "BT",
            Self::Dd => "DD",
            Self::Ddk => "DDK",
            Self::Ell1 => "ELL1",
            Self::Ell1h => "ELL1H",
            Self::Other(value) => value.as_str(),
        }
    }

    fn from_token(value: &str) -> Self {
        match value {
            "BT" => Self::Bt,
            "DD" => Self::Dd,
            "DDK" => Self::Ddk,
            "ELL1" => Self::Ell1,
            "ELL1H" => Self::Ell1h,
            other => Self::Other(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectorTerm {
    pub flag: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterTerm {
    pub name: String,
    pub raw_value: String,
    pub value: Option<f64>,
    pub fit: Option<bool>,
    pub uncertainty: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaggedTerm {
    pub name: String,
    pub selectors: Vec<SelectorTerm>,
    pub raw_value: String,
    pub value: Option<f64>,
    pub fit: Option<bool>,
    pub uncertainty: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct AstrometryTerms {
    pub raj: Option<ParameterTerm>,
    pub decj: Option<ParameterTerm>,
    pub elong: Option<ParameterTerm>,
    pub elat: Option<ParameterTerm>,
    pub pmelong: Option<ParameterTerm>,
    pub pmelat: Option<ParameterTerm>,
    pub pmra: Option<ParameterTerm>,
    pub pmdec: Option<ParameterTerm>,
    pub px: Option<ParameterTerm>,
}

#[derive(Debug, Clone, Default)]
pub struct DmxWindow {
    pub label: String,
    pub dmx: Option<ParameterTerm>,
    pub start_mjd: Option<f64>,
    pub end_mjd: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct DispersionTerms {
    pub dm: Option<ParameterTerm>,
    pub dmepoch: Option<ParameterTerm>,
    pub dmx_step: Option<ParameterTerm>,
    pub dmx_windows: Vec<DmxWindow>,
}

#[derive(Debug, Clone)]
pub struct TimingModel {
    pub band: ReleaseBand,
    pub source_path: PathBuf,
    pub solution_id: String,
    pub pulsar_id: String,
    pub format: Option<String>,
    pub ephem: Option<String>,
    pub clk: Option<String>,
    pub units: Option<String>,
    pub timeeph: Option<String>,
    pub t2cmethod: Option<String>,
    pub ecl: Option<String>,
    pub start_mjd: Option<f64>,
    pub finish_mjd: Option<f64>,
    pub ntoa: Option<usize>,
    pub chi2: Option<f64>,
    pub binary_family: Option<BinaryFamily>,
    pub astrometry: AstrometryTerms,
    pub dispersion: DispersionTerms,
    pub spin_terms: Vec<ParameterTerm>,
    pub fd_terms: Vec<ParameterTerm>,
    pub jumps: Vec<TaggedTerm>,
    pub dmjumps: Vec<TaggedTerm>,
    pub noise_terms: Vec<TaggedTerm>,
    pub other_terms: Vec<ParameterTerm>,
}

impl TimingModel {
    pub fn observing_span_days(&self) -> Option<f64> {
        Some(self.finish_mjd? - self.start_mjd?)
    }

    pub fn fit_parameter_count(&self) -> usize {
        self.spin_terms
            .iter()
            .chain(self.fd_terms.iter())
            .chain(self.other_terms.iter())
            .filter(|term| term.fit == Some(true))
            .count()
            + self
                .jumps
                .iter()
                .filter(|term| term.fit == Some(true))
                .count()
            + self
                .dmjumps
                .iter()
                .filter(|term| term.fit == Some(true))
                .count()
            + self
                .noise_terms
                .iter()
                .filter(|term| term.fit == Some(true))
                .count()
            + usize::from(
                self.astrometry
                    .raj
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .decj
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .elong
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .elat
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .pmelong
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .pmelat
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .pmra
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .pmdec
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.astrometry
                    .px
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.dispersion
                    .dm
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.dispersion
                    .dmepoch
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + usize::from(
                self.dispersion
                    .dmx_step
                    .as_ref()
                    .is_some_and(|term| term.fit == Some(true)),
            )
            + self
                .dispersion
                .dmx_windows
                .iter()
                .filter(|window| {
                    window
                        .dmx
                        .as_ref()
                        .is_some_and(|term| term.fit == Some(true))
                })
                .count()
    }

    pub fn parameter_term(&self, name: &str) -> Option<&ParameterTerm> {
        let astrometry = [
            self.astrometry.raj.as_ref(),
            self.astrometry.decj.as_ref(),
            self.astrometry.elong.as_ref(),
            self.astrometry.elat.as_ref(),
            self.astrometry.pmelong.as_ref(),
            self.astrometry.pmelat.as_ref(),
            self.astrometry.pmra.as_ref(),
            self.astrometry.pmdec.as_ref(),
            self.astrometry.px.as_ref(),
            self.dispersion.dm.as_ref(),
            self.dispersion.dmepoch.as_ref(),
            self.dispersion.dmx_step.as_ref(),
        ];
        astrometry
            .into_iter()
            .flatten()
            .find(|term| term.name == name)
            .or_else(|| {
                self.dispersion
                    .dmx_windows
                    .iter()
                    .filter_map(|window| window.dmx.as_ref())
                    .find(|term| term.name == name)
            })
            .or_else(|| self.spin_terms.iter().find(|term| term.name == name))
            .or_else(|| self.fd_terms.iter().find(|term| term.name == name))
            .or_else(|| self.other_terms.iter().find(|term| term.name == name))
    }

    pub fn parameter_value(&self, name: &str) -> Option<f64> {
        self.parameter_term(name).and_then(|term| term.value)
    }

    pub fn parameter_bool(&self, name: &str) -> Option<bool> {
        let term = self.parameter_term(name)?;
        match term.raw_value.as_str() {
            "Y" | "y" | "T" | "t" | "true" | "True" | "1" => Some(true),
            "N" | "n" | "F" | "f" | "false" | "False" | "0" => Some(false),
            _ => term.value.map(|value| value != 0.0),
        }
    }
}

pub fn load_release_timing_models(root: &Path, band: ReleaseBand) -> Result<Vec<TimingModel>> {
    let par_dir = root.join(band.par_dir_name());
    let mut models = Vec::new();
    for entry in
        fs::read_dir(&par_dir).with_context(|| format!("read_dir {}", par_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("par") {
            continue;
        }
        models.push(parse_par_timing_model(&path, band)?);
    }
    models.sort_by(|left, right| {
        left.pulsar_id
            .cmp(&right.pulsar_id)
            .then_with(|| left.solution_id.cmp(&right.solution_id))
    });
    Ok(models)
}

pub fn parse_par_timing_model(path: &Path, band: ReleaseBand) -> Result<TimingModel> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    parse_par_timing_model_text(path, band, &content)
}

fn parse_par_timing_model_text(
    path: &Path,
    band: ReleaseBand,
    content: &str,
) -> Result<TimingModel> {
    let solution_id = path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing UTF-8 file stem for {}", path.display()))?;
    let pulsar_id = canonical_pulsar_name(solution_id.clone());

    let spin_pattern = Regex::new(r"^F\d+$").expect("spin regex");
    let fd_pattern = Regex::new(r"^FD\d+$").expect("fd regex");
    let dmx_pattern = Regex::new(r"^DMX_(\d+)$").expect("dmx regex");
    let dmxr1_pattern = Regex::new(r"^DMXR1_(\d+)$").expect("dmxr1 regex");
    let dmxr2_pattern = Regex::new(r"^DMXR2_(\d+)$").expect("dmxr2 regex");

    let mut model = TimingModel {
        band,
        source_path: path.to_path_buf(),
        solution_id,
        pulsar_id,
        format: None,
        ephem: None,
        clk: None,
        units: None,
        timeeph: None,
        t2cmethod: None,
        ecl: None,
        start_mjd: None,
        finish_mjd: None,
        ntoa: None,
        chi2: None,
        binary_family: None,
        astrometry: AstrometryTerms::default(),
        dispersion: DispersionTerms::default(),
        spin_terms: Vec::new(),
        fd_terms: Vec::new(),
        jumps: Vec::new(),
        dmjumps: Vec::new(),
        noise_terms: Vec::new(),
        other_terms: Vec::new(),
    };
    let mut dmx_windows: BTreeMap<String, DmxWindow> = BTreeMap::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 2 {
            continue;
        }
        let name = fields[0];
        match name {
            "FORMAT" => model.format = Some(fields[1].to_string()),
            "EPHEM" => model.ephem = Some(fields[1].to_string()),
            "CLK" => model.clk = Some(fields[1].to_string()),
            "UNITS" => model.units = Some(fields[1].to_string()),
            "TIMEEPH" => model.timeeph = Some(fields[1].to_string()),
            "T2CMETHOD" => model.t2cmethod = Some(fields[1].to_string()),
            "ECL" => model.ecl = Some(fields[1].to_string()),
            "START" => model.start_mjd = parse_f64(fields[1]),
            "FINISH" => model.finish_mjd = parse_f64(fields[1]),
            "NTOA" => model.ntoa = parse_usize(fields[1]),
            "CHI2" => model.chi2 = parse_f64(fields[1]),
            "BINARY" => model.binary_family = Some(BinaryFamily::from_token(fields[1])),
            "RAJ" => model.astrometry.raj = Some(parse_parameter_term(&fields)),
            "DECJ" => model.astrometry.decj = Some(parse_parameter_term(&fields)),
            "ELONG" => model.astrometry.elong = Some(parse_parameter_term(&fields)),
            "ELAT" => model.astrometry.elat = Some(parse_parameter_term(&fields)),
            "PMELONG" => model.astrometry.pmelong = Some(parse_parameter_term(&fields)),
            "PMELAT" => model.astrometry.pmelat = Some(parse_parameter_term(&fields)),
            "PMRA" => model.astrometry.pmra = Some(parse_parameter_term(&fields)),
            "PMDEC" => model.astrometry.pmdec = Some(parse_parameter_term(&fields)),
            "PX" => model.astrometry.px = Some(parse_parameter_term(&fields)),
            "DM" if fields.get(1).copied() != Some("DMDATA") => {
                model.dispersion.dm = Some(parse_parameter_term(&fields));
            }
            "DMEPOCH" => model.dispersion.dmepoch = Some(parse_parameter_term(&fields)),
            "DMX" => model.dispersion.dmx_step = Some(parse_parameter_term(&fields)),
            "JUMP" => model.jumps.push(parse_tagged_term(&fields)),
            "DMJUMP" => model.dmjumps.push(parse_tagged_term(&fields)),
            "EFAC" | "EQUAD" | "ECORR" | "TNEF" | "TNEQ" | "DMEFAC" | "DMEQUAD" => {
                model.noise_terms.push(parse_tagged_term(&fields));
            }
            _ if spin_pattern.is_match(name) => {
                model.spin_terms.push(parse_parameter_term(&fields))
            }
            _ if fd_pattern.is_match(name) => model.fd_terms.push(parse_parameter_term(&fields)),
            _ if dmx_pattern.is_match(name) => {
                let label = dmx_pattern
                    .captures(name)
                    .and_then(|captures| captures.get(1))
                    .map(|capture| capture.as_str().to_string())
                    .expect("dmx label");
                let window = dmx_windows
                    .entry(label.clone())
                    .or_insert_with(|| DmxWindow {
                        label,
                        ..DmxWindow::default()
                    });
                window.dmx = Some(parse_parameter_term(&fields));
            }
            _ if dmxr1_pattern.is_match(name) => {
                let label = dmxr1_pattern
                    .captures(name)
                    .and_then(|captures| captures.get(1))
                    .map(|capture| capture.as_str().to_string())
                    .expect("dmxr1 label");
                let window = dmx_windows
                    .entry(label.clone())
                    .or_insert_with(|| DmxWindow {
                        label,
                        ..DmxWindow::default()
                    });
                window.start_mjd = parse_f64(fields[1]);
            }
            _ if dmxr2_pattern.is_match(name) => {
                let label = dmxr2_pattern
                    .captures(name)
                    .and_then(|captures| captures.get(1))
                    .map(|capture| capture.as_str().to_string())
                    .expect("dmxr2 label");
                let window = dmx_windows
                    .entry(label.clone())
                    .or_insert_with(|| DmxWindow {
                        label,
                        ..DmxWindow::default()
                    });
                window.end_mjd = parse_f64(fields[1]);
            }
            _ => model.other_terms.push(parse_parameter_term(&fields)),
        }
    }

    model.dispersion.dmx_windows = dmx_windows.into_values().collect();
    Ok(model)
}

fn parse_parameter_term(fields: &[&str]) -> ParameterTerm {
    ParameterTerm {
        name: fields[0].to_string(),
        raw_value: fields.get(1).unwrap_or(&"").to_string(),
        value: fields.get(1).and_then(|value| parse_f64(value)),
        fit: fields.get(2).and_then(|value| parse_fit_flag(value)),
        uncertainty: fields.get(3).and_then(|value| parse_f64(value)),
    }
}

fn parse_tagged_term(fields: &[&str]) -> TaggedTerm {
    let mut selectors = Vec::new();
    let mut index = 1usize;
    while fields.len().saturating_sub(index) > 3 && fields[index].starts_with('-') {
        if let Some(value) = fields.get(index + 1) {
            selectors.push(SelectorTerm {
                flag: fields[index].to_string(),
                value: (*value).to_string(),
            });
            index += 2;
        } else {
            break;
        }
    }
    TaggedTerm {
        name: fields[0].to_string(),
        selectors,
        raw_value: fields.get(index).unwrap_or(&"").to_string(),
        value: fields.get(index).and_then(|value| parse_f64(value)),
        fit: fields
            .get(index + 1)
            .and_then(|value| parse_fit_flag(value)),
        uncertainty: fields.get(index + 2).and_then(|value| parse_f64(value)),
    }
}

fn parse_f64(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|parsed| parsed.is_finite())
}

fn parse_usize(value: &str) -> Option<usize> {
    value.parse::<usize>().ok()
}

fn parse_fit_flag(value: &str) -> Option<bool> {
    match value {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryFamily, ReleaseBand, parse_par_timing_model_text};
    use std::path::Path;

    #[test]
    fn parses_timing_model_groups() {
        let content = "\
PSR J1713+0747\n\
FORMAT pint\n\
EPHEM DE440\n\
CLK TT(BIPM2019)\n\
START 54882.0\n\
FINISH 59056.0\n\
NTOA 123\n\
CHI2 456.5\n\
RAJ 17:13:49.532 1 0.1\n\
DECJ +07:47:37.49 1 0.2\n\
ELONG 100.0 1 0.01\n\
ELAT 20.0 1 0.02\n\
PX 0.3 1 0.03\n\
F0 218.0 1 1e-12\n\
F1 -1.0e-15 1 2e-20\n\
DM 15.9\n\
DMEPOCH 55000.0\n\
DMX 0.5\n\
DMX_0001 0.001 1 0.0001\n\
DMXR1_0001 55000.0\n\
DMXR2_0001 55100.0\n\
JUMP -fe Rcvr1_2 -h pks 1.2e-05 1 3e-07\n\
DMJUMP -fe Rcvr1_2 0.001 1 0.0001\n\
EFAC -f Rcvr1_2_GUPPI 1.05\n\
FD1 1.0e-05 1 1e-06\n\
BINARY DDK\n";
        let model = parse_par_timing_model_text(
            Path::new("J1713+0747_PINT_20230131.wb.par"),
            ReleaseBand::Wideband,
            content,
        )
        .expect("parse model");

        assert_eq!(model.pulsar_id, "J1713+0747");
        assert_eq!(model.solution_id, "J1713+0747_PINT_20230131.wb");
        assert_eq!(model.binary_family, Some(BinaryFamily::Ddk));
        assert_eq!(model.ntoa, Some(123));
        assert_eq!(model.spin_terms.len(), 2);
        assert_eq!(model.fd_terms.len(), 1);
        assert_eq!(model.jumps.len(), 1);
        assert_eq!(model.dmjumps.len(), 1);
        assert_eq!(model.noise_terms.len(), 1);
        assert_eq!(model.dispersion.dmx_windows.len(), 1);
        assert_eq!(model.dispersion.dmx_windows[0].start_mjd, Some(55000.0));
        assert_eq!(model.dispersion.dmx_windows[0].end_mjd, Some(55100.0));
        assert_eq!(model.fit_parameter_count(), 11);
    }
}
