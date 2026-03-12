use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use clap::Parser;
use data_core::catalogs::pioneer::{PioneerSpacecraft, parse_pioneer_file, pioneer_to_omni};
use gororoba_cli_physics::{
    anomaly_residual::{
        PioneerBenchmarkContextRecord, PioneerBenchmarkRecord, read_csv_records, write_csv_records,
    },
    heliosphere_boundary::{OmniBoundaryProvider, PlasmaBoundaryProvider},
};
use std::path::PathBuf;

const AU_KM: f64 = 149_597_870.7;
const PROTON_MASS_KG: f64 = 1.672_621_9e-27;

#[derive(Debug, Parser)]
#[command(name = "pioneer-residual-audit")]
#[command(about = "Generate governed Pioneer anomaly benchmark-vs-environment artifacts")]
struct Args {
    #[arg(
        long,
        default_value = "data/external/pioneer_anomaly/pioneer_anomaly_benchmark.csv"
    )]
    input: PathBuf,
    #[arg(
        long,
        default_value = "data/external/pioneer/pioneer10/pds_ppi_merged/p10_*.TAB"
    )]
    p10_glob: String,
    #[arg(
        long,
        default_value = "data/external/pioneer/pioneer11/pds_ppi_merged/p11_*.TAB"
    )]
    p11_glob: String,
    #[arg(
        long,
        default_value = "data/output/anomaly/pioneer/pioneer_benchmark_vs_environment.csv"
    )]
    out: PathBuf,
}

fn parse_utc(text: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(text)
        .with_context(|| format!("parsing RFC3339 timestamp {text}"))?
        .with_timezone(&Utc))
}

fn dynamic_pressure_npa(density_cm3: f64, speed_km_s: f64) -> f64 {
    let density_m3 = density_cm3 * 1.0e6;
    let speed_m_s = speed_km_s * 1.0e3;
    PROTON_MASS_KG * density_m3 * speed_m_s * speed_m_s * 1.0e9
}

fn bmag_from_components(components: [Option<f64>; 3]) -> Option<f64> {
    let [bx, by, bz] = components;
    match (bx, by, bz) {
        (Some(x), Some(y), Some(z)) => Some((x * x + y * y + z * z).sqrt()),
        _ => None,
    }
}

fn load_provider(
    pattern: &str,
    spacecraft: PioneerSpacecraft,
    label: &str,
) -> Result<OmniBoundaryProvider> {
    let mut paths: Vec<PathBuf> = glob::glob(pattern)
        .with_context(|| format!("expanding glob {pattern}"))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("collecting glob matches for {pattern}"))?;
    if paths.is_empty() {
        bail!("no Pioneer files matched {}", pattern);
    }
    paths.sort();

    let mut merged = Vec::new();
    for path in &paths {
        let records = parse_pioneer_file(path, spacecraft)
            .with_context(|| format!("parsing Pioneer file {}", path.display()))?;
        merged.extend(records);
    }
    let omni = pioneer_to_omni(&merged);
    OmniBoundaryProvider::new(label, omni, true, true)
}

fn sample_note(
    benchmark_start_ms: i64,
    benchmark_end_ms: i64,
    provider: &dyn PlasmaBoundaryProvider,
) -> (i64, bool, String) {
    let bounds = provider.time_bounds();
    let overlap_start = benchmark_start_ms.max(bounds.start_ms);
    let overlap_end = benchmark_end_ms.min(bounds.end_ms);
    if overlap_end < overlap_start {
        return (
            benchmark_start_ms,
            false,
            "no overlap between benchmark interval and staged provider coverage".to_string(),
        );
    }
    let full = overlap_start == benchmark_start_ms && overlap_end == benchmark_end_ms;
    let midpoint = overlap_start + (overlap_end - overlap_start) / 2;
    let note = if full {
        "provider covers full benchmark interval".to_string()
    } else {
        format!(
            "provider covers only partial benchmark interval: overlap {} to {}",
            DateTime::<Utc>::from_timestamp_millis(overlap_start)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| overlap_start.to_string()),
            DateTime::<Utc>::from_timestamp_millis(overlap_end)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_else(|| overlap_end.to_string())
        )
    };
    (midpoint, full, note)
}

fn provider_for_spacecraft<'a>(
    spacecraft: &str,
    p10: &'a OmniBoundaryProvider,
    p11: &'a OmniBoundaryProvider,
) -> Result<&'a OmniBoundaryProvider> {
    match spacecraft {
        "Pioneer 10" => Ok(p10),
        "Pioneer 11" => Ok(p11),
        other => bail!("unknown Pioneer benchmark spacecraft {}", other),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let benchmarks = read_csv_records::<PioneerBenchmarkRecord>(&args.input)?;
    let p10 = load_provider(
        &args.p10_glob,
        PioneerSpacecraft::P10,
        "Pioneer 10 annual merged PDS/PPI",
    )?;
    let p11 = load_provider(
        &args.p11_glob,
        PioneerSpacecraft::P11,
        "Pioneer 11 annual merged PDS/PPI",
    )?;

    let mut rows = Vec::new();
    for benchmark in &benchmarks {
        let provider = provider_for_spacecraft(&benchmark.spacecraft, &p10, &p11)?;
        let start = parse_utc(&benchmark.interval_start_utc)?;
        let end = parse_utc(&benchmark.interval_end_utc)?;
        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();
        let (sample_ms, provider_contract_satisfied, coverage_note) =
            sample_note(start_ms, end_ms, provider);
        let sample = provider.sample_at_unix_milliseconds(sample_ms)?;
        let sample_dt = DateTime::<Utc>::from_timestamp_millis(sample_ms)
            .ok_or_else(|| anyhow::anyhow!("invalid sample ms {}", sample_ms))?;

        let (sampled_r_au, sampled_speed_km_s, sampled_density_cm3, sampled_bmag_nt) =
            if let Some(state) = &sample {
                (
                    state.r_au,
                    state.bulk_speed_kms,
                    state.proton_density,
                    bmag_from_components(state.b_gse_nt),
                )
            } else {
                (None, None, None, None)
            };
        let dynamic_pressure = match (sampled_density_cm3, sampled_speed_km_s) {
            (Some(n), Some(v)) => Some(dynamic_pressure_npa(n, v)),
            _ => None,
        };
        let fractal_predicted = match (sampled_r_au, sampled_speed_km_s) {
            (Some(r_au), Some(v_km_s)) => Some(
                gr_core::fractal_metric::pioneer_anomaly_prediction(2.7, r_au * AU_KM, v_km_s)
                    * 1.0e3,
            ),
            _ => None,
        };
        let source_label = sample
            .as_ref()
            .map(|state| state.source_label.clone())
            .unwrap_or_else(|| provider.label().to_string());

        rows.push(PioneerBenchmarkContextRecord {
            spacecraft: benchmark.spacecraft.clone(),
            interval_start_utc: benchmark.interval_start_utc.clone(),
            interval_end_utc: benchmark.interval_end_utc.clone(),
            sample_time_utc: sample_dt.to_rfc3339(),
            observed_anomalous_accel_m_s2: benchmark.observed_anomalous_accel_m_s2,
            observed_sigma_m_s2: benchmark.observed_sigma_m_s2,
            sampled_r_au,
            sampled_speed_km_s,
            sampled_density_cm3,
            sampled_bmag_nt,
            dynamic_pressure_npa: dynamic_pressure,
            fractal_predicted_accel_m_s2_df27: fractal_predicted,
            provider_contract_satisfied,
            source_label,
            source_bib: benchmark.source_bib.clone(),
            benchmark_kind: benchmark.benchmark_kind.clone(),
            notes: format!("{}; {}", benchmark.notes, coverage_note),
        });
    }

    write_csv_records(&args.out, &rows)?;
    println!("artifact: {}", args.out.display());
    println!("rows: {}", rows.len());
    Ok(())
}
