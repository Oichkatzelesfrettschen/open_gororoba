use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(name = "ultrametric-core-extract")]
#[command(about = "Pure-Rust extractor for ultrametric hierarchy cores")]
struct Cli {
    /// Exploration CSV emitted by multi-dataset-ultrametric --explore
    #[arg(
        long,
        default_value = "data/csv/c071g_exploration_gpu_10M_1000perm.csv"
    )]
    input: PathBuf,

    /// Machine-readable TOML report path
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ExploreRow {
    dataset: String,
    subset: String,
    significant: bool,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct DatasetCoreReport {
    dataset: String,
    significant_subset_count: usize,
    core_count: usize,
    core_subsets: Vec<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct UltrametricCoreReport {
    generated_at_utc: String,
    input_csv_path: String,
    significant_row_count: usize,
    dataset_count: usize,
    total_core_count: usize,
    datasets: Vec<DatasetCoreReport>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let output = cli.output.unwrap_or_else(default_output_path);
    let report = build_report(&cli.input)?;
    write_report(&output, &report)?;

    println!("Datasets with cores: {}", report.dataset_count);
    println!("Total core subsets:  {}", report.total_core_count);
    println!("Report:              {}", output.display());
    for dataset in &report.datasets {
        println!(
            "{}: {} cores from {} significant subsets",
            dataset.dataset, dataset.core_count, dataset.significant_subset_count
        );
    }
    Ok(())
}

fn build_report(path: &Path) -> Result<UltrametricCoreReport> {
    let rows = load_rows(path)?;
    let significant_row_count = rows.iter().filter(|row| row.significant).count();

    let mut subsets_by_dataset: BTreeMap<String, BTreeSet<BTreeSet<String>>> = BTreeMap::new();
    for row in rows.into_iter().filter(|row| row.significant) {
        let subset = row
            .subset
            .split('+')
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(ToOwned::to_owned)
            .collect::<BTreeSet<_>>();
        if subset.is_empty() {
            continue;
        }
        subsets_by_dataset
            .entry(row.dataset)
            .or_default()
            .insert(subset);
    }

    let mut datasets = Vec::new();
    let mut total_core_count = 0usize;
    for (dataset, subsets) in subsets_by_dataset {
        let subset_vec = subsets.into_iter().collect::<Vec<_>>();
        let mut cores = Vec::new();
        for subset in &subset_vec {
            let is_core = !subset_vec
                .iter()
                .any(|other| other.len() < subset.len() && other.is_subset(subset));
            if is_core {
                cores.push(subset.iter().cloned().collect::<Vec<_>>());
            }
        }
        cores.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
        total_core_count += cores.len();
        datasets.push(DatasetCoreReport {
            dataset,
            significant_subset_count: subset_vec.len(),
            core_count: cores.len(),
            core_subsets: cores,
        });
    }

    Ok(UltrametricCoreReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        input_csv_path: path.display().to_string(),
        significant_row_count,
        dataset_count: datasets.len(),
        total_core_count,
        datasets,
    })
}

fn load_rows(path: &Path) -> Result<Vec<ExploreRow>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open exploration CSV {}", path.display()))?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row.with_context(|| format!("parse exploration CSV {}", path.display()))?);
    }
    Ok(rows)
}

fn write_report(path: &Path, report: &UltrametricCoreReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string_pretty(report)?)?;
    Ok(())
}

fn default_output_path() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "ultrametric_core_report_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn extracts_minimal_cores_only() {
        let csv = "\
dataset,subset,metric,n_objects,value,null_mean,effect_size,raw_p,adj_p,significant
Demo,a+b,um_fraction_eps05,10,0,0,0,0.001,0.01,true
Demo,a+b,tolerance_auc,10,0,0,0,0.001,0.01,true
Demo,a+b+c,um_fraction_eps05,10,0,0,0,0.001,0.01,true
Demo,b+c,um_fraction_eps05,10,0,0,0,0.001,0.01,true
Other,x+y,um_fraction_eps05,10,0,0,0,0.001,0.01,true
Other,x+y+z,um_fraction_eps05,10,0,0,0,0.001,0.01,false
";
        let file = write_temp_csv(csv);
        let report = build_report(file.path()).unwrap();
        assert_eq!(report.dataset_count, 2);
        assert_eq!(report.total_core_count, 3);
        assert_eq!(
            report.datasets[0],
            DatasetCoreReport {
                dataset: "Demo".to_string(),
                significant_subset_count: 3,
                core_count: 2,
                core_subsets: vec![
                    vec!["a".to_string(), "b".to_string()],
                    vec!["b".to_string(), "c".to_string()],
                ],
            }
        );
        assert_eq!(report.datasets[1].dataset, "Other");
        assert_eq!(
            report.datasets[1].core_subsets,
            vec![vec!["x".to_string(), "y".to_string()]]
        );
    }
}
