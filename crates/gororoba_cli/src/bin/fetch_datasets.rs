//! Unified dataset fetcher for all astrophysical and geophysical catalogs.
//!
//! Usage:
//!   fetch-datasets --list                       Show all datasets with cache status
//!   fetch-datasets --all                        Fetch everything
//!   fetch-datasets --category astro             Fetch astrophysical datasets
//!   fetch-datasets --category cosmology         Fetch cosmological datasets
//!   fetch-datasets --category geophysical       Fetch geophysical datasets
//!   fetch-datasets --dataset "CHIME FRB Cat 2"  Fetch a specific dataset
//!   fetch-datasets --skip-existing              Honor cache (default: true)

use clap::Parser;
use data_core::fetcher::{DatasetProvider, FetchConfig};

#[derive(Parser, Debug)]
#[command(name = "fetch-datasets", about = "Unified dataset acquisition tool")]
struct Args {
    /// List all datasets with cache status.
    #[arg(long)]
    list: bool,

    /// Fetch all datasets.
    #[arg(long)]
    all: bool,

    /// Fetch datasets in a specific category (astro, cosmology, geophysical).
    #[arg(long)]
    category: Option<String>,

    /// Fetch a specific dataset by name (substring match).
    #[arg(long)]
    dataset: Option<String>,

    /// Output directory for downloaded data (default: data/external).
    #[arg(long, default_value = "data/external")]
    output_dir: String,

    /// Skip download if file already exists (default: true).
    #[arg(long, default_value_t = true)]
    skip_existing: bool,
}

struct DatasetEntry {
    provider: Box<dyn DatasetProvider>,
    category: &'static str,
    size_hint: &'static str,
}

fn build_registry() -> Vec<DatasetEntry> {
    use data_core::catalogs::*;
    use data_core::geophysical::*;

    vec![
        // Astrophysical catalogs
        DatasetEntry {
            provider: Box::new(chime::ChimeCat2Provider),
            category: "astro",
            size_hint: "~15 MB",
        },
        DatasetEntry {
            provider: Box::new(gwtc::Gwtc3Provider),
            category: "astro",
            size_hint: "~2 MB",
        },
        DatasetEntry {
            provider: Box::new(atnf::AtnfProvider),
            category: "astro",
            size_hint: "~5 MB",
        },
        DatasetEntry {
            provider: Box::new(mcgill::McgillProvider),
            category: "astro",
            size_hint: "~50 KB",
        },
        DatasetEntry {
            provider: Box::new(fermi_gbm::FermiGbmProvider),
            category: "astro",
            size_hint: "~10 MB",
        },
        DatasetEntry {
            provider: Box::new(nanograv::NanoGrav15yrProvider),
            category: "astro",
            size_hint: "~10 KB",
        },
        DatasetEntry {
            provider: Box::new(sdss::SdssQsoProvider),
            category: "astro",
            size_hint: "~20 MB",
        },
        DatasetEntry {
            provider: Box::new(gaia::GaiaDr3Provider),
            category: "astro",
            size_hint: "~15 MB",
        },
        DatasetEntry {
            provider: Box::new(tsi::TsisTsiProvider),
            category: "astro",
            size_hint: "~500 KB",
        },
        // Cosmological datasets
        DatasetEntry {
            provider: Box::new(pantheon::PantheonProvider),
            category: "cosmology",
            size_hint: "~200 KB",
        },
        DatasetEntry {
            provider: Box::new(planck::PlanckSummaryProvider),
            category: "cosmology",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(planck::Wmap9ChainsProvider),
            category: "cosmology",
            size_hint: "~100 MB",
        },
        DatasetEntry {
            provider: Box::new(planck::PlanckChainsProvider),
            category: "cosmology",
            size_hint: "~9 GB",
        },
        // Geophysical datasets
        DatasetEntry {
            provider: Box::new(igrf::Igrf13Provider),
            category: "geophysical",
            size_hint: "~30 KB",
        },
        DatasetEntry {
            provider: Box::new(wmm::Wmm2025Provider),
            category: "geophysical",
            size_hint: "~2 MB",
        },
        DatasetEntry {
            provider: Box::new(grace::GraceGgm05sProvider),
            category: "geophysical",
            size_hint: "~350 KB",
        },
        DatasetEntry {
            provider: Box::new(grail::GrailGrgm1200bProvider),
            category: "geophysical",
            size_hint: "~84 MB",
        },
        DatasetEntry {
            provider: Box::new(jpl_ephemeris::JplEphemerisProvider),
            category: "geophysical",
            size_hint: "~200 KB",
        },
    ]
}

fn main() {
    let args = Args::parse();
    let config = FetchConfig {
        output_dir: std::path::PathBuf::from(&args.output_dir),
        skip_existing: args.skip_existing,
        verify_checksums: true,
    };

    let registry = build_registry();

    if args.list {
        println!("{:<35} {:<15} {:<10} Cached", "Dataset", "Category", "Size");
        println!("{}", "-".repeat(75));
        for entry in &registry {
            let cached = if entry.provider.is_cached(&config) { "yes" } else { "no" };
            println!(
                "{:<35} {:<15} {:<10} {}",
                entry.provider.name(),
                entry.category,
                entry.size_hint,
                cached
            );
        }
        return;
    }

    let targets: Vec<&DatasetEntry> = if args.all {
        registry.iter().collect()
    } else if let Some(ref cat) = args.category {
        registry.iter().filter(|e| e.category == cat.as_str()).collect()
    } else if let Some(ref name) = args.dataset {
        let lower = name.to_lowercase();
        registry
            .iter()
            .filter(|e| e.provider.name().to_lowercase().contains(&lower))
            .collect()
    } else {
        eprintln!("No action specified. Use --list, --all, --category, or --dataset.");
        eprintln!("Run with --help for usage information.");
        std::process::exit(1);
    };

    if targets.is_empty() {
        eprintln!("No datasets matched the filter.");
        std::process::exit(1);
    }

    let mut success = 0;
    let mut failed = 0;

    for entry in &targets {
        eprintln!("[{}/{}] {}", success + failed + 1, targets.len(), entry.provider.name());
        match entry.provider.fetch(&config) {
            Ok(path) => {
                eprintln!("  OK: {}", path.display());
                success += 1;
            }
            Err(e) => {
                eprintln!("  FAILED: {}", e);
                failed += 1;
            }
        }
    }

    eprintln!("\nDone: {} succeeded, {} failed out of {} datasets.", success, failed, targets.len());
    if failed > 0 {
        std::process::exit(1);
    }
}
