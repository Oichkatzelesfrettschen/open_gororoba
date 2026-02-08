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

    /// Fetch datasets in a scientific pillar:
    ///   candle       Standard candles/rulers (Pantheon+, Union3, DESI BAO)
    ///   gravitational  GW events + PTA (GWTC, NANOGrav)
    ///   electromagnetic  EM transients + imaging (Fermi GBM, EHT)
    ///   survey       Multi-object surveys (Gaia, SDSS, ATNF, CHIME, etc.)
    ///   cmb          CMB/WMAP chains and parameters (Planck, WMAP)
    ///   solar        Solar irradiance (TSIS, SORCE)
    ///   geophysical  Gravity + magnetic field models (IGRF, WMM, GRACE, etc.)
    ///   materials    Condensed-matter DFT databases (JARVIS, AFLOW)
    #[arg(long)]
    pillar: Option<String>,

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
    pillar: &'static str,
    size_hint: &'static str,
}

const VALID_PILLARS: &[&str] = &[
    "candle", "gravitational", "electromagnetic", "survey", "cmb", "solar", "geophysical",
    "materials",
];

fn build_registry() -> Vec<DatasetEntry> {
    use data_core::catalogs::*;
    use data_core::geophysical::*;

    vec![
        // -- Survey pillar: multi-object catalogs --
        DatasetEntry {
            provider: Box::new(chime::ChimeCat2Provider),
            category: "astro",
            pillar: "survey",
            size_hint: "~15 MB",
        },
        DatasetEntry {
            provider: Box::new(atnf::AtnfProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~5 MB",
        },
        DatasetEntry {
            provider: Box::new(mcgill::McgillProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~50 KB",
        },
        DatasetEntry {
            provider: Box::new(sdss::SdssQsoProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~20 MB",
        },
        DatasetEntry {
            provider: Box::new(gaia::GaiaDr3Provider),
            category: "astro",
            pillar: "survey",
            size_hint: "~15 MB",
        },
        DatasetEntry {
            provider: Box::new(hipparcos::HipparcosProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~35 MB",
        },
        // -- Gravitational pillar: GW events + PTA --
        DatasetEntry {
            provider: Box::new(gwtc::Gwtc3Provider),
            category: "astro",
            pillar: "gravitational",
            size_hint: "~2 MB",
        },
        DatasetEntry {
            provider: Box::new(gwtc::GwoscCombinedProvider),
            category: "astro",
            pillar: "gravitational",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(nanograv::NanoGrav15yrProvider),
            category: "astro",
            pillar: "gravitational",
            size_hint: "~10 KB",
        },
        // -- Electromagnetic pillar: EM transients + imaging --
        DatasetEntry {
            provider: Box::new(fermi_gbm::FermiGbmProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~10 MB",
        },
        DatasetEntry {
            provider: Box::new(eht::EhtM87_2017Provider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~5 MB (CSV+UVFITS+TXT)",
        },
        DatasetEntry {
            provider: Box::new(eht::EhtM87Provider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~8 MB (CSV+UVFITS+TXT)",
        },
        DatasetEntry {
            provider: Box::new(eht::EhtSgrAProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~21 MB (CSV+UVFITS+TXT)",
        },
        DatasetEntry {
            provider: Box::new(eht::Eht3c279Provider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~5 MB (CSV+UVFITS+TXT)",
        },
        DatasetEntry {
            provider: Box::new(eht::EhtCenAProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~3 MB (CSV+UVFITS+TXT)",
        },
        DatasetEntry {
            provider: Box::new(eht::EhtM87LegacyProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~24 KB",
        },
        // -- Solar pillar: irradiance --
        DatasetEntry {
            provider: Box::new(tsi::TsisTsiProvider),
            category: "astro",
            pillar: "solar",
            size_hint: "~500 KB",
        },
        DatasetEntry {
            provider: Box::new(sorce::SorceTsiProvider),
            category: "astro",
            pillar: "solar",
            size_hint: "~2 MB",
        },
        // -- Candle pillar: standard candles/rulers --
        DatasetEntry {
            provider: Box::new(pantheon::PantheonProvider),
            category: "cosmology",
            pillar: "candle",
            size_hint: "~200 KB",
        },
        DatasetEntry {
            provider: Box::new(union3::Union3Provider),
            category: "cosmology",
            pillar: "candle",
            size_hint: "~15 MB",
        },
        // -- CMB pillar: CMB parameter chains --
        DatasetEntry {
            provider: Box::new(planck::PlanckSummaryProvider),
            category: "cosmology",
            pillar: "cmb",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(planck::Wmap9ChainsProvider),
            category: "cosmology",
            pillar: "cmb",
            size_hint: "~100 MB",
        },
        DatasetEntry {
            provider: Box::new(planck::PlanckChainsProvider),
            category: "cosmology",
            pillar: "cmb",
            size_hint: "~9 GB",
        },
        // -- Geophysical pillar: gravity + magnetic field models --
        DatasetEntry {
            provider: Box::new(igrf::Igrf13Provider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~30 KB",
        },
        DatasetEntry {
            provider: Box::new(wmm::Wmm2025Provider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~2 MB",
        },
        DatasetEntry {
            provider: Box::new(grace::GraceGgm05sProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~350 KB",
        },
        DatasetEntry {
            provider: Box::new(grace_fo::GraceFoProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~3 MB",
        },
        DatasetEntry {
            provider: Box::new(grail::GrailGrgm1200bProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~84 MB",
        },
        DatasetEntry {
            provider: Box::new(egm2008::Egm2008Provider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~75 MB",
        },
        DatasetEntry {
            provider: Box::new(swarm::SwarmMagAProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(landsat::LandsatStacProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(de_ephemeris::De440Provider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~120 MB",
        },
        DatasetEntry {
            provider: Box::new(de_ephemeris::De441Provider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~2.6 GB",
        },
        DatasetEntry {
            provider: Box::new(jpl_ephemeris::JplEphemerisProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~200 KB",
        },
        // -- Materials pillar: condensed-matter DFT databases --
        DatasetEntry {
            provider: Box::new(jarvis::JarvisProvider),
            category: "materials",
            pillar: "materials",
            size_hint: "~50 MB",
        },
        DatasetEntry {
            provider: Box::new(aflow::AflowProvider),
            category: "materials",
            pillar: "materials",
            size_hint: "~500 MB",
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
        println!(
            "{:<35} {:<15} {:<16} {:<10} Cached",
            "Dataset", "Category", "Pillar", "Size"
        );
        println!("{}", "-".repeat(90));
        for entry in &registry {
            let cached = if entry.provider.is_cached(&config) {
                "yes"
            } else {
                "no"
            };
            println!(
                "{:<35} {:<15} {:<16} {:<10} {}",
                entry.provider.name(),
                entry.category,
                entry.pillar,
                entry.size_hint,
                cached
            );
        }
        return;
    }

    // Validate --pillar value
    if let Some(ref p) = args.pillar {
        if !VALID_PILLARS.contains(&p.as_str()) {
            eprintln!(
                "Error: unknown pillar '{}'. Valid pillars: {}",
                p,
                VALID_PILLARS.join(", ")
            );
            std::process::exit(1);
        }
    }

    let targets: Vec<&DatasetEntry> = if args.all {
        registry.iter().collect()
    } else if let Some(ref p) = args.pillar {
        registry
            .iter()
            .filter(|e| e.pillar == p.as_str())
            .collect()
    } else if let Some(ref cat) = args.category {
        registry
            .iter()
            .filter(|e| e.category == cat.as_str())
            .collect()
    } else if let Some(ref name) = args.dataset {
        let lower = name.to_lowercase();
        registry
            .iter()
            .filter(|e| e.provider.name().to_lowercase().contains(&lower))
            .collect()
    } else {
        eprintln!("No action specified. Use --list, --all, --category, --pillar, or --dataset.");
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
        eprintln!(
            "[{}/{}] {}",
            success + failed + 1,
            targets.len(),
            entry.provider.name()
        );
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

    eprintln!(
        "\nDone: {} succeeded, {} failed out of {} datasets.",
        success,
        failed,
        targets.len()
    );
    if failed > 0 {
        std::process::exit(1);
    }
}
