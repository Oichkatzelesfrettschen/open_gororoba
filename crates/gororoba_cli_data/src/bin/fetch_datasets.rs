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
//!   fetch-datasets --skip-existing=false        Force refresh existing files

use clap::{ArgAction, Parser};
use data_core::fetcher::{DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

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
    /// Accepts either `--skip-existing` or `--skip-existing=<true|false>`.
    #[arg(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    skip_existing: bool,
}

struct DatasetEntry {
    provider: Box<dyn DatasetProvider>,
    category: &'static str,
    pillar: &'static str,
    size_hint: &'static str,
}

struct NamedDatasetProvider<P> {
    name: &'static str,
    inner: P,
}

impl<P> NamedDatasetProvider<P> {
    fn new(name: &'static str, inner: P) -> Self {
        Self { name, inner }
    }
}

impl<P: DatasetProvider> DatasetProvider for NamedDatasetProvider<P> {
    fn name(&self) -> &str {
        self.name
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        self.inner.fetch(config)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        self.inner.is_cached(config)
    }
}

const VALID_PILLARS: &[&str] = &[
    "candle",
    "gravitational",
    "electromagnetic",
    "survey",
    "cmb",
    "solar",
    "geophysical",
    "materials",
];

fn build_registry() -> Vec<DatasetEntry> {
    use data_core::{catalogs::*, geophysical::*};

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
        DatasetEntry {
            provider: Box::new(things::ThingsTablesProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(things::ThingsPreferredCubesProvider),
            category: "astro",
            pillar: "survey",
            size_hint: "~20-30 GB",
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
        DatasetEntry {
            provider: Box::new(omni::OmniProvider {
                year_start: 2020,
                year_end: 2020,
            }),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~3 MB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "NASA OMNI2 Solar Wind + IMF (2016)",
                omni::OmniProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~3 MB",
        },
        DatasetEntry {
            provider: Box::new(ace_mag::AceMagProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "ACE MAG L2 Browse 16-sec (2016)",
                ace_mag::AceMagProvider {
                    year_start: 2016,
                    year_end: 2016,
                    doy_range: None,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 MB",
        },
        DatasetEntry {
            provider: Box::new(solar_wind::AceSwepamProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~2 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "ACE SWEPAM Solar Wind (2016)",
                solar_wind::AceSwepamProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~2 MB",
        },
        DatasetEntry {
            provider: Box::new(wind_swe::WindMfiProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~3 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "WIND MFI 1-hour Magnetic Field (2016)",
                wind_swe::WindMfiProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~3 MB",
        },
        DatasetEntry {
            provider: Box::new(wind_swe::WindSweProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~20 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "WIND SWE KP Unspiked Plasma (2016)",
                wind_swe::WindSweProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~20 MB",
        },
        DatasetEntry {
            provider: Box::new(stereo_plastic::StereoPlasticProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~50-100 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "STEREO-A PLASTIC 1-hour Plasma (2016)",
                stereo_plastic::StereoPlasticProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~50-100 MB",
        },
        DatasetEntry {
            provider: Box::new(stereo_plastic::StereoMagProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~directory placeholder",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "STEREO-A IMPACT/MAG MAGPLASMA (2016)",
                stereo_plastic::StereoMagProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~directory placeholder",
        },
        DatasetEntry {
            provider: Box::new(voyager::VoyagerProvider {
                spacecraft: voyager::VoyagerSpacecraft::V1,
                year_start: 2020,
                year_end: 2020,
            }),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Voyager 1 Merged Hourly (2016)",
                voyager::VoyagerProvider {
                    spacecraft: voyager::VoyagerSpacecraft::V1,
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(voyager::VoyagerProvider {
                spacecraft: voyager::VoyagerSpacecraft::V2,
                year_start: 2020,
                year_end: 2020,
            }),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Voyager 2 Merged Hourly (2016)",
                voyager::VoyagerProvider {
                    spacecraft: voyager::VoyagerSpacecraft::V2,
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~100 KB",
        },
        DatasetEntry {
            provider: Box::new(voyager_crs_flux::VoyagerCrsFluxProvider {
                spacecraft: 1,
                year_start: 2020,
                year_end: 2020,
            }),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Voyager 1 CRS Daily Flux (2016)",
                voyager_crs_flux::VoyagerCrsFluxProvider {
                    spacecraft: 1,
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(voyager_crs_flux::VoyagerCrsFluxProvider {
                spacecraft: 2,
                year_start: 2020,
                year_end: 2020,
            }),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Voyager 2 CRS Daily Flux (2016)",
                voyager_crs_flux::VoyagerCrsFluxProvider {
                    spacecraft: 2,
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(ulysses::UlyssesProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB/year",
        },
        DatasetEntry {
            provider: Box::new(helios::HeliosProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Helios 2 Merged Hourly",
                helios::HeliosProvider {
                    spacecraft: helios::HeliosSpacecraft::H2,
                    year_start: 1976,
                    year_end: 1980,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB/year",
        },
        DatasetEntry {
            provider: Box::new(cassini::CassiniCruiseProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-10 MB",
        },
        DatasetEntry {
            provider: Box::new(juno::JunoCruiseProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-10 MB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Juno Cruise Merged Hourly (2016)",
                juno::JunoCruiseProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(new_horizons::NhSwapProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-10 MB",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "New Horizons SWAP Hourly (2016)",
                new_horizons::NhSwapProvider {
                    year_start: 2016,
                    year_end: 2016,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1 MB",
        },
        DatasetEntry {
            provider: Box::new(ibex::IbexProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1-5 MB",
        },
        DatasetEntry {
            provider: Box::new(ibex::IbexOrbitProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-20 MB/year",
        },
        DatasetEntry {
            provider: Box::new(soho_celias::SohoCeliasBundleProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~250 MB",
        },
        DatasetEntry {
            provider: Box::new(soho_celias::SohoCeliasPm5MinProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~15 MB/year",
        },
        DatasetEntry {
            provider: Box::new(soho_celias::SohoLascoDaySampleProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~10-50 MB",
        },
        DatasetEntry {
            provider: Box::new(imap::ImapHelio1hrProvider),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~250 KB",
        },
        DatasetEntry {
            provider: Box::new(imap::ImapHiL2H90Provider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~2 MB/year",
        },
        DatasetEntry {
            provider: Box::new(psp::PspProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-20 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Parker Solar Probe Merged Hourly (2020)",
                psp::PspProvider {
                    year_start: 2020,
                    year_end: 2020,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-20 MB",
        },
        DatasetEntry {
            provider: Box::new(solar_orbiter::SolarOrbiterProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-20 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "Solar Orbiter Merged Hourly (2020)",
                solar_orbiter::SolarOrbiterProvider {
                    year_start: 2020,
                    year_end: 2020,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~5-20 MB",
        },
        DatasetEntry {
            provider: Box::new(bepicolombo::BepicolomboProvider::default()),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1-5 MB/year",
        },
        DatasetEntry {
            provider: Box::new(NamedDatasetProvider::new(
                "BepiColombo Position Hourly (2020)",
                bepicolombo::BepicolomboProvider {
                    year_start: 2020,
                    year_end: 2020,
                },
            )),
            category: "geophysical",
            pillar: "geophysical",
            size_hint: "~1-5 MB",
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
        // -- Electromagnetic pillar: technosignature research --
        DatasetEntry {
            provider: Box::new(wow::WowPrintoutProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~500 KB",
        },
        DatasetEntry {
            provider: Box::new(wow::Bl6equj5ManifestProvider),
            category: "astro",
            pillar: "electromagnetic",
            size_hint: "~5 KB",
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

fn staged_file_count(path: &Path) -> usize {
    if path.is_file() {
        return 1;
    }
    if !path.is_dir() {
        return 0;
    }
    let mut total = 0usize;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_file() {
                total += 1;
            } else if child.is_dir() {
                stack.push(child);
            }
        }
    }
    total
}

fn main() {
    env_logger::init();
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
    if let Some(ref p) = args.pillar
        && !VALID_PILLARS.contains(&p.as_str())
    {
        eprintln!(
            "Error: unknown pillar '{}'. Valid pillars: {}",
            p,
            VALID_PILLARS.join(", ")
        );
        std::process::exit(1);
    }

    let targets: Vec<&DatasetEntry> = if args.all {
        registry.iter().collect()
    } else if args.pillar.is_some() || args.category.is_some() || args.dataset.is_some() {
        let dataset_name_filter = args.dataset.as_ref().map(|s| s.to_lowercase());
        registry
            .iter()
            .filter(|entry| {
                if let Some(ref p) = args.pillar
                    && entry.pillar != p
                {
                    return false;
                }
                if let Some(ref cat) = args.category
                    && entry.category != cat
                {
                    return false;
                }
                if let Some(ref name_filter) = dataset_name_filter
                    && !entry.provider.name().to_lowercase().contains(name_filter)
                {
                    return false;
                }
                true
            })
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
                let staged_files = staged_file_count(&path);
                if path.is_dir() && staged_files == 0 {
                    eprintln!(
                        "  FAILED: provider returned an empty staging directory {}",
                        path.display()
                    );
                    failed += 1;
                } else {
                    eprintln!(
                        "  OK: {} ({} staged files)",
                        path.display(),
                        staged_files.max(1)
                    );
                    success += 1;
                }
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
