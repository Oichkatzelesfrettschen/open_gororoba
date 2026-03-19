//! Fetch and verify all dataset providers in a single invocation.
//!
//! Iterates every registered DatasetProvider, reports cache status, and
//! optionally downloads missing datasets.

use clap::{Parser, Subcommand};
use data_core::fetcher::{DatasetProvider, FetchConfig};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dataset-fetch-all", about = "Fetch all registered datasets")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,

    /// Output directory for downloaded data.
    #[arg(long, default_value = "data/external")]
    output_dir: PathBuf,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show cache status for every registered provider.
    Status,
    /// Download all missing datasets.
    Fetch,
    /// Verify integrity of all cached datasets.
    Verify,
}

fn all_providers() -> Vec<Box<dyn DatasetProvider>> {
    use data_core::{catalogs::*, geophysical::*};

    vec![
        // Astrophysical catalogs
        Box::new(chime::ChimeCat1Provider),
        Box::new(chime::ChimeCat2Provider),
        Box::new(gwtc::Gwtc3Provider),
        Box::new(gwtc::GwoscCombinedProvider),
        Box::new(atnf::AtnfProvider),
        Box::new(pantheon::PantheonProvider),
        Box::new(union3::Union3Provider),
        Box::new(sdss::SdssQsoProvider),
        Box::new(desi_bao::DesiBaoProvider),
        Box::new(nanograv::NanoGrav15yrProvider),
        Box::new(nanograv::NanoGrav15yrTimingProvider),
        Box::new(planck::PlanckChainsProvider),
        Box::new(planck::Wmap9ChainsProvider),
        Box::new(planck::PlanckSummaryProvider),
        Box::new(fermi_gbm::FermiGbmProvider),
        Box::new(gaia::GaiaDr3Provider),
        Box::new(hipparcos::HipparcosProvider),
        Box::new(mcgill::McgillProvider),
        // EHT VLBI (6 targets)
        Box::new(eht::EhtM87_2017Provider),
        Box::new(eht::EhtM87Provider),
        Box::new(eht::EhtSgrAProvider),
        Box::new(eht::Eht3c279Provider),
        Box::new(eht::EhtCenAProvider),
        Box::new(eht::EhtM87LegacyProvider),
        // Solar / TSI
        Box::new(tsi::TsisTsiProvider),
        Box::new(sorce::SorceTsiProvider),
        // Earth observation
        Box::new(landsat::LandsatStacProvider),
        // Geophysical
        Box::new(swarm::SwarmMagAProvider),
        Box::new(de_ephemeris::De440Provider),
        Box::new(de_ephemeris::De441Provider),
        Box::new(egm2008::Egm2008Provider),
        Box::new(grace::GraceGgm05sProvider),
        Box::new(grace_fo::GraceFoProvider),
        Box::new(grail::GrailGrgm1200bProvider),
        Box::new(igrf::Igrf13Provider),
        Box::new(wmm::Wmm2025Provider),
        Box::new(jpl_ephemeris::JplEphemerisProvider),
        // Materials
        Box::new(jarvis::JarvisProvider),
        Box::new(aflow::AflowProvider),
        // Wow! Signal
        Box::new(wow::WowPrintoutProvider),
        Box::new(wow::Bl6equj5ManifestProvider),
        // Heavy-ion R_AA (HEPData)
        Box::new(hic_raa::HicRaaProvider),
    ]
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    let config = FetchConfig {
        output_dir: cli.output_dir.clone(),
        skip_existing: true,
        verify_checksums: true,
    };

    let providers = all_providers();

    match cli.cmd {
        Cmd::Status => {
            println!(
                "=== Dataset Cache Status ({} providers) ===",
                providers.len()
            );
            let mut cached = 0usize;
            let mut missing = 0usize;
            for p in &providers {
                let status = if p.is_cached(&config) {
                    cached += 1;
                    "CACHED"
                } else {
                    missing += 1;
                    "MISSING"
                };
                println!("  [{status:^7}] {}", p.name());
            }
            println!("---");
            println!(
                "Cached: {cached}, Missing: {missing}, Total: {}",
                providers.len()
            );
        }
        Cmd::Fetch => {
            println!(
                "=== Fetching All Datasets ({} providers) ===",
                providers.len()
            );
            let mut ok = 0usize;
            let mut fail = 0usize;
            for p in &providers {
                if p.is_cached(&config) {
                    println!("  [SKIP] {} (already cached)", p.name());
                    ok += 1;
                    continue;
                }
                print!("  [FETCH] {} ... ", p.name());
                match p.fetch(&config) {
                    Ok(path) => {
                        println!("OK -> {}", path.display());
                        ok += 1;
                    }
                    Err(e) => {
                        println!("FAIL: {e}");
                        fail += 1;
                    }
                }
            }
            println!("---");
            println!("OK: {ok}, Failed: {fail}");
        }
        Cmd::Verify => {
            println!("=== Verifying All Cached Datasets ===");
            let mut verified = 0usize;
            let mut missing = 0usize;
            for p in &providers {
                if p.is_cached(&config) {
                    println!("  [OK] {}", p.name());
                    verified += 1;
                } else {
                    println!("  [MISSING] {}", p.name());
                    missing += 1;
                }
            }
            println!("---");
            println!("Verified: {verified}, Missing: {missing}");
        }
    }
}
