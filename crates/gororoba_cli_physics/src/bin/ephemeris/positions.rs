//! Heliocentric ecliptic positions of the planet barycentres at one epoch.
//!
//! Headless on purpose. A solar-system chart is only as good as the numbers
//! under it, and those are checkable against JPL Horizons before any pixel is
//! drawn: request the same body, the same TDB epoch, the ecliptic-of-J2000
//! frame and observer @sun, and compare `r_au`, `lat_deg` and `lon_deg`.
//!
//! Usage:
//!   ephemeris positions --jed 2451545.0
//!   ephemeris positions --jed 2460000.5 --body mars --format csv

use anyhow::Result;
use clap::{Args, ValueEnum};
use std::path::PathBuf;

use gororoba_cli_physics::ephemeris_loader::{HeliocentricEphemeris, SolarSystemBody};

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Table,
    Csv,
}

#[derive(Args)]
pub struct Cli {
    /// JPL DE-series kernel. DE440 covers 1550 through 2650.
    #[arg(long, default_value = "data/external/de440.bsp")]
    kernel: PathBuf,

    /// Julian Ephemeris Date in TDB. Defaults to J2000.0.
    #[arg(long, default_value_t = 2_451_545.0)]
    jed: f64,

    /// Restrict output to one body. Repeat to name several.
    #[arg(long)]
    body: Vec<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
    format: OutputFormat,
}

pub fn run(cli: Cli) -> Result<()> {
    let ephemeris = HeliocentricEphemeris::load(&cli.kernel)?;

    let selected: Vec<SolarSystemBody> = if cli.body.is_empty() {
        SolarSystemBody::ALL.to_vec()
    } else {
        cli.body
            .iter()
            .map(|name| {
                SolarSystemBody::ALL
                    .into_iter()
                    .find(|b| b.name() == name.to_lowercase())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown body '{name}'; expected one of: {}",
                            SolarSystemBody::ALL
                                .iter()
                                .map(|b| b.name())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })
            })
            .collect::<Result<_>>()?
    };

    if cli.format == OutputFormat::Csv {
        println!("body,naif_id,jed_tdb,r_au,lat_deg,lon_deg");
    } else {
        println!("Heliocentric ecliptic of J2000, JED(TDB) = {}", cli.jed);
        println!("  kernel: {}", cli.kernel.display());
        println!();
        println!(
            "  {:<22} {:>6}  {:>12}  {:>10}  {:>11}",
            "body", "naif", "r_au", "lat_deg", "lon_deg"
        );
    }

    for body in selected {
        let p = ephemeris.body_ecliptic(body, cli.jed)?;
        if cli.format == OutputFormat::Csv {
            println!(
                "{},{},{},{:.9},{:.6},{:.6}",
                body.name(),
                body.naif_id(),
                cli.jed,
                p.r_au,
                p.lat_deg,
                p.lon_deg
            );
        } else {
            println!(
                "  {:<22} {:>6}  {:>12.8}  {:>10.5}  {:>11.5}",
                body.name(),
                body.naif_id(),
                p.r_au,
                p.lat_deg,
                p.lon_deg
            );
        }
    }

    Ok(())
}
