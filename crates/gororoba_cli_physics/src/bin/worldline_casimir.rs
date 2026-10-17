//! Real-scalar Dirichlet two-body interaction in three spatial dimensions.
//! Local density and integrated energy per area have separate units and
//! comparisons. Single-body self-energy requires a different renormalization.

use casimir_core::{
    AsciiGrid, CasimirGeometry, ParallelPlates, SphereInCylinder, WorldlineCasimirConfig,
    scalar_parallel_plates_exact, scalar_parallel_plates_worldline, try_casimir_energy_at_point,
};
use clap::Parser;
use rayon::prelude::*;
use std::{error::Error, fmt::Write as _, fs, path::PathBuf};

type CliResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Parser)]
#[command(name = "worldline-casimir")]
#[command(about = "Computed real-scalar two-body Dirichlet interaction, hbar=c=1")]
struct Args {
    /// Disconnected two-body geometry: plates or sphere-cylinder.
    #[arg(long, default_value = "plates")]
    geometry: String,
    /// Plate separation or outer-cylinder radius in micrometres.
    #[arg(long, default_value_t = 1.0)]
    separation: f64,
    #[arg(long, default_value_t = 10000)]
    n_loops: usize,
    #[arg(long, default_value_t = 500)]
    n_loop_points: usize,
    #[arg(long, default_value_t = 20)]
    n_profile: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Render a checked scalar interaction cross-section.
    #[arg(long)]
    ascii: bool,
}

fn validate_arguments(args: &Args) -> CliResult<()> {
    if !args.separation.is_finite()
        || args.separation <= 0.0
        || args.n_profile == 0
        || args.n_loops < 2
        || args.n_loop_points < 3
    {
        return Err("positive finite separation, positive profile size, at least two loops and three vertices are required".into());
    }
    match args.geometry.as_str() {
        "plates" | "sphere-cylinder" => Ok(()),
        "sphere" | "cylinder" | "pillar" => Err(
            "the selected geometry requires a single-body or connected-body self-energy renormalization; the scalar interaction estimator supports two disconnected bodies".into()),
        _ => Err("unknown geometry; select plates or sphere-cylinder".into()),
    }
}

fn profile<G: CasimirGeometry>(
    geometry: &G,
    start: [f64; 3],
    end: [f64; 3],
    args: &Args,
    config: &WorldlineCasimirConfig,
) -> CliResult<String> {
    geometry.validate_interaction()?;
    let results: Result<Vec<_>, _> = (0..args.n_profile)
        .into_par_iter()
        .map(|index| {
            let fraction = index as f64 / args.n_profile.saturating_sub(1).max(1) as f64;
            let point = std::array::from_fn(|coordinate| {
                start[coordinate] + fraction * (end[coordinate] - start[coordinate])
            });
            try_casimir_energy_at_point(geometry, point, config).map(|result| (point, result))
        })
        .collect();
    let mut csv = String::from(
        "x_um,y_um,z_um,scalar_interaction_density_inv_um4,mc_standard_error_inv_um4\n",
    );
    for (point, result) in results? {
        writeln!(
            csv,
            "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            point[0], point[1], point[2], result.energy, result.error
        )?;
    }
    if args.ascii {
        let rows = 20;
        let columns = 40;
        let values: Result<Vec<_>, _> = (0..rows * columns)
            .into_par_iter()
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                let z = args.separation - (row as f64 + 0.5) * 2.0 * args.separation / rows as f64;
                let y = -args.separation
                    + (column as f64 + 0.5) * 2.0 * args.separation / columns as f64;
                try_casimir_energy_at_point(geometry, [0.0, y, z], config)
                    .map(|result| result.energy)
            })
            .collect();
        let mut grid = AsciiGrid::new(
            rows,
            columns,
            (
                -args.separation,
                args.separation,
                -args.separation,
                args.separation,
            ),
        );
        grid.data = values?;
        eprintln!("{}", grid.render());
    }
    Ok(csv)
}

fn main() -> CliResult<()> {
    let args = Args::parse();
    validate_arguments(&args)?;
    let separation = args.separation;
    let config = WorldlineCasimirConfig {
        n_loop_points: args.n_loop_points,
        n_loops: args.n_loops,
        t_min: 0.001 * separation * separation,
        t_max: 10.0 * separation * separation,
        n_t_points: 16,
        seed: args.seed,
    };
    let csv = match args.geometry.as_str() {
        "plates" => {
            let integrated = scalar_parallel_plates_worldline(
                separation,
                args.n_loop_points,
                args.n_loops,
                args.seed,
            )?;
            let exact = scalar_parallel_plates_exact(separation);
            eprintln!(
                "computed scalar plate E/A={:.9e}, MC standard error={:.9e}, exact={:.9e} inverse_um3; finite-polygon bias remains separate",
                integrated.energy, integrated.error, exact
            );
            profile(
                &ParallelPlates { separation },
                [0.0, 0.0, 0.1 * separation],
                [0.0, 0.0, 0.9 * separation],
                &args,
                &config,
            )?
        }
        "sphere-cylinder" => profile(
            &SphereInCylinder {
                sphere_radius: separation * 0.3,
                cylinder_radius: separation,
            },
            [0.0; 3],
            [separation * 0.9, 0.0, 0.0],
            &args,
            &config,
        )?,
        _ => unreachable!("argument validation selects a supported interaction"),
    };
    eprintln!(
        "local scalar density uses inverse_um4; multiply by hbar*c*1e24 for J/m^3. Statistical errors exclude quadrature, proper-time and polygon biases."
    );
    match args.output {
        Some(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, csv)?;
        }
        None => print!("{csv}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_energy_geometry_and_invalid_sampling_reject_before_execution() {
        for geometry in ["sphere", "cylinder", "pillar", "unknown"] {
            let args = Args::try_parse_from(["worldline-casimir", "--geometry", geometry]).unwrap();
            assert!(validate_arguments(&args).is_err());
        }
        for arguments in [
            ["--separation", "0"],
            ["--n-profile", "0"],
            ["--n-loops", "1"],
        ] {
            let args =
                Args::try_parse_from(["worldline-casimir", arguments[0], arguments[1]]).unwrap();
            assert!(validate_arguments(&args).is_err());
        }
    }

    #[test]
    fn local_density_csv_has_its_own_units_and_positive_error() {
        let args = Args::try_parse_from([
            "worldline-casimir",
            "--n-profile",
            "1",
            "--n-loops",
            "30",
            "--n-loop-points",
            "12",
        ])
        .unwrap();
        let config = WorldlineCasimirConfig {
            n_loops: 30,
            n_loop_points: 12,
            n_t_points: 4,
            ..WorldlineCasimirConfig::default()
        };
        let csv = profile(
            &ParallelPlates { separation: 1.0 },
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.5],
            &args,
            &config,
        )
        .unwrap();
        assert!(csv.starts_with(
            "x_um,y_um,z_um,scalar_interaction_density_inv_um4,mc_standard_error_inv_um4\n"
        ));
        let values: Vec<f64> = csv
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .map(|value| value.parse().unwrap())
            .collect();
        assert!(values[3] <= 0.0);
        assert!(values[4] >= 0.0);
    }
}
