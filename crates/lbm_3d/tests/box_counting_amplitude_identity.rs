use cosmology_core::sersic::{
    box_counting_fractal_dim, box_counting_fractal_dim_threshold, otsu_threshold,
};
use lbm_3d::{
    lattice::D3Q19Lattice,
    solver::{CollisionMode, LbmSolver3D, aosoa_idx},
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, io::Write};

const DENSITY_PATH: &str = "data/output/audit/claim-family-evidence-adjudication/null-pilot-cpu/C1-uniform-fzd-0/rho.f64le";
const DENSITY_SHA256: &str = "9ad28ff0dbb91703f930756f2c9d0f2a9f7655d6e0efeb120b4c23f9321bdaa7";
const FORCE_PATH: &str = "data/output/audit/claim-family-evidence-adjudication/null-pilot-cpu/C1-uniform-fzd-0/force.xyz.f64le";
const FORCE_SHA256: &str = "2050ed246e70e02a6aadafedc5f70147ba7ee980d937435c9dc16931f6557166";

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn read_f64(path: &str, expected_hash: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let bytes = std::fs::read(repo_root::path!(path))?;
    assert_eq!(digest(&bytes), expected_hash);
    assert_eq!(bytes.len() % 8, 0);
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[derive(Debug)]
struct GeometryObservation {
    counts: [usize; 4],
    mask: Vec<u8>,
    measured_slope: Option<f64>,
}

fn observe_geometry(density: &[f64], threshold: f64) -> GeometryObservation {
    assert_eq!(density.len(), 16 * 16 * 16);
    assert!(density.iter().all(|value| value.is_finite()));
    let mask: Vec<_> = density
        .iter()
        .map(|value| u8::from(*value > threshold))
        .collect();
    let counts = [1, 2, 4, 8].map(|scale| {
        let mut occupied = BTreeSet::new();
        for (index, selected) in mask.iter().enumerate() {
            if *selected != 0 {
                occupied.insert((
                    (index % 16) / scale,
                    ((index / 16) % 16) / scale,
                    (index / 256) / scale,
                ));
            }
        }
        occupied.len()
    });
    let measured_slope = if counts.iter().all(|count| *count > 0) {
        Some(
            -counts
                .iter()
                .enumerate()
                .map(|(index, count)| (index as f64 - 1.5) * (*count as f64).ln())
                .sum::<f64>()
                / (5.0 * 2.0_f64.ln()),
        )
    } else {
        None
    };
    GeometryObservation {
        counts,
        mask,
        measured_slope,
    }
}

fn record_geometry(
    label: &str,
    density: &[f64],
    threshold: f64,
    observation: &GeometryObservation,
) {
    let minimum = density.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = density.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "label={label} min={minimum:.17} max={maximum:.17} range={:.17} threshold={threshold:.17} scales=[1,2,4,8] counts={:?} mask_sha256={} measured_slope={:?} scalar={:.17}",
        maximum - minimum,
        observation.counts,
        digest(&observation.mask),
        observation.measured_slope,
        box_counting_fractal_dim_threshold(density, 16, 16, 16, threshold)
    );
}

struct PopulationObservation {
    density: Vec<f64>,
    mass: f64,
    minimum_density: f64,
    minimum_population: f64,
    maximum_mach: f64,
}

fn inspect_solver(solver: &LbmSolver3D) -> PopulationObservation {
    let lattice = D3Q19Lattice::new();
    let cell_count = solver.nx * solver.ny * solver.nz;
    let mut density = Vec::with_capacity(cell_count);
    let mut mass = 0.0;
    let mut minimum_density = f64::INFINITY;
    let mut minimum_population = f64::INFINITY;
    let mut maximum_mach = 0.0_f64;
    for cell_index in 0..cell_count {
        let mut cell_density = 0.0;
        let mut momentum = [0.0; 3];
        for direction in 0..19 {
            let population = solver.f[aosoa_idx(cell_index, direction)];
            assert!(population.is_finite());
            minimum_population = minimum_population.min(population);
            cell_density += population;
            let velocity = lattice.velocity(direction);
            for axis in 0..3 {
                momentum[axis] += population * f64::from(velocity[axis]);
            }
        }
        let mach = momentum
            .into_iter()
            .map(|component| (component / cell_density).powi(2))
            .sum::<f64>()
            .sqrt()
            * 3.0_f64.sqrt();
        assert!(cell_density.is_finite() && mach.is_finite());
        minimum_density = minimum_density.min(cell_density);
        maximum_mach = maximum_mach.max(mach);
        mass += cell_density;
        density.push(cell_density);
    }
    assert!(mass.is_finite());
    PopulationObservation {
        density,
        mass,
        minimum_density,
        minimum_population,
        maximum_mach,
    }
}

fn require_stable(observation: &PopulationObservation, initial_mass: f64) {
    assert!(initial_mass.is_finite() && initial_mass > 0.0);
    assert!(observation.mass.is_finite());
    assert!(observation.minimum_density > 0.0);
    assert!(observation.minimum_population >= 0.0);
    assert!((observation.mass / initial_mass - 1.0).abs() <= 1e-5);
    assert!(observation.maximum_mach <= 0.3);
}

#[test]
#[ignore = "requires the hydrated scientific payload archive"]
fn retained_uniform_force_replay_separates_amplitude_from_adaptive_geometry()
-> Result<(), Box<dyn Error>> {
    let density = read_f64(DENSITY_PATH, DENSITY_SHA256)?;
    let force_values = read_f64(FORCE_PATH, FORCE_SHA256)?;
    let force: Vec<[f64; 3]> = force_values
        .chunks_exact(3)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();
    let mut solver = LbmSolver3D::new_mrt(16, 16, 16, 0.8);
    assert_eq!(solver.collision_mode, CollisionMode::Mrt);
    solver.rho.copy_from_slice(&density);
    solver.u.fill([0.0; 3]);
    solver.reinitialize_from_macroscopic();
    solver.set_force_field(force)?;
    let initial_mass = inspect_solver(&solver).mass;
    for _ in 0..24 {
        solver.evolve_one_step();
        require_stable(&inspect_solver(&solver), initial_mass);
    }
    let final_density = inspect_solver(&solver).density;
    if let Ok(path) = std::env::var("BOX_COUNTING_REPLAY_FIELD") {
        let mut output = std::fs::File::create_new(path)?;
        for value in &final_density {
            output.write_all(&value.to_le_bytes())?;
        }
        output.sync_all()?;
    }
    let original = observe_geometry(&final_density, otsu_threshold(&final_density));
    let scalar = box_counting_fractal_dim(&final_density, 16, 16, 16);
    assert!(
        (scalar - 2.207_681_559_705_082_7).abs() < 1e-10,
        "retained scalar differs: {scalar}"
    );
    assert!((original.measured_slope.unwrap() - scalar).abs() < 1e-12);
    for amplitude in [2.0_f64.powi(-8), 1.0, 2.0_f64.powi(8)] {
        let transformed: Vec<_> = final_density
            .iter()
            .map(|value| 1.0 + amplitude * (value - 1.0))
            .collect();
        let adaptive = otsu_threshold(&transformed);
        let adaptive_observation = observe_geometry(&transformed, adaptive);
        record_geometry(
            &format!("replay_amplitude{amplitude}_adaptive"),
            &transformed,
            adaptive,
            &adaptive_observation,
        );
        assert_eq!(
            adaptive_observation.mask, original.mask,
            "adaptive superlevel mask changed at amplitude{amplitude}"
        );
        let fixed = 1.0 + 2.0_f64.powi(-20);
        let fixed_observation = observe_geometry(&transformed, fixed);
        record_geometry(
            &format!("replay_amplitude{amplitude}_fixed"),
            &transformed,
            fixed,
            &fixed_observation,
        );
    }
    Ok(())
}
