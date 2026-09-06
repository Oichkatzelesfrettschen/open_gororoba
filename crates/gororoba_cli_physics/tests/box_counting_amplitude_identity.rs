use cosmology_core::sersic::{
    box_counting_fractal_dim, box_counting_fractal_dim_threshold, otsu_threshold,
};
use gororoba_cli_physics::{lbm_dispatch::LbmBackend, lbm_population_diagnostics::inspect_fields};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, error::Error, io::Write};

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug)]
struct Observation {
    counts: [usize; 4],
    mask: Vec<u8>,
    measured_slope: Option<f64>,
}

fn observe(density: &[f64], threshold: f64) -> Observation {
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
    Observation {
        counts,
        mask,
        measured_slope,
    }
}

fn record(label: &str, density: &[f64], threshold: f64, observation: &Observation) {
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

#[test]
fn exact_plane_affine_controls_preserve_geometry_at_tiny_contrast() {
    for offset in [0.0, 1.0] {
        for amplitude in [1.0, 2.0_f64.powi(-20), 2.0_f64.powi(-40)] {
            let density: Vec<_> = (0..4096)
                .map(|index| offset + if index % 16 == 8 { amplitude } else { 0.0 })
                .collect();
            let threshold = otsu_threshold(&density);
            let observation = observe(&density, threshold);
            assert_eq!(observation.counts, [256, 64, 16, 4]);
            assert!((observation.measured_slope.unwrap() - 2.0).abs() < 1e-12);
            assert!((box_counting_fractal_dim(&density, 16, 16, 16) - 2.0).abs() < 1e-12);
            record(
                &format!("plane_offset{offset}_amplitude{amplitude}"),
                &density,
                threshold,
                &observation,
            );
        }
    }
    let uniform = vec![1.0; 4096];
    let observation = observe(&uniform, otsu_threshold(&uniform));
    assert_eq!(observation.counts, [0; 4]);
    assert!(observation.measured_slope.is_none());
    assert_eq!(box_counting_fractal_dim(&uniform, 16, 16, 16), 3.0);
    record(
        "exact_uniform_no_occupied_convention",
        &uniform,
        1.0,
        &observation,
    );
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

#[test]
fn retained_uniform_force_replay_separates_amplitude_from_adaptive_geometry()
-> Result<(), Box<dyn Error>> {
    let density = read_f64(
        "data/output/audit/claim-family-evidence-adjudication/null-pilot-cpu/C1-uniform-fzd-0/rho.f64le",
        "9ad28ff0dbb91703f930756f2c9d0f2a9f7655d6e0efeb120b4c23f9321bdaa7",
    )?;
    let force = read_f64(
        "data/output/audit/claim-family-evidence-adjudication/null-pilot-cpu/C1-uniform-fzd-0/force.xyz.f64le",
        "2050ed246e70e02a6aadafedc5f70147ba7ee980d937435c9dc16931f6557166",
    )?;
    let force: Vec<[f64; 3]> = force
        .chunks_exact(3)
        .map(|chunk| chunk.try_into().unwrap())
        .collect();
    let mut backend = LbmBackend::cpu(16, 16, 16, 0.8, lbm_3d::solver::CollisionMode::Mrt);
    backend.initialize_custom(&density, &vec![[0.0; 3]; 4096])?;
    backend.set_force_field(force)?;
    let initial_mass = inspect_fields(&mut backend)?.mass;
    for _ in 0..24 {
        backend.step()?;
        inspect_fields(&mut backend)?.require_stable(initial_mass, 1e-5, 0.3)?;
    }
    let final_density = inspect_fields(&mut backend)?.density;
    if let Ok(path) = std::env::var("BOX_COUNTING_REPLAY_FIELD") {
        let mut output = std::fs::File::create_new(path)?;
        for value in &final_density {
            output.write_all(&value.to_le_bytes())?;
        }
        output.sync_all()?;
    }
    let original = observe(&final_density, otsu_threshold(&final_density));
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
        let adaptive_observation = observe(&transformed, adaptive);
        record(
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
        let fixed_observation = observe(&transformed, fixed);
        record(
            &format!("replay_amplitude{amplitude}_fixed"),
            &transformed,
            fixed,
            &fixed_observation,
        );
    }
    Ok(())
}
