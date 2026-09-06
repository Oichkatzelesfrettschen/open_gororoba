//! Independent receipt integrity and paired-admission checks for the frozen LBM pilot.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

const CONDITIONS: [&str; 7] = [
    "C0-uniform-zero",
    "C1-uniform-fzd",
    "C2-noise-zero",
    "C3-noise-fzd",
    "C4-sersic-zero",
    "C5-sersic-fzd",
    "C6-sersic-nfw",
];
type Identity = (String, usize);
#[derive(Debug, Deserialize)]
struct Row {
    condition: String,
    trial: usize,
    seed: Option<u64>,
    df_initial: f64,
    df_final: f64,
    observed_step: usize,
    attempted_step: usize,
    mass_error: f64,
    max_mach: f64,
    minimum_population: f64,
    minimum_density: f64,
    finite_state: bool,
    positive_density: bool,
    nonnegative_population: bool,
    mass_within_budget: bool,
    mach_within_budget: bool,
    failure: String,
}
impl Row {
    fn admitted(&self) -> bool {
        self.failure.is_empty()
            && self.observed_step == 24
            && self.attempted_step == 24
            && self.finite_state
            && self.positive_density
            && self.nonnegative_population
            && self.mass_within_budget
            && self.mach_within_budget
    }
}
struct Trial {
    row: Row,
    metadata: Value,
    metadata_sha256: String,
    density: Vec<u8>,
    force: Vec<u8>,
}
fn identities() -> Vec<Identity> {
    CONDITIONS
        .iter()
        .enumerate()
        .flat_map(|(index, condition)| {
            (0..if index == 2 || index == 3 { 10 } else { 1 })
                .map(move |trial| ((*condition).to_string(), trial))
        })
        .collect()
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn verify_array(metadata: &Value, bytes: &[u8], expected_bytes: usize) -> Result<()> {
    ensure!(
        metadata["encoding"] == "IEEE754_f64_little_endian",
        "array encoding"
    );
    ensure!(
        bytes.len() == expected_bytes && metadata["bytes"].as_u64() == Some(bytes.len() as u64),
        "array byte count"
    );
    ensure!(
        metadata["sha256"].as_str() == Some(digest(bytes).as_str()),
        "array SHA256"
    );
    ensure!(
        bytes
            .chunks_exact(8)
            .all(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()).is_finite()),
        "nonfinite input"
    );
    Ok(())
}
fn load_backend(
    read: impl Fn(&str) -> Result<Vec<u8>>,
    backend: &str,
) -> Result<BTreeMap<Identity, Trial>> {
    let csv = read("trials.csv")?;
    let mut rows = BTreeMap::new();
    for decoded in csv::Reader::from_reader(csv.as_slice()).deserialize::<Row>() {
        let row = decoded?;
        let identity = (row.condition.clone(), row.trial);
        ensure!(identities().contains(&identity), "unexpected identity");
        ensure!(!rows.contains_key(&identity), "duplicate identity");
        let directory = format!("{}-{}", row.condition, row.trial);
        let metadata_bytes = read(&format!("{directory}/input.json"))?;
        let metadata: Value = serde_json::from_slice(&metadata_bytes)?;
        ensure!(
            metadata["condition"] == row.condition
                && metadata["trial"].as_u64() == Some(row.trial as u64),
            "metadata identity"
        );
        ensure!(metadata["backend"] == backend, "backend identity");
        let expected_seed = if row.condition.starts_with("C2-") || row.condition.starts_with("C3-")
        {
            Some(42 + row.trial as u64)
        } else {
            None
        };
        ensure!(
            row.seed == expected_seed && metadata["seed"] == json!(expected_seed),
            "paired seed"
        );
        for (key, expected) in [
            ("grid", json!(16)),
            ("steps", json!(24)),
            ("tau", json!(0.8)),
            ("alpha_zd", json!(0.1)),
            ("dx_kpc", json!(1.0)),
            ("density_floor", json!(0.045)),
            ("softening_eps", json!(0.5)),
            ("smagorinsky_cs", json!(0.0)),
            ("collision", json!("MRT")),
            ("max_relative_mass_error", json!(1e-5)),
            ("max_mach", json!(0.3)),
            ("zero_initial_velocity", json!(true)),
            (
                "observation",
                json!("direct_every_step_post_step_population_moments"),
            ),
        ] {
            ensure!(metadata[key] == expected, "frozen metadata {key}");
        }
        ensure!(
            metadata["selected_object_id"]
                .as_str()
                .is_some_and(|value| !value.is_empty()),
            "selected object"
        );
        let density = read(&format!("{directory}/rho.f64le"))?;
        let force = read(&format!("{directory}/force.xyz.f64le"))?;
        verify_array(&metadata["rho.f64le"], &density, 16usize.pow(3) * 8)?;
        verify_array(&metadata["force.xyz.f64le"], &force, 16usize.pow(3) * 24)?;
        ensure!(
            row.observed_step <= row.attempted_step && row.attempted_step <= 24,
            "step order"
        );
        ensure!(
            row.df_initial.is_finite() && (0.0..=3.0).contains(&row.df_initial),
            "initial dimension"
        );
        ensure!(
            row.failure.is_empty() == row.admitted(),
            "success requires all gates and complete observations"
        );
        if row.admitted() {
            ensure!(
                row.df_final.is_finite() && (0.0..=3.0).contains(&row.df_final),
                "final dimension"
            );
            ensure!(
                row.mass_error.is_finite()
                    && (0.0..=1e-5).contains(&row.mass_error)
                    && row.max_mach.is_finite()
                    && (0.0..=0.3).contains(&row.max_mach),
                "admitted moment diagnostics"
            );
            ensure!(
                row.minimum_population.is_finite()
                    && row.minimum_population >= 0.0
                    && row.minimum_density.is_finite()
                    && row.minimum_density > 0.0,
                "admitted population diagnostics"
            );
        }
        rows.insert(
            identity,
            Trial {
                row,
                metadata,
                metadata_sha256: digest(&metadata_bytes),
                density,
                force,
            },
        );
    }
    ensure!(
        rows.keys().cloned().collect::<Vec<_>>() == identities(),
        "exact 25 identities required"
    );
    for trial in 0..10 {
        ensure!(
            rows[&(CONDITIONS[2].into(), trial)].density
                == rows[&(CONDITIONS[3].into(), trial)].density,
            "noise pairing"
        );
    }
    let full = &rows[&(CONDITIONS[5].into(), 0)];
    let nfw = &rows[&(CONDITIONS[6].into(), 0)];
    ensure!(
        full.density == nfw.density && full.force != nfw.force,
        "galaxy control separation"
    );
    Ok(rows)
}
fn compare(cpu: &BTreeMap<Identity, Trial>, cuda: &BTreeMap<Identity, Trial>) -> Result<Value> {
    let mut differences = Vec::new();
    let mut failures = Vec::new();
    let mut inputs = Vec::new();
    for identity in identities() {
        let left = &cpu[&identity];
        let right = &cuda[&identity];
        for (backend, trial) in [("CPU_FP64", left), ("CUDA_FP32", right)] {
            inputs.push(json!({"backend":backend,"condition":identity.0,"trial":identity.1,"metadata_sha256":trial.metadata_sha256,"rho":trial.metadata["rho.f64le"],"force":trial.metadata["force.xyz.f64le"]}));
        }
        ensure!(
            left.density == right.density && left.force == right.force,
            "cross-backend input bytes {identity:?}"
        );
        let mut left_metadata = left.metadata.clone();
        let mut right_metadata = right.metadata.clone();
        left_metadata.as_object_mut().unwrap().remove("backend");
        right_metadata.as_object_mut().unwrap().remove("backend");
        ensure!(
            left_metadata == right_metadata,
            "cross-backend metadata {identity:?}"
        );
        for (backend, row) in [("CPU_FP64", &left.row), ("CUDA_FP32", &right.row)] {
            if !row.admitted() {
                failures.push(json!({"backend":backend,"condition":row.condition,"trial":row.trial,"attempted_step":row.attempted_step,"observed_step":row.observed_step,"failure":row.failure,"finite_state":row.finite_state,"positive_density":row.positive_density,"nonnegative_population":row.nonnegative_population,"mass_within_budget":row.mass_within_budget,"mach_within_budget":row.mach_within_budget,"max_mach":row.max_mach}));
            }
        }
        if left.row.admitted()
            && right.row.admitted()
            && left.row.observed_step == right.row.observed_step
        {
            differences.push(json!({"condition":identity.0,"trial":identity.1,"observed_step":left.row.observed_step,"cpu_minus_cuda_df":left.row.df_final-right.row.df_final}));
        }
    }
    Ok(
        json!({"trial_count":50,"array_count":100,"cross_backend_pairs":25,"inputs":inputs,"admitted_same_stage_pairs":differences,"failures":failures,"numerical_accuracy_gate":"unassessed","historical_pilot_exit_status":1,"scope":"Receipt integrity and descriptive same-stage comparison; pilot failure remains a failed numerical control result."}),
    )
}

#[test]
fn rejects_tampered_array_hash_and_count() {
    let bytes = 1.0_f64.to_le_bytes();
    let metadata =
        json!({"encoding":"IEEE754_f64_little_endian","bytes":8,"sha256":digest(&bytes)});
    verify_array(&metadata, &bytes, 8).unwrap();
    let mut changed = bytes;
    changed[0] ^= 1;
    assert!(verify_array(&metadata, &changed, 8).is_err());
    assert!(verify_array(&metadata, &bytes[..7], 8).is_err());
    let mut changed_metadata = metadata;
    changed_metadata["sha256"] = json!("0".repeat(64));
    assert!(verify_array(&changed_metadata, &bytes, 8).is_err());
}

#[test]
fn rejects_missing_rows() {
    let result = load_backend(
        |name| {
            ensure!(name == "trials.csv", "unexpected file");
            Ok(b"condition,trial,seed\n".to_vec())
        },
        "CPU_FP64",
    );
    assert!(result.is_err());
}

#[test]
#[ignore = "requires retained CPU/CUDA pilot arrays; set NULL_PILOT_AUDIT_ROOT"]
fn audit_retained_pilot_receipts() -> Result<()> {
    let root = std::env::var("NULL_PILOT_AUDIT_ROOT")
        .context("set NULL_PILOT_AUDIT_ROOT to the retained audit directory")?;
    let root = Path::new(&root);
    let load = |directory: &str, backend: &str| {
        load_backend(
            |name| Ok(std::fs::read(root.join(directory).join(name))?),
            backend,
        )
    };
    let cpu = load("null-pilot-cpu", "CPU_FP64")?;
    let cuda = load("null-pilot-cuda", "CUDA_FP32")?;
    for mutation in ["array", "metadata", "missing_row", "false_success"] {
        let altered = load_backend(
            |name| {
                let mut bytes = std::fs::read(root.join("null-pilot-cpu").join(name))?;
                if mutation == "array" && name == "C0-uniform-zero-0/rho.f64le" {
                    bytes[0] ^= 1;
                }
                if mutation == "metadata" && name == "C0-uniform-zero-0/input.json" {
                    let mut metadata: Value = serde_json::from_slice(&bytes)?;
                    metadata["tau"] = json!(0.9);
                    bytes = serde_json::to_vec(&metadata)?;
                }
                if name == "trials.csv" && mutation == "missing_row" {
                    let text = String::from_utf8(bytes)?;
                    let mut lines: Vec<_> = text.lines().collect();
                    lines.pop();
                    bytes = lines.join("\n").into_bytes();
                }
                if name == "trials.csv" && mutation == "false_success" {
                    bytes = String::from_utf8(bytes)?
                        .replace("raw population-moment Mach budget exceeded", "")
                        .into_bytes();
                }
                Ok(bytes)
            },
            "CPU_FP64",
        );
        ensure!(altered.is_err(), "mutation survived: {mutation}");
    }
    let summary = compare(&cpu, &cuda)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if let Ok(destination) = std::env::var("NULL_PILOT_ANALYSIS_OUTPUT") {
        let file = std::fs::File::create_new(destination)?;
        serde_json::to_writer_pretty(&file, &summary)?;
        file.sync_all()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires retained pilot arrays; set NULL_PILOT_AUDIT_ROOT"]
fn unforced_rest_streaming_predicts_initial_mach_failure() -> Result<()> {
    let root = std::env::var("NULL_PILOT_AUDIT_ROOT")
        .context("set NULL_PILOT_AUDIT_ROOT to the retained audit directory")?;
    let root = Path::new(&root);
    let trials = load_backend(
        |name| Ok(std::fs::read(root.join("null-pilot-cpu").join(name))?),
        "CPU_FP64",
    )?;
    let trial = &trials[&("C4-sersic-zero".into(), 0)];
    ensure!(
        trial
            .force
            .chunks_exact(8)
            .all(|bytes| { f64::from_le_bytes(bytes.try_into().unwrap()) == 0.0 }),
        "unforced oracle requires zero force"
    );
    let density: Vec<_> = trial
        .density
        .chunks_exact(8)
        .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    // Rest equilibrium is invariant under an unforced collision. Generate
    // the D3Q19 stencil independently and stream the retained density once.
    let mut maximum_mach = 0.0_f64;
    for cell in 0..16usize.pow(3) {
        let position = [cell % 16, cell / 16 % 16, cell / 256];
        let mut streamed_density = 0.0;
        let mut momentum = [0.0; 3];
        for velocity_x in -1_i32..=1 {
            for velocity_y in -1_i32..=1 {
                for velocity_z in -1_i32..=1 {
                    let velocity = [velocity_x, velocity_y, velocity_z];
                    let squared_speed = velocity.iter().map(|value| value * value).sum::<i32>();
                    let weight = match squared_speed {
                        0 => 1.0 / 3.0,
                        1 => 1.0 / 18.0,
                        2 => 1.0 / 36.0,
                        _ => continue,
                    };
                    let source: [usize; 3] = std::array::from_fn(|axis| {
                        (position[axis] as i32 - velocity[axis]).rem_euclid(16) as usize
                    });
                    let population = weight * density[source[0] + 16 * source[1] + 256 * source[2]];
                    streamed_density += population;
                    for axis in 0..3 {
                        momentum[axis] += f64::from(velocity[axis]) * population;
                    }
                }
            }
        }
        let mach = (3.0 * momentum.iter().map(|value| value * value).sum::<f64>()).sqrt()
            / streamed_density;
        maximum_mach = maximum_mach.max(mach);
    }
    ensure!(
        trial.row.observed_step == 1,
        "first-step observation required"
    );
    ensure!(
        maximum_mach > 0.3,
        "frozen Mach gate must discriminate failure"
    );
    ensure!(
        (maximum_mach - trial.row.max_mach).abs() < 1e-12,
        "independent equilibrium-streaming oracle disagrees"
    );
    println!("independent_rest_streaming_max_mach={maximum_mach:.17}");
    Ok(())
}
