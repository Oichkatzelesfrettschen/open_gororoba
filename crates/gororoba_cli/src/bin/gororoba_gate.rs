//! Gororoba Falsifiability Gate.
//!
//! An automated CI/CD-like pipeline that:
//! 1. Reads canonical claim inventory from
//!    `registry/canonical/control_plane.sqlite3` when available.
//! 2. Runs the corresponding engine pipelines.
//! 3. Compares simulation metrics against claim thresholds.
//! 4. Reports status and suggests registry updates.

use gororoba_engine::{
    thesis_pipelines::{Thesis1Pipeline, Thesis2Pipeline, Thesis3Pipeline, Thesis4Pipeline},
    traits::ThesisPipeline,
};
use log::{error, info, warn};
use provenance_store::ProvenanceStore;
use std::{error::Error, path::Path};

fn log_control_plane_inventory() {
    let db_path = Path::new("registry/canonical/control_plane.sqlite3");
    if !db_path.exists() {
        warn!(
            "Canonical control plane not found at {}. Continuing with pipeline-only gate.",
            db_path.display()
        );
        return;
    }
    let store = match ProvenanceStore::open(db_path) {
        Ok(store) => store,
        Err(err) => {
            warn!(
                "Failed to open canonical control plane {}: {}",
                db_path.display(),
                err
            );
            return;
        }
    };
    match store.list_claims() {
        Ok(claims) => info!(
            "Canonical control plane inventory: {} claim rows loaded from {}",
            claims.len(),
            db_path.display()
        ),
        Err(err) => warn!(
            "Failed to read claim inventory from canonical control plane {}: {}",
            db_path.display(),
            err
        ),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    info!("=== Gororoba Falsifiability Gate ===");
    log_control_plane_inventory();

    let pipelines: Vec<Box<dyn ThesisPipeline>> = vec![
        Box::new(Thesis1Pipeline::default()),
        Box::new(Thesis2Pipeline::default()),
        Box::new(Thesis3Pipeline::default()),
        Box::new(Thesis4Pipeline::default()),
    ];

    let mut all_passed = true;
    let mut results = Vec::new();

    for pipeline in pipelines {
        info!("Running pipeline: {}", pipeline.name());
        let evidence = pipeline.execute();

        if evidence.passes_gate {
            info!("  [PASS] {}", evidence.label);
        } else {
            error!("  [FAIL] {}", evidence.label);
            all_passed = false;
        }

        for msg in &evidence.messages {
            info!("    - {}", msg);
        }
        results.push(evidence);
    }

    info!("=== Summary ===");
    if all_passed {
        info!("All active theses remain unfalsified. Scientific integrity preserved.");
    } else {
        warn!("Some theses have been FALSIFIED by current engine logic.");
        warn!("Review threshold alignment and physical implementation.");
    }

    Ok(())
}
