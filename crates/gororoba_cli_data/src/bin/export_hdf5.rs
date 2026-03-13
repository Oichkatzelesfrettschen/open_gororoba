//! Export computed data to HDF5 format.
//!
//! Reads canonical control-plane compatibility data and writes a consolidated
//! HDF5 file with groups:
//! - `/registry/claims` -- claims table (ID + status)
//! - `/registry/insights` -- insights summary
//!
//! Lattice data export requires running motif-census first to generate CSVs.
//! This binary is a thin CLI wrapper around data_core::hdf5_export.
//!
//! Requires the `hdf5-export` feature on data_core (and libhdf5 at link time).

fn main() {
    // Feature-gate check: if data_core was not compiled with hdf5-export,
    // this binary still compiles but prints an error.
    #[cfg(not(feature = "hdf5-export"))]
    {
        eprintln!("ERROR: export-hdf5 requires the hdf5-export feature.");
        eprintln!(
            "Rebuild with: cargo run --release --features data_core/hdf5-export --bin export-hdf5"
        );
        std::process::exit(1);
    }

    #[cfg(feature = "hdf5-export")]
    {
        run_export();
    }
}

#[cfg(feature = "hdf5-export")]
fn run_export() {
    use clap::Parser;
    use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
    use std::path::PathBuf;

    #[derive(Parser)]
    #[command(name = "export-hdf5")]
    struct Args {
        /// Output HDF5 file
        #[arg(long, default_value = "data/h5/gororoba.h5")]
        output: PathBuf,

        /// Registry directory
        #[arg(long, default_value = "registry")]
        registry_dir: PathBuf,

        /// Canonical control-plane database. When present, registry summaries
        /// are loaded from SQLite-backed compatibility text first.
        #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
        canonical_db: PathBuf,
    }

    let args = Args::parse();

    // Ensure output directory exists
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Remove existing file to start fresh
    if args.output.exists() {
        std::fs::remove_file(&args.output).unwrap();
    }

    // --- Export claims ---
    let claims_path = args.registry_dir.join("claims.toml");
    let claims_text = if args.canonical_db.exists() {
        let mut store = ProvenanceStore::open(&args.canonical_db).unwrap_or_else(|err| {
            panic!("open canonical db {}: {err}", args.canonical_db.display())
        });
        store
            .control_plane_compat_text(ControlPlaneCompatKind::Claims)
            .unwrap_or_else(|err| {
                panic!(
                    "render claims compatibility text from canonical db {}: {err}",
                    args.canonical_db.display()
                )
            })
    } else if claims_path.exists() {
        std::fs::read_to_string(&claims_path).unwrap()
    } else {
        String::new()
    };
    if !claims_text.is_empty() {
        #[derive(serde::Deserialize)]
        struct ClaimsRegistry {
            claim: Vec<ClaimEntry>,
        }
        #[derive(serde::Deserialize)]
        struct ClaimEntry {
            id: String,
            status: String,
        }

        let registry: ClaimsRegistry = toml::from_str(&claims_text).unwrap();

        let ids: Vec<String> = registry.claim.iter().map(|c| c.id.clone()).collect();
        let statuses: Vec<String> = registry.claim.iter().map(|c| c.status.clone()).collect();

        data_core::hdf5_export::export_claims_summary(&args.output, &ids, &statuses).unwrap();
        println!("Exported {} claims to {}", ids.len(), args.output.display());
    } else {
        eprintln!(
            "WARNING: missing both canonical db {} and fallback {}",
            args.canonical_db.display(),
            claims_path.display()
        );
    }

    println!("HDF5 export complete: {}", args.output.display());
}
