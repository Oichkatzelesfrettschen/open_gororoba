//! LoTSS DR3 scoped tile enumerator (plan P6A.S5.T3 binary behind
//! docs/adr/lotss-footprint-policy.md).
//!
//! Per the footprint-policy ADR, we do NOT walk the full 5400 sq deg
//! HBA-Dutch survey: we enumerate tiles intersecting the scoped
//! discovery surface (HETDEX field + active-dataset references).
//!
//! This binary generates the TILE QUERY MANIFEST only; it does NOT
//! hit the ASTRON VO endpoint. The manifest is a TOML file listing
//! (tile_id, center_ra, center_dec, radius_deg) triples that a
//! future fetch binary consumes (one query per tile, 250 ms gap).
//!
//! RCA-style design choice: generating the manifest OFFLINE means
//! - no network dependency during `make governance-gate`
//! - deterministic output across machines
//! - a human can review the tile set before any bulk VO query lands
//!
//! The scoped-discovery tiling uses a simple HEALPix-free grid: we
//! tesselate the HETDEX rectangle (RA 150-250, DEC 45-60) with
//! 2-degree cone-search queries spaced on a 1-degree grid, plus any
//! point targets in `registry/datasets.toml` with
//! `server_ref = "lotss_vo"`. Overlap is acceptable -- the VO
//! endpoint deduplicates internally.

use std::{
    collections::BTreeSet,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Parser;

const DEFAULT_REGISTRY: &str = "registry/datasets.toml";
const DEFAULT_OUTPUT: &str = "registry/lotss_dr3_tiles.toml";

#[derive(Parser)]
#[command(
    name = "lotss-enumerate",
    about = "Generate a scoped LoTSS DR3 tile-query manifest (plan P6A.S5.T3)."
)]
struct Args {
    /// Datasets registry (for lotss_vo cross-reference).
    #[arg(long, default_value = DEFAULT_REGISTRY)]
    datasets: PathBuf,

    /// Output manifest path.
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,

    /// Tile cone-search radius in degrees.
    #[arg(long, default_value_t = 2.0)]
    radius_deg: f64,

    /// Grid spacing in degrees.
    #[arg(long, default_value_t = 1.0)]
    step_deg: f64,

    /// HETDEX field bounds (RA_min, RA_max, DEC_min, DEC_max) in degrees.
    #[arg(long, default_values_t = vec![150.0, 250.0, 45.0, 60.0])]
    hetdex_bounds: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Tile {
    id: String,
    ra: f64,
    dec: f64,
    radius: f64,
    source: &'static str,
}

fn main() -> ExitCode {
    let args = Args::parse();

    if args.hetdex_bounds.len() != 4 {
        eprintln!("ERROR: --hetdex-bounds expects 4 floats (RA_min RA_max DEC_min DEC_max)");
        return ExitCode::FAILURE;
    }
    let (ra_min, ra_max, dec_min, dec_max) = (
        args.hetdex_bounds[0],
        args.hetdex_bounds[1],
        args.hetdex_bounds[2],
        args.hetdex_bounds[3],
    );

    let mut tiles: Vec<Tile> = Vec::new();

    // HETDEX grid tiles.
    let mut dec = dec_min;
    while dec <= dec_max + 1e-9 {
        let mut ra = ra_min;
        while ra <= ra_max + 1e-9 {
            tiles.push(Tile {
                id: format!("hetdex_ra{:05.1}_dec{:+05.1}", ra, dec),
                ra,
                dec,
                radius: args.radius_deg,
                source: "hetdex_grid",
            });
            ra += args.step_deg;
        }
        dec += args.step_deg;
    }

    // Point targets: any datasets.toml row with server_ref=lotss_vo and
    // a locatable position in notes. For now, we only cross-reference
    // existence of lotss_vo rows; the individual RA/DEC pins are left
    // for P6A.S5.T3 follow-up when the registry schema adds a
    // `center_coords` field.
    let lotss_vo_datasets = match load_lotss_vo_datasets(&args.datasets) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("WARN: could not load datasets registry: {e}");
            Vec::new()
        }
    };

    // Deterministic de-dup using rounded coordinates at 0.05 deg.
    let mut seen: BTreeSet<(i64, i64)> = BTreeSet::new();
    tiles.retain(|t| {
        let key = ((t.ra * 20.0).round() as i64, (t.dec * 20.0).round() as i64);
        seen.insert(key)
    });

    // Sort deterministically by id.
    tiles.sort_by(|a, b| a.id.cmp(&b.id));

    if let Err(e) = write_manifest(
        &args.output,
        &tiles,
        &lotss_vo_datasets,
        args.radius_deg,
        args.step_deg,
        (ra_min, ra_max, dec_min, dec_max),
    ) {
        eprintln!("ERROR: write {}: {}", args.output.display(), e);
        return ExitCode::FAILURE;
    }
    println!(
        "[lotss-enumerate] wrote {} ({} tiles, {} lotss_vo datasets cross-referenced)",
        args.output.display(),
        tiles.len(),
        lotss_vo_datasets.len()
    );
    ExitCode::SUCCESS
}

fn load_lotss_vo_datasets(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let doc: toml::Value =
        toml::from_str(&text).map_err(|e| format!("{}: {}", path.display(), e))?;
    let empty: Vec<toml::Value> = Vec::new();
    let rows = doc
        .get("dataset")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty);
    let mut out = Vec::new();
    for r in rows {
        let server = r.get("server_ref").and_then(|v| v.as_str()).unwrap_or("");
        if server == "lotss_vo"
            && let Some(id) = r.get("id").and_then(|v| v.as_str())
        {
            out.push(id.to_string());
        }
    }
    Ok(out)
}

fn write_manifest(
    path: &Path,
    tiles: &[Tile],
    cross_refs: &[String],
    radius_deg: f64,
    step_deg: f64,
    hetdex: (f64, f64, f64, f64),
) -> Result<(), String> {
    let mut out = String::new();
    out.push_str(
        "# LoTSS DR3 tile-query manifest (plan Phase 6A.S5.T3).\n\
# Generated by crates/gororoba_cli_data/src/bin/lotss_enumerate.rs.\n\
# Refresh: `make lotss-tiles-refresh` (future target).\n\
# Consumed by a future fetch binary that issues cone-search queries\n\
# against https://vo.astron.nl/hetdex/q with 250 ms inter-request\n\
# delay (per registry/data_servers.toml#lotss_vo).\n\
#\n\
# Schema per docs/adr/lotss-footprint-policy.md.\n\n",
    );
    out.push_str("[lotss_dr3_tiles]\n");
    out.push_str("updated = \"2026-04-17\"\n");
    out.push_str(&format!("radius_deg = {}\n", radius_deg));
    out.push_str(&format!("step_deg = {}\n", step_deg));
    out.push_str(&format!(
        "hetdex_bounds = [ {}, {}, {}, {} ]\n",
        hetdex.0, hetdex.1, hetdex.2, hetdex.3
    ));
    out.push_str(&format!("tile_count = {}\n", tiles.len()));
    out.push_str(&format!(
        "lotss_vo_dataset_refs = {}\n",
        format_array_of_strings(cross_refs)
    ));
    out.push('\n');

    for t in tiles {
        out.push_str("[[tile]]\n");
        out.push_str(&format!("id = \"{}\"\n", t.id));
        out.push_str(&format!("ra = {:.6}\n", t.ra));
        out.push_str(&format!("dec = {:.6}\n", t.dec));
        out.push_str(&format!("radius = {:.6}\n", t.radius));
        out.push_str(&format!("source = \"{}\"\n", t.source));
        out.push('\n');
    }

    let mut f = File::create(path).map_err(|e| e.to_string())?;
    f.write_all(out.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

fn format_array_of_strings(items: &[String]) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let mut s = String::from("[\n");
    for item in items {
        s.push_str(&format!("    \"{item}\",\n"));
    }
    s.push(']');
    s
}
