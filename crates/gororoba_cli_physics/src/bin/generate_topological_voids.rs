//! generate-topological-voids: Generate macroscopic topological voids
//! required to match the CDG-2 astrophysical observations (VF=0.0).
//!
//! Creates a 3D density field where baryonic matter is excluded from
//! regions of high Cayley-Dickson associator frustration.
//!
//! Supports dim=16 (sedenion), 32 (pathion), and 64 (chingon). Higher
//! dimensions produce denser zero-divisor networks whose connected
//! components seed cosmic web filament topology.

use clap::Parser;
use sign_imbalance::bridge::{CdField, SedenionField};
use std::{fs::File, io::Write};

#[derive(Parser)]
struct Args {
    /// Grid dimension.
    #[arg(long, default_value = "128")]
    grid: usize,
    /// Cayley-Dickson algebra dimension (16=sedenion, 32=pathion, 64=chingon).
    #[arg(long, default_value = "16")]
    dim: usize,
    /// Associator threshold for void exclusion.
    #[arg(long, default_value = "0.8")]
    threshold: f64,
    /// Path to a CIF file for anisotropic grid seeding.
    /// Uses unit cell aspect ratios to set anisotropic grid dimensions.
    #[arg(long)]
    cif: Option<String>,
    /// Path to save the resulting density field (CSV).
    #[arg(long, default_value = "data/topological_voids.csv")]
    output: String,
}

/// Resolve grid dimensions from CIF file or use isotropic default.
fn resolve_grid_dims(args: &Args) -> (usize, usize, usize) {
    if let Some(cif_path) = &args.cif {
        match std::fs::read_to_string(cif_path) {
            Ok(content) => {
                if let Some(params) = materials_core::cif_parser::parse_cif_cell_params(&content) {
                    // Scale grid dimensions by unit cell aspect ratios
                    let max_len = params.a.max(params.b).max(params.c);
                    let nx = ((params.a / max_len) * args.grid as f64).round().max(4.0) as usize;
                    let ny = ((params.b / max_len) * args.grid as f64).round().max(4.0) as usize;
                    let nz = ((params.c / max_len) * args.grid as f64).round().max(4.0) as usize;
                    eprintln!(
                        "CIF seeding: a={:.3}, b={:.3}, c={:.3} -> grid {}x{}x{}",
                        params.a, params.b, params.c, nx, ny, nz
                    );
                    return (nx, ny, nz);
                }
                eprintln!("Warning: Could not parse CIF cell params from {}", cif_path);
            }
            Err(e) => {
                eprintln!("Warning: Could not read CIF file {}: {}", cif_path, e);
            }
        }
    }
    (args.grid, args.grid, args.grid)
}

/// Pin rayon thread pool to physical cores for V-Cache locality.
fn pin_physical_cores() {
    #[cfg(target_os = "linux")]
    {
        use std::{collections::BTreeMap, fs};

        if let Ok(online) = fs::read_to_string("/sys/devices/system/cpu/online") {
            let mut core_groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
            for part in online.trim().split(',') {
                let range: Vec<&str> = part.split('-').collect();
                let (lo, hi) = if range.len() == 2 {
                    (
                        range[0].parse::<usize>().unwrap_or(0),
                        range[1].parse::<usize>().unwrap_or(0),
                    )
                } else {
                    let v = range[0].parse::<usize>().unwrap_or(0);
                    (v, v)
                };
                for cpu in lo..=hi {
                    let pkg = fs::read_to_string(format!(
                        "/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                    ))
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                    let core = fs::read_to_string(format!(
                        "/sys/devices/system/cpu/cpu{cpu}/topology/core_id"
                    ))
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(cpu);
                    core_groups.entry((pkg, core)).or_default().push(cpu);
                }
            }
            let physical_ids: Vec<usize> = core_groups
                .values()
                .filter_map(|cpus| cpus.iter().min().copied())
                .collect();

            let n = physical_ids.len().max(1);
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .start_handler(move |idx| {
                    if idx < physical_ids.len() {
                        core_affinity::set_for_current(core_affinity::CoreId {
                            id: physical_ids[idx],
                        });
                    }
                })
                .build_global();
            eprintln!("Pinned {} rayon threads to physical cores", n);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    pin_physical_cores();
    let (nx, ny, nz) = resolve_grid_dims(&args);
    let dim = args.dim;

    eprintln!(
        "Generating Macroscopic Topological Voids (Grid: {}x{}x{}, CD dim: {})",
        nx, ny, nz, dim
    );

    // Validate dimension
    if !dim.is_power_of_two() || dim < 16 {
        anyhow::bail!("--dim must be a power of 2 >= 16 (got {})", dim);
    }

    // For dim=16, use the optimized SedenionField path
    let imbalance = if dim == 16 {
        let mut field = SedenionField::uniform(nx, ny, nz);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let s = field.get_mut(x, y, z);
                    let fx = x as f64 / nx as f64;
                    let fy = y as f64 / ny as f64;
                    let fz = z as f64 / nz as f64;
                    for (i, si) in s.iter_mut().enumerate().skip(1).take(15) {
                        let freq = (i as f64 * 7.0).sqrt() * 10.0;
                        *si = (freq * (fx + fy + fz)).sin();
                    }
                }
            }
        }
        eprintln!("Computing imbalance field (dim={})...", dim);
        field.local_imbalance_density(dim)
    } else {
        // Use CdField for dim > 16
        let mut field = CdField::uniform(nx, ny, nz, dim);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let s = field.get_mut(x, y, z);
                    let fx = x as f64 / nx as f64;
                    let fy = y as f64 / ny as f64;
                    let fz = z as f64 / nz as f64;
                    for (i, si) in s.iter_mut().enumerate().skip(1).take(dim - 1) {
                        let freq = (i as f64 * 7.0).sqrt() * 10.0;
                        *si = (freq * (fx + fy + fz)).sin();
                    }
                }
            }
        }
        eprintln!("Computing imbalance field (dim={}, parallel)...", dim);
        field.local_imbalance_density_par()
    };

    // For dim >= 32, also extract ZD graph topology
    if dim >= 32 {
        eprintln!("Extracting ZD graph topology (dim={})...", dim);
        let (num_comp, sizes, edges) = materials_core::e8_crystal_bridge::zd_graph_topology(dim);
        eprintln!(
            "  ZD graph: {} components, {} edges, largest component: {}",
            num_comp,
            edges,
            sizes.first().copied().unwrap_or(0)
        );
        eprintln!("  Component sizes: {:?}", &sizes[..sizes.len().min(10)]);
    }

    // Apply exclusion principle: rho = 0 if imbalance > threshold
    let total_cells = nx * ny * nz;
    let mut rho_field = vec![1.0f64; total_cells];
    let mut void_count = 0;

    for (s, &imb) in rho_field.iter_mut().zip(imbalance.iter()) {
        if imb > args.threshold {
            *s = 0.0;
            void_count += 1;
        }
    }

    let vf = void_count as f64 / total_cells as f64;
    eprintln!("Void Generation Complete.");
    eprintln!("  Threshold: {}", args.threshold);
    eprintln!("  Void Volume Fraction (Exclusion): {:.6}", vf);
    eprintln!("  Baryon VF (Remaining): {:.6}", 1.0 - vf);

    if (1.0 - vf) < 0.0006 {
        eprintln!("  [SUCCESS] Matches CDG-2 astrophysical observation (VF < 0.0006)");
    } else {
        eprintln!(
            "  [NOTICE] Baryon fraction exceeds CDG-2 bound. Lower threshold or increase coupling."
        );
    }

    // Save to CSV
    let mut file = File::create(&args.output)?;
    writeln!(file, "x,y,z,rho,imbalance")?;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + nz * z);
                if rho_field[idx] < 0.1 || idx % 100 == 0 {
                    writeln!(
                        file,
                        "{},{},{},{:.4},{:.4}",
                        x, y, z, rho_field[idx], imbalance[idx]
                    )?;
                }
            }
        }
    }

    eprintln!("Data saved to {}", args.output);

    // TODO(Phase 7.5 -- moyo post-analysis):
    // moyo is strictly analytical: classifies symmetry AFTER density field generation.
    // 1. classify_void_symmetry(): extract peak/void 3D positions from rho_field
    // 2. moyo::determine_spacegroup() on extracted fractional coordinates
    // This is post-processing only -- not a runtime dependency of the generator.

    // TODO(Phase 7.6 -- Vulkan compute migration for dim>=256):
    // For 256^3 grids, migrate AVT contraction to Vulkan compute shader.
    // Reference: lbm_vulkan/src/compute.rs for ash 0.38.0 pipeline pattern.
    // Buffer architecture:
    //   SSBO: AVT violations tensor (stored once, read-only)
    //   UBO: time-varying parameters (Moon/Sun/Earth positions from ephemeris)
    // Target shader: rk4_chingon.comp (GLSL compute)
    // For 256D: ~16.7M ops/timestep, 4x per RK4 step = ~67M/step
    //   ~150k steps/flyby = ~10T FLOPs per flyby -- GPU territory

    Ok(())
}
