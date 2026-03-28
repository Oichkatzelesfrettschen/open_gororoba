//! Inspect a BOUT++ tokamak grid file (NetCDF-3 classic format).
//! Uses the pure-Rust netcdf3 crate -- no C dependencies.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "bout-grid-inspect")]
struct Cli {
    /// Path to BOUT++ grid file (NetCDF-3 classic .nc).
    #[arg(long)]
    grid: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== BOUT++ Grid Inspector (netcdf3 pure Rust) ===");
    println!("  File: {}", cli.grid.display());

    let mut reader = netcdf3::FileReader::open(&cli.grid)
        .map_err(|e| anyhow::anyhow!("Failed to open NC3: {:?}", e))?;

    let ds = reader.data_set();

    // Dimensions
    println!("\n  Dimensions ({}):", ds.num_dims());
    for dim in ds.get_dims() {
        println!("    {}: size={}, unlimited={}", dim.name(), dim.size(), dim.is_unlimited());
    }

    // Global attributes
    println!("\n  Global attributes ({}):", ds.num_global_attrs());
    for name in ds.get_global_attr_names() {
        println!("    {}", name);
    }

    // Variables
    println!("\n  Variables ({}):", ds.num_vars());
    for var in ds.get_vars() {
        let dim_names = var.dim_names();
        println!(
            "    {}: type={:?}, dims=[{}], len={}",
            var.name(),
            var.data_type(),
            dim_names.join(", "),
            var.len()
        );
    }

    // Read key tokamak grid variables
    for key in ["Rxy", "Zxy", "Bpxy", "Btxy", "psixy", "nx", "ny"] {
        match reader.read_var(key) {
            Ok(data) => {
                let vals: Vec<f64> = match data {
                    netcdf3::DataVector::F64(v) => v,
                    netcdf3::DataVector::F32(v) => v.into_iter().map(|x| x as f64).collect(),
                    netcdf3::DataVector::I32(v) => v.into_iter().map(|x| x as f64).collect(),
                    netcdf3::DataVector::I16(v) => v.into_iter().map(|x| x as f64).collect(),
                    netcdf3::DataVector::U8(v) => v.into_iter().map(|x| x as f64).collect(),
                    netcdf3::DataVector::I8(v) => v.into_iter().map(|x| x as f64).collect(),
                };
                if !vals.is_empty() {
                    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!("\n  {}: len={}, min={:.6}, max={:.6}", key, vals.len(), min, max);
                }
            }
            Err(_) => {}
        }
    }

    println!("\n  Grid inspection complete.");
    Ok(())
}
