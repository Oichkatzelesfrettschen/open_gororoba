use std::fs::File;
use std::io::Write;
use std::path::Path;

use algebra_experimental::fractal_dimension::compute_zd_fractal_dimension;

// From Theorem 12.3: Calcagni Spectral Dimension runs from UV: 2.0 to IR: 4.0
// We map the scale parameter 's' to the Dimensional ZD topology.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Initializing Algebraic Spectral Dimension Sweep...");
    
    // We compute the exact algebraic fractal dimensions for CD manifolds
    let fbf_16d = 0.5;
    let nodes_16d = 84;
    let df_16d = compute_zd_fractal_dimension(fbf_16d, 4.0, nodes_16d);
    
    let fbf_32d = 4.0 / 7.0;
    let nodes_32d = 588;
    let df_32d = compute_zd_fractal_dimension(fbf_32d, 4.0, nodes_32d);

    let fbf_64d = 1854.0 / 3036.0;
    let nodes_64d = 3036;
    let df_64d = compute_zd_fractal_dimension(fbf_64d, 4.0, nodes_64d);

    println!("   - 16D Fractal Dimension: {:.4}", df_16d);
    println!("   - 32D Fractal Dimension: {:.4}", df_32d);
    println!("   - 64D Fractal Dimension: {:.4}", df_64d);
    
    // Now we map this algebraic compactness to the continuous Calcagni flow:
    // d_S(s) = 4 - 2/(1+s)
    // We solve for the effective scale factor 's' corresponding to each algebraic dimension
    // s = 2 / (4 - d_S) - 1

    let compute_s = |d_s: f64| -> f64 {
        2.0 / (4.0 - d_s) - 1.0
    };

    let s_16d = compute_s(df_16d);
    let s_32d = compute_s(df_32d);
    let s_64d = compute_s(df_64d);

    println!("===================================================");
    println!("🔥 Topo-Spectral Coupling Analysis:");
    println!("   (Mapping Discrete ZD Topology to Continuous Quantum Gravity)");
    println!("   16D Effective Scale s: {:.6}", s_16d);
    println!("   32D Effective Scale s: {:.6}", s_32d);
    println!("   64D Effective Scale s: {:.6}", s_64d);

    // As dimension increases, the fractal dimension drops, which pushes the 
    // continuous Calcagni scale factor `s` towards the UV limit (s -> 0).
    
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    
    let out_path = out_dir.join("spectral_dimension_manifold_coupling.csv");
    let mut file = File::create(&out_path)?;
    writeln!(file, "manifold,df,calcagni_s")?;
    writeln!(file, "16,{},{}", df_16d, s_16d)?;
    writeln!(file, "32,{},{}", df_32d, s_32d)?;
    writeln!(file, "64,{},{}", df_64d, s_64d)?;

    println!("✅ Exact spectral mappings generated at {}", out_path.display());
    Ok(())
}
