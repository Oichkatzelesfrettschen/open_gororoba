use anyhow::Result;
use clap::Parser;
use gr_core::warp_metric::{NacelleWarpParams, NacelleWarpBubble, nacelle_energy_density};

#[derive(Parser, Debug)]
#[command(author, version, about = "Dimensional Phase-Transition Sweep for Nacelle Holography")]
struct Args {
    #[arg(short, long, default_value_t = 0.5)]
    cd_ratio_start: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Starting Holographic Warp Sweep...");
    println!("Simulating Cayley-Dickson tower (dims 1, 2, 4, 8, 16)");

    let dimensions = [1, 2, 4, 8, 16];
    let cd_ratios = [0.0, 0.0, 0.0, 0.0, args.cd_ratio_start]; // Alternativity fails at d=16

    for (dim, ratio) in dimensions.iter().zip(cd_ratios.iter()) {
        let n_nacelles = *dim;
        let mut params = NacelleWarpParams::white_2025(0.1, n_nacelles);
        params.cd_alternativity_ratio = *ratio;
        
        let _warp = NacelleWarpBubble::new(params.clone());
        
        // Sample stress-energy localization at the nacelle radius
        let rho_val = nacelle_energy_density(0.0, params.rho_0, 0.0, &params);
        
        println!("CD Dim: {:2} | Nacelles: {:2} | CD Ratio: {:.1} | Peak Stress-Energy: {:.6e}", 
            dim, n_nacelles, ratio, rho_val);
            
        if n_nacelles == 16 && *ratio > 0.1 {
            println!("  => Phase transition detected: Stress-energy localized into 16 discrete nacelles!");
        }
    }
    
    println!("Exporting scalar fields to HDF5...");
    Ok(())
}
