//! eht-pathion-separation: Compute EHT angular separation of Pathion sub-rings.
//!
//! Breakthrough Phase 3:
//! Predicts the observable micro-arcsecond splitting of the Kerr photon ring
//! for M87* and Sgr A* due to 32D Pathion non-associativity.

use pathion_ellip::pathion_shadow::pathion_deflection;

const G: f64 = 6.67430e-20; // km^3 kg^-1 s^-2
const C: f64 = 299792.458;  // km s^-1
const M_SUN: f64 = 1.98847e30; // kg
const MPC_TO_KM: f64 = 3.08567758e19;
const KPC_TO_KM: f64 = 3.08567758e16;
const RAD_TO_UAS: f64 = 180.0 * 3600.0 * 1e6 / std::f64::consts::PI;

struct BlackHole {
    name: &'static str,
    mass_solar: f64,
    dist_km: f64,
}

fn main() -> anyhow::Result<()> {
    println!("=== Phase 3: EHT Pathion Shadow Angular Separation ===");

    let targets = vec![
        BlackHole {
            name: "M87*",
            mass_solar: 6.5e9,
            dist_km: 16.8 * MPC_TO_KM,
        },
        BlackHole {
            name: "Sgr A*",
            mass_solar: 4.15e6,
            dist_km: 8.17 * KPC_TO_KM,
        },
    ];

    for bh in targets {
        let rs = (G * bh.mass_solar * M_SUN) / (C * C);
        let scale_uas = (rs / bh.dist_km) * RAD_TO_UAS;
        
        println!("\nTarget: {}", bh.name);
        println!("  Schwarzschild Radius: {:.2} km", rs);
        println!("  Angular Scale (Rs/D): {:.2} uas", scale_uas);

        // Pathion Deflection Calculation
        let mut spin = [0.0; 32];
        spin[0] = 0.94; // Extreme Kerr a=0.94
        for (i, s) in spin.iter_mut().enumerate().skip(1) { *s = 0.05 * (i as f64).cos(); }
        
        let mut xi = [0.0; 32]; xi[0] = 2.0;
        let mut eta = [0.0; 32]; eta[0] = 5.0;

        let result = pathion_deflection(1.0, &spin, &xi, &eta)?;
        
        // Calculate the angular variance in uas
        let mean_defl = result.iter().sum::<f64>() / 32.0;
        let p_variations: Vec<f64> = result.iter().map(|&x| (x - mean_defl).abs()).collect();
        let max_separation = p_variations.iter().cloned().fold(f64::NAN, f64::max) * scale_uas;

        println!("  Max Sub-ring Separation: {:.4} uas", max_separation);
        
        if max_separation > 1.0 {
            println!("  [OBSERVABLE] Separation exceeds EHT resolution floor (1 uas).");
        } else {
            println!("  [SUB-RESOLUTION] Separation hidden within diffraction limit.");
        }
    }

    Ok(())
}
