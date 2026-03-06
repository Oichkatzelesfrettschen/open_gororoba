use pathion_ellip::pathion_shadow::pathion_deflection;
use pathion_ellip::diagonalizer::PathionDiagonalizer;

fn main() -> anyhow::Result<()> {
    println!("=== Pathion Shadow Analysis: 16-Plane Spectral Splitting ===");
    
    let mass = 1.0;
    // Extreme spin along the real axis, but with Pathion fluctuations
    let mut spin = [0.0; 32];
    spin[0] = 0.99; // a = 0.99
    for (i, s) in spin.iter_mut().enumerate().skip(1) { *s = 0.01 * (i as f64).sin(); }
    
    // Impact parameters
    let mut xi = [0.0; 32];
    xi[0] = 2.0;
    let mut eta = [0.0; 32];
    eta[0] = 5.0;
    
    println!("Computing 32D Pathion Deflection via 16-Plane Carlson Integrals...");
    let result = pathion_deflection(mass, &spin, &xi, &eta)?;
    
    let diag = PathionDiagonalizer::new();
    let planes = diag.project(&result);
    
    println!("\nDecomposed Spectral Components (16 Complex Planes):");
    println!("{:<8} | {:<20} | {:<20}", "Plane", "Real (Deflection)", "Imag (Rotation)");
    println!("{:-<8}-|-{:-<20}-|-{:-<20}", "", "", "");
    
    for (i, p) in planes.iter().enumerate() {
        println!("{:<8} | {:<20.8} | {:<20.8}", i, p.re, p.im);
    }
    
    let avg_defl = result.iter().sum::<f64>() / 32.0;
    let variance = result.iter().map(|&x| (x - avg_defl).powi(2)).sum::<f64>() / 32.0;
    
    println!("\nPathion Statistical Profile:");
    println!("  Mean Deflection:  {:.8}", avg_defl);
    println!("  Pathion Variance: {:.8e} (The 'Sub-ring' potential)", variance);
    
    if variance > 1e-10 {
        println!("  [VERDICT] Non-zero Pathion variance confirmed. Shadow will splinter into 15+1 sub-rings.");
    }
    
    Ok(())
}
