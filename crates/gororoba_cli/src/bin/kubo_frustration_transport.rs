//! Kubo linear-response transport from Cayley-Dickson structure constants.
//!
//! Computes thermal conductivity via the Kubo formula for:
//! 1. CD Heisenberg models at dim=4 (quaternion), dim=8 (octonion)
//! 2. J1-J2 chain with alpha sweep (comparison with arXiv:1809.08429)
//! 3. Interpolated models sweeping lambda from unfrustrated to full CD
//!
//! GPU-first: uses cuSOLVER syevd + cuBLAS dgemm when available (--features gpu).
//! CPU fallback: uses nalgebra-based O(dim^3) eigenbasis transformation.
//! Both paths are O(dim^3), not the naive O(dim^4).

use std::fs;
use std::io;
use std::path::Path;
use vacuum_frustration::kubo_transport::{
    build_cd_heisenberg, build_interpolated, build_j1j2_chain, exact_diagonalize,
    graph_frustration_index, kubo_transport_optimized, thermodynamic_quantities,
    HeisenbergModel, KuboTransport,
};

/// Transport computation dispatcher: GPU-first, CPU fallback.
struct TransportDispatcher {
    #[cfg(feature = "gpu")]
    gpu_ctx: Option<vacuum_frustration::kubo_transport_gpu::GpuKuboContext>,
    use_gpu: bool,
}

impl TransportDispatcher {
    fn new() -> Self {
        #[cfg(feature = "gpu")]
        {
            match vacuum_frustration::kubo_transport_gpu::GpuKuboContext::new() {
                Ok(ctx) => {
                    println!("  Backend: GPU (cuSOLVER + cuBLAS)");
                    Self {
                        gpu_ctx: Some(ctx),
                        use_gpu: true,
                    }
                }
                Err(e) => {
                    println!("  Backend: CPU (GPU init failed: {})", e);
                    Self {
                        gpu_ctx: None,
                        use_gpu: false,
                    }
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("  Backend: CPU (optimized O(n^3))");
            Self { use_gpu: false }
        }
    }

    fn compute(&self, model: &HeisenbergModel, temperature: f64) -> KuboTransport {
        #[cfg(feature = "gpu")]
        if let Some(ref ctx) = self.gpu_ctx {
            match ctx.kubo_transport(model, temperature, 1e-10) {
                Ok(t) => return t,
                Err(e) => {
                    eprintln!("  GPU transport failed ({}), falling back to CPU", e);
                }
            }
        }
        // CPU fallback: O(dim^3) optimized algorithm
        kubo_transport_optimized(model, temperature, 1e-10)
    }
}

fn main() -> io::Result<()> {
    let output_dir = Path::new("data/kubo_transport");
    fs::create_dir_all(output_dir)?;

    println!("Kubo Linear-Response Transport from CD Structure Constants");
    println!("==========================================================");
    let dispatcher = TransportDispatcher::new();
    println!();

    // -----------------------------------------------------------------------
    // Section 1: CD Heisenberg models at different dimensions
    // -----------------------------------------------------------------------
    println!("[1/4] CD Heisenberg model transport across dimensions");
    println!("-----------------------------------------------------");

    let temperatures = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0];
    let cd_dims = [4, 8]; // dim=16 has 2^15=32768 states, skipped for now

    let mut cd_results = Vec::new();

    for &dim in &cd_dims {
        let n_sites = dim - 1;
        let hilbert = 1usize << n_sites;
        println!("  dim={} ({} sites, {} Hilbert states)", dim, n_sites, hilbert);

        let model = build_cd_heisenberg(dim, 0.0);
        let frustration = graph_frustration_index(&model);
        println!("    Frustration index: {:.4}", frustration);

        let ed = exact_diagonalize(&model);
        println!("    Ground state energy: {:.6}", ed.eigenvalues[0]);
        println!("    Energy gap: {:.6}", ed.eigenvalues[1] - ed.eigenvalues[0]);

        for &t in &temperatures {
            let thermo = thermodynamic_quantities(&ed, t);
            let transport = dispatcher.compute(&model, t);

            cd_results.push((dim, t, frustration, thermo.specific_heat,
                            transport.drude_weight_spin, transport.total_weight_energy,
                            transport.total_weight_spin));

            println!(
                "    T={:.1}: C_V={:.4}, D_S={:.6}, I0_E={:.6}, I0_S={:.6}",
                t, thermo.specific_heat, transport.drude_weight_spin,
                transport.total_weight_energy, transport.total_weight_spin
            );
        }
        println!();
    }

    // -----------------------------------------------------------------------
    // Section 2: J1-J2 chain alpha sweep (reproducing arXiv:1809.08429 Fig.1)
    // -----------------------------------------------------------------------
    println!("[2/4] J1-J2 chain alpha sweep (N=10, B=3.0, T=0.1)");
    println!("---------------------------------------------------");
    println!("  Reproducing Stolpp et al. arXiv:1809.08429 Fig.1");

    let n_chain = 10;
    let field_b = 3.0; // Near saturation
    let temp = 0.1;
    let alphas: Vec<f64> = (0..30).map(|i| i as f64 * 0.05).collect();

    let mut j1j2_results = Vec::new();

    for &alpha in &alphas {
        let model = build_j1j2_chain(n_chain, alpha, 1.0, field_b);
        let frustration = graph_frustration_index(&model);
        let transport = dispatcher.compute(&model, temp);
        let thermo = thermodynamic_quantities(&exact_diagonalize(&model), temp);

        j1j2_results.push((alpha, frustration, transport.drude_weight_spin,
                           transport.total_weight_energy, transport.total_weight_spin,
                           thermo.specific_heat));

        println!(
            "  alpha={:.2}: f={:.3}, D_S={:.6}, I0_S={:.6}, I0_E={:.6}",
            alpha, frustration, transport.drude_weight_spin,
            transport.total_weight_spin, transport.total_weight_energy
        );
    }

    // Check for non-monotonic behavior in total energy spectral weight
    let k_values: Vec<f64> = j1j2_results.iter().map(|r| r.3).collect();
    let mut has_minimum = false;
    for i in 1..k_values.len() - 1 {
        if k_values[i] < k_values[i - 1] && k_values[i] < k_values[i + 1] {
            has_minimum = true;
            println!(
                "\n  NON-MONOTONIC minimum at alpha={:.2}: I0_E={:.6}",
                alphas[i], k_values[i]
            );
            break;
        }
    }
    if !has_minimum {
        println!("\n  No non-monotonic minimum detected in I0_E(alpha)");
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 3: Lambda interpolation (unfrustrated -> CD)
    // -----------------------------------------------------------------------
    println!("[3/4] Lambda interpolation: unfrustrated -> CD octonion");
    println!("------------------------------------------------------");

    let cd8 = build_cd_heisenberg(8, 0.0);
    let lambdas: Vec<f64> = (0..21).map(|i| i as f64 * 0.05).collect();
    let temp_interp = 0.5;

    let mut interp_results = Vec::new();

    for &lam in &lambdas {
        let model = build_interpolated(&cd8, lam);
        let frustration = graph_frustration_index(&model);
        let transport = dispatcher.compute(&model, temp_interp);

        interp_results.push((lam, frustration, transport.drude_weight_spin,
                            transport.total_weight_energy, transport.total_weight_spin));

        println!(
            "  lambda={:.2}: f={:.4}, D_S={:.6}, I0_S={:.6}, I0_E={:.6}",
            lam, frustration, transport.drude_weight_spin,
            transport.total_weight_spin, transport.total_weight_energy
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 4: CD vs J1-J2 frustration-matched comparison
    // -----------------------------------------------------------------------
    println!("[4/4] Frustration-matched comparison: CD vs J1-J2");
    println!("-------------------------------------------------");

    for &dim in &cd_dims {
        let cd_model = build_cd_heisenberg(dim, 0.0);
        let f_cd = graph_frustration_index(&cd_model);
        let cd_transport = dispatcher.compute(&cd_model, 0.5);

        // Find J1-J2 alpha with matching frustration
        let mut best_alpha = 0.0;
        let mut best_diff = f64::MAX;
        for i in 0..200 {
            let alpha = i as f64 * 0.01;
            let chain = build_j1j2_chain(cd_model.n_sites, alpha, 1.0, 0.0);
            let f_chain = graph_frustration_index(&chain);
            let diff = (f_chain - f_cd).abs();
            if diff < best_diff {
                best_diff = diff;
                best_alpha = alpha;
            }
        }

        let matched_chain = build_j1j2_chain(cd_model.n_sites, best_alpha, 1.0, 0.0);
        let f_matched = graph_frustration_index(&matched_chain);
        let chain_transport = dispatcher.compute(&matched_chain, 0.5);

        println!("  dim={}: CD f={:.4}, matched J1-J2 alpha={:.2} f={:.4}", dim, f_cd, best_alpha, f_matched);
        println!("    CD:    D_S={:.6}, I0_S={:.6}, I0_E={:.6}", cd_transport.drude_weight_spin, cd_transport.total_weight_spin, cd_transport.total_weight_energy);
        println!("    Chain: D_S={:.6}, I0_S={:.6}, I0_E={:.6}", chain_transport.drude_weight_spin, chain_transport.total_weight_spin, chain_transport.total_weight_energy);
        let ratio = cd_transport.total_weight_spin / chain_transport.total_weight_spin.max(1e-20);
        println!("    I0_S ratio (CD/chain): {:.4}", ratio);
        println!();
    }

    // -----------------------------------------------------------------------
    // Write TOML output
    // -----------------------------------------------------------------------
    let toml_path = output_dir.join("kubo_transport_results.toml");
    let mut toml = String::new();
    toml.push_str("# Kubo linear-response transport from CD structure constants\n");
    toml.push_str("# First-principles derivation -- no arbitrary lambda coupling\n");
    toml.push_str("# Reference: Stolpp et al., arXiv:1809.08429 (2018)\n\n");

    toml.push_str("[metadata]\n");
    toml.push_str("method = \"Kubo linear-response, exact diagonalization\"\n");
    toml.push_str("reference = \"arXiv:1809.08429\"\n");
    toml.push_str(&format!("j1j2_chain_length = {}\n", n_chain));
    toml.push_str(&format!("j1j2_field_b = {}\n", field_b));
    toml.push_str(&format!("j1j2_temperature = {}\n", temp));
    toml.push_str(&format!("backend = \"{}\"\n\n", if dispatcher.use_gpu { "GPU" } else { "CPU" }));

    // CD results
    for &(dim, t, f, cv, ds, de, kth) in &cd_results {
        toml.push_str("[[cd_transport]]\n");
        toml.push_str(&format!("dim = {}\n", dim));
        toml.push_str(&format!("temperature = {}\n", t));
        toml.push_str(&format!("frustration = {}\n", f));
        toml.push_str(&format!("specific_heat = {}\n", cv));
        toml.push_str(&format!("drude_weight_spin = {}\n", ds));
        toml.push_str(&format!("total_weight_energy = {}\n", de));
        toml.push_str(&format!("total_weight_spin = {}\n\n", kth));
    }

    // J1-J2 results
    for &(alpha, f, ds, de, kth, cv) in &j1j2_results {
        toml.push_str("[[j1j2_transport]]\n");
        toml.push_str(&format!("alpha = {}\n", alpha));
        toml.push_str(&format!("frustration = {}\n", f));
        toml.push_str(&format!("drude_weight_spin = {}\n", ds));
        toml.push_str(&format!("total_weight_energy = {}\n", de));
        toml.push_str(&format!("total_weight_spin = {}\n", kth));
        toml.push_str(&format!("specific_heat = {}\n\n", cv));
    }

    // Interpolation results
    for &(lam, f, ds, de, kth) in &interp_results {
        toml.push_str("[[interpolation]]\n");
        toml.push_str(&format!("lambda = {}\n", lam));
        toml.push_str(&format!("frustration = {}\n", f));
        toml.push_str(&format!("drude_weight_spin = {}\n", ds));
        toml.push_str(&format!("total_weight_energy = {}\n", de));
        toml.push_str(&format!("total_weight_spin = {}\n\n", kth));
    }

    fs::write(&toml_path, &toml)?;
    println!("Output: {}", toml_path.display());
    println!("\nDone.");

    Ok(())
}
