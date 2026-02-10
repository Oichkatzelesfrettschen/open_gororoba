use algebra_core::analysis::boxkites::compute_frustration_ratio;
use algebra_core::construction::cayley_dickson::{cd_basis_mul_sign_split_iter, CdSignature};
use std::time::Instant;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .unwrap_or(default)
}

#[test]
fn test_split_octonion_psi_fraction() {
    // 1. Construct Split-Octonion Signature (dim=8, all gamma=+1)
    let dim = 8;
    let sig = CdSignature::split(dim);
    assert!(sig.is_split());
    assert_eq!(sig.dim(), 8);

    // 2. Compute full Psi matrix (sign table)
    // psi[p, q] = sign(e_p * e_q)
    let mut neg_count = 0;
    let mut total_count = 0;

    for p in 0..dim {
        for q in 0..dim {
            let sign = cd_basis_mul_sign_split_iter(dim, p, q, &sig);
            if sign == -1 {
                neg_count += 1;
            }
            total_count += 1;
        }
    }

    let fraction = neg_count as f64 / total_count as f64;
    println!(
        "Split-Octonion (dim=8) negative sign fraction: {}/{} = {}",
        neg_count, total_count, fraction
    );

    // 3. Verify exact 3/8 ratio
    // 3/8 = 0.375.
    // 24 / 64 = 0.375.
    assert_eq!(neg_count, 24, "Should have exactly 24 negative entries");
    assert_eq!(total_count, 64);
    assert!(
        (fraction - 0.375).abs() < 1e-10,
        "Fraction should be exactly 3/8"
    );
}

#[test]
fn test_split_octonion_attractor_coincidence() {
    // This test documents and verifies the numerical coincidence between
    // the Split-Octonion psi=1 fraction and the high-dim standard CD frustration ratio.

    // A. Split-Octonion Fraction = 0.375 (Verified above)
    let split_target = 0.375;

    // B. Standard CD Frustration Ratio Convergence
    // We compute low-dim values to establish the trend.
    // High-dim values (dim=1024 -> 0.378) are expensive to compute in a unit test,
    // so we verify the low-dim trajectory here.

    let dims = [16, 32, 64];
    let mut ratios = Vec::new();

    println!("Standard CD Frustration Ratios:");
    for &d in &dims {
        let res = compute_frustration_ratio(d);
        ratios.push(res.frustration_ratio);
        println!("dim={}: {:.6}", d, res.frustration_ratio);
    }

    // dim=16: 0.0 (Coboundary)
    assert!(ratios[0] < 1e-9);

    // dim=32: ~0.307
    assert!((ratios[1] - 0.307).abs() < 0.01);

    // dim=64: ~0.377 (Overshoot/Peak approach)
    assert!((ratios[2] - 0.377).abs() < 0.01);

    // Low-dim trajectory should rise as non-associative structure develops.
    assert!(ratios[1] > ratios[0], "dim=32 ratio must exceed dim=16");
    assert!(ratios[2] > ratios[1], "dim=64 ratio must exceed dim=32");

    // The insight (I-040/C-529) claims convergence to ~0.375 from above after peaking.
    // We assert that the dim=64 value is close to the split target.
    let diff_64 = (ratios[2] - split_target).abs();
    println!("Diff at dim=64: {:.6}", diff_64);

    // At dim=64, the ratio should still sit above the split target but already be close.
    assert!(
        ratios[2] > split_target,
        "dim=64 should approach 3/8 from above"
    );
    assert!(diff_64 < 0.01, "dim=64 should be within 0.01 of 3/8");
}

#[test]
fn test_split_octonion_attractor_regression_dim_128_256_guarded() {
    // Runtime guard knobs (seconds) for heavier regression checks.
    // Override locally/CI via env vars when hardware budget differs.
    let skip_heavy = std::env::var("CD_ATTRACTOR_SKIP_HEAVY")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if skip_heavy {
        eprintln!("Skipping dim 128/256 attractor regression (CD_ATTRACTOR_SKIP_HEAVY=1).");
        return;
    }

    let per_dim_budget_s = env_f64("CD_ATTRACTOR_PER_DIM_BUDGET_S", 20.0);
    let total_budget_s = env_f64("CD_ATTRACTOR_TOTAL_BUDGET_S", 30.0);
    let max_diff_128 = env_f64("CD_ATTRACTOR_MAX_DIFF_128", 0.02);
    let max_diff_256 = env_f64("CD_ATTRACTOR_MAX_DIFF_256", 0.02);
    let split_target = 0.375;
    let dims = [128usize, 256usize];

    let mut ratios = Vec::with_capacity(dims.len());
    let suite_start = Instant::now();

    println!("Guarded high-dim attractor regression:");
    for &dim in &dims {
        let started = Instant::now();
        let res = compute_frustration_ratio(dim);
        let elapsed_s = started.elapsed().as_secs_f64();
        println!(
            "dim={}: ratio={:.6}, elapsed={:.3}s",
            dim, res.frustration_ratio, elapsed_s
        );

        assert!(
            elapsed_s <= per_dim_budget_s,
            "dim={} exceeded per-dim budget ({:.3}s > {:.3}s). Override CD_ATTRACTOR_PER_DIM_BUDGET_S if needed.",
            dim,
            elapsed_s,
            per_dim_budget_s
        );
        assert!(
            res.frustration_ratio > split_target,
            "dim={} ratio should remain above 3/8 during observed convergence window",
            dim
        );
        assert!(
            res.frustration_ratio < 0.39,
            "dim={} ratio should remain in expected high-dim envelope (< 0.39)",
            dim
        );
        ratios.push(res.frustration_ratio);
    }

    let total_elapsed_s = suite_start.elapsed().as_secs_f64();
    assert!(
        total_elapsed_s <= total_budget_s,
        "dim 128/256 attractor suite exceeded total budget ({:.3}s > {:.3}s). Override CD_ATTRACTOR_TOTAL_BUDGET_S if needed.",
        total_elapsed_s,
        total_budget_s
    );

    let diff_128 = (ratios[0] - split_target).abs();
    let diff_256 = (ratios[1] - split_target).abs();
    println!("diff_128={:.6}, diff_256={:.6}", diff_128, diff_256);

    assert!(
        diff_128 <= max_diff_128,
        "dim=128 distance to 3/8 ({:.6}) exceeded limit {:.6}",
        diff_128,
        max_diff_128
    );
    assert!(
        diff_256 <= max_diff_256,
        "dim=256 distance to 3/8 ({:.6}) exceeded limit {:.6}",
        diff_256,
        max_diff_256
    );
    // Allow mild oscillation while still requiring no large divergence in the high-dim regime.
    assert!(
        ratios[1] <= ratios[0] + 0.01,
        "dim=256 should not drift far above dim=128 in attractor regime"
    );
}

#[test]
fn test_split_octonion_attractor_delta_shrink_128_256_512_guarded() {
    // Deep guard: include dim=512 only on explicit opt-in.
    let include_512 = std::env::var("CD_ATTRACTOR_INCLUDE_512")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(false);
    if !include_512 {
        eprintln!("Skipping dim 128/256/512 attractor delta-shrink regression (set CD_ATTRACTOR_INCLUDE_512=1 to enable).");
        return;
    }

    let split_target = 0.375;
    let per_dim_budget_s = env_f64("CD_ATTRACTOR_PER_DIM_BUDGET_S_512", 80.0);
    let total_budget_s = env_f64("CD_ATTRACTOR_TOTAL_BUDGET_S_512", 140.0);
    let monotone_slack = env_f64("CD_ATTRACTOR_DELTA_MONOTONE_SLACK", 0.001);
    let dims = [128usize, 256usize, 512usize];

    let mut deltas = Vec::with_capacity(dims.len());
    let suite_start = Instant::now();

    println!("Guarded attractor delta-shrink regression (128/256/512):");
    for &dim in &dims {
        let started = Instant::now();
        let res = compute_frustration_ratio(dim);
        let elapsed_s = started.elapsed().as_secs_f64();
        let delta = res.frustration_ratio - split_target;
        println!(
            "dim={}: ratio={:.6}, delta={:.6}, elapsed={:.3}s",
            dim, res.frustration_ratio, delta, elapsed_s
        );

        assert!(
            elapsed_s <= per_dim_budget_s,
            "dim={} exceeded per-dim budget ({:.3}s > {:.3}s). Override CD_ATTRACTOR_PER_DIM_BUDGET_S_512 if needed.",
            dim,
            elapsed_s,
            per_dim_budget_s
        );
        assert!(
            res.frustration_ratio > split_target,
            "dim={} ratio should stay above 3/8 in the observed attractor regime",
            dim
        );
        assert!(
            res.frustration_ratio < 0.39,
            "dim={} ratio should stay within expected high-dim envelope (< 0.39)",
            dim
        );
        deltas.push(delta);
    }

    let total_elapsed_s = suite_start.elapsed().as_secs_f64();
    assert!(
        total_elapsed_s <= total_budget_s,
        "dim 128/256/512 suite exceeded total budget ({:.3}s > {:.3}s). Override CD_ATTRACTOR_TOTAL_BUDGET_S_512 if needed.",
        total_elapsed_s,
        total_budget_s
    );

    assert!(
        deltas[1] <= deltas[0] + monotone_slack,
        "delta(256) should not exceed delta(128) beyond slack"
    );
    assert!(
        deltas[2] <= deltas[1] + monotone_slack,
        "delta(512) should not exceed delta(256) beyond slack"
    );
}
