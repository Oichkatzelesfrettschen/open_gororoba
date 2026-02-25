//! Integration tests for Son & Chekhova (2026) SFWM reproduction.
//!
//! These tests verify the key physical predictions of the paper using
//! both paper-calibrated and Sellmeier-derived wavevector mismatches.

use optics_core::{
    SfwmMaterialParams, coherence_length, maker_fringe_sweep, rate_ratio_with_dk,
    rate_sweep_with_dk,
};

#[test]
fn test_paper_dominance_ratio_at_10um() {
    let p = SfwmMaterialParams::linbo3_paper_calibrated();
    let wm = p.paper_wavevector_mismatches();
    let r = rate_ratio_with_dk(&p, &wm);

    // Paper reports R_cas/R_dir ~ 0.048 at L = 10 um.
    // Allow tolerance to 0.15 for susceptibility value uncertainty.
    assert!(
        r.rate_ratio_cas_to_dir < 0.15,
        "R_cas/R_dir = {:.4} should be < 0.15 (paper ~ 0.048)",
        r.rate_ratio_cas_to_dir
    );
    // Direct must dominate
    assert!(
        r.a_dir_sq > r.a_cas_sq,
        "Direct amplitude should dominate cascaded"
    );
}

#[test]
fn test_maker_fringes_sfwm_monotonic_shg_oscillates() {
    let p = SfwmMaterialParams::linbo3_paper_calibrated();
    let wm = p.paper_wavevector_mismatches();
    let l_coh_sfwm = coherence_length(wm.dk_sfwm);

    let thicknesses: Vec<f64> = (1..=200).map(|i| i as f64 * 0.5).collect();
    let fringes = maker_fringe_sweep(&p, &thicknesses);

    // SFWM |F|^2 should be monotonically increasing below 60% of L_coh.
    // F(L) = sin(dk*L/2)/(dk/2) peaks at dk*L/2 = pi/2 => L = L_coh,
    // and the sinc envelope starts flattening around 60-70% of peak.
    let below_coh: Vec<_> = fringes.iter().filter(|f| f.0 < l_coh_sfwm * 0.6).collect();
    for w in below_coh.windows(2) {
        assert!(
            w[1].1 >= w[0].1 * 0.99,
            "SFWM |F|^2 should be monotonic below L_coh: at L={:.1} um, {:.4} < {:.4}",
            w[1].0,
            w[1].1,
            w[0].1
        );
    }

    // SHG |F|^2 should oscillate (Maker fringes)
    let shg_vals: Vec<f64> = fringes.iter().map(|f| f.2).collect();
    let mut sign_changes = 0;
    for w in shg_vals.windows(3) {
        if (w[2] - w[1]) * (w[1] - w[0]) < 0.0 {
            sign_changes += 1;
        }
    }
    assert!(
        sign_changes >= 5,
        "SHG should oscillate with multiple Maker fringes: {} sign changes",
        sign_changes
    );
}

#[test]
fn test_rate_sweep_direct_dominates_below_30um() {
    let p = SfwmMaterialParams::linbo3_paper_calibrated();
    let wm = p.paper_wavevector_mismatches();
    let thicknesses: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let rates = rate_sweep_with_dk(&p, &wm, &thicknesses);

    for &(l, r_dir, r_cas) in &rates {
        assert!(
            r_dir > r_cas,
            "Direct should dominate cascaded at L = {:.0} um: R_dir = {:.4e}, R_cas = {:.4e}",
            l,
            r_dir,
            r_cas
        );
    }
}

#[test]
fn test_maker_fringe_period_consistent_with_l_coh_shg() {
    let p = SfwmMaterialParams::linbo3_paper_calibrated();
    let wm = p.paper_wavevector_mismatches();
    let l_coh_shg = coherence_length(wm.dk_shg);

    // The SHG Maker fringe period should be ~2 * L_coh_SHG
    let thicknesses: Vec<f64> = (1..=500).map(|i| i as f64 * 0.1).collect();
    let fringes = maker_fringe_sweep(&p, &thicknesses);

    // Find first two peaks of SHG |F|^2
    let shg_vals: Vec<(f64, f64)> = fringes.iter().map(|f| (f.0, f.2)).collect();
    let mut peaks = vec![];
    for i in 1..shg_vals.len() - 1 {
        if shg_vals[i].1 > shg_vals[i - 1].1 && shg_vals[i].1 > shg_vals[i + 1].1 {
            peaks.push(shg_vals[i].0);
        }
    }

    if peaks.len() >= 2 {
        let period = peaks[1] - peaks[0];
        // Maker fringe period = 2 * L_coh for SHG
        let expected = 2.0 * l_coh_shg;
        let rel_err = (period - expected).abs() / expected;
        assert!(
            rel_err < 0.15,
            "SHG Maker fringe period {:.2} um vs expected {:.2} um (2*L_coh), rel err {:.2}%",
            period,
            expected,
            rel_err * 100.0
        );
    }
}

#[test]
fn test_sellmeier_coherence_lengths_order() {
    // Even with Sellmeier-derived dk, the hierarchy L_coh_SFWM >> L_coh_SHG ~ L_coh_SPDC
    // must hold for the thin-layer dominance argument to work.
    let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
    let wm = p.wavevector_mismatches();

    let l_sfwm = coherence_length(wm.dk_sfwm);
    let l_shg = coherence_length(wm.dk_shg);
    let l_spdc = coherence_length(wm.dk_spdc);

    assert!(
        l_sfwm > 3.0 * l_shg,
        "L_coh_SFWM ({:.1}) should be >3x L_coh_SHG ({:.1})",
        l_sfwm,
        l_shg
    );
    assert!(
        l_sfwm > 3.0 * l_spdc,
        "L_coh_SFWM ({:.1}) should be >3x L_coh_SPDC ({:.1})",
        l_sfwm,
        l_spdc
    );
}
