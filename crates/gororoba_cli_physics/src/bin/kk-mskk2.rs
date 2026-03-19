use anyhow::Result;
use num_complex::Complex64;
use materials_core::kramers_kronig::{kk_standard_from_im, kk_sskk_from_im, kk_mskk2_from_im};

fn eps_hn(omega: f64) -> Complex64 {
    let eps_inf = 2.05;
    let delta_eps = 3.6;
    let tau = 7e-1; // Adjusted tau to put peak in measurable range
    let alpha = 0.74;
    let beta = 0.65;
    
    let wta = Complex64::new(0.0, omega * tau).powf(alpha);
    let hn = delta_eps / (Complex64::new(1.0, 0.0) + wta).powf(beta);
    
    Complex64::new(eps_inf, 0.0) + hn
}

fn rms_rel(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for (va, vb) in a.iter().zip(b.iter()) {
        let err = (va - vb) / vb.abs().max(1e-3);
        sum_sq += err * err;
    }
    (sum_sq / a.len() as f64).sqrt()
}

fn interp_log(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let mut idx = 0;
    while idx < xp.len() - 1 && xp[idx + 1] < x {
        idx += 1;
    }
    if idx == xp.len() - 1 {
        return fp[idx];
    }
    let t = (x.ln() - xp[idx].ln()) / (xp[idx + 1].ln() - xp[idx].ln());
    fp[idx] + t * (fp[idx + 1] - fp[idx])
}

fn main() -> Result<()> {
    println!("Running MSKK-2 (Multi-Anchor KK) comparison...");

    let n_true = 100001;
    let mut w_true = Vec::with_capacity(n_true);
    let mut eps1_true = Vec::with_capacity(n_true);
    let mut eps2_true = Vec::with_capacity(n_true);

    // Wide spectral range for "Ground Truth"
    let log_start = -2.0_f64;
    let log_end = 8.0_f64;
    for i in 0..n_true {
        let l = log_start + (log_end - log_start) * (i as f64) / (n_true as f64 - 1.0);
        let w = 10.0_f64.powf(l);
        w_true.push(w);
        let eps = eps_hn(w);
        eps1_true.push(eps.re);
        eps2_true.push(eps.im);
    }

    // Measurement bands (where we pretend to have data)
    let bands = vec![
        (1e-1, 1e7, "[1e-01,1e+07] (Wide)"),
        (1e0, 1e6,  "[1e+00,1e+06] (Medium)"),
        (5e0, 1e5,  "[5e+00,1e+05] (Narrow)"),
    ];

    // Evaluate in a central region
    let n_eval = 200;
    let mut omega_eval = Vec::with_capacity(n_eval);
    let log_eval_start = 1.0_f64; // 1e1
    let log_eval_end = 4.0_f64;   // 1e4
    for i in 0..n_eval {
        let l = log_eval_start + (log_eval_end - log_eval_start) * (i as f64) / (n_eval as f64 - 1.0);
        omega_eval.push(10.0_f64.powf(l));
    }

    let true_eval: Vec<f64> = omega_eval.iter().map(|&w| interp_log(w, &w_true, &eps1_true)).collect();

    struct ResultRow {
        label: String,
        r_std: f64,
        r_ss: f64,
        r_ms: f64,
    }
    let mut results = Vec::new();

    for (wmin, wmax, label) in bands {
        let mut w_meas = Vec::new();
        let mut eps2_meas = Vec::new();
        for i in 0..n_true {
            if w_true[i] >= wmin && w_true[i] <= wmax {
                w_meas.push(w_true[i]);
                eps2_meas.push(eps2_true[i]);
            }
        }

        let eps1_std = kk_standard_from_im(&w_meas, &eps2_meas, &omega_eval, 2.05);

        let w0 = (wmin * wmax).sqrt();
        let eps1_w0 = interp_log(w0, &w_true, &eps1_true);
        let eps1_ss = kk_sskk_from_im(&w_meas, &eps2_meas, &omega_eval, w0, eps1_w0);

        let w1 = (wmin * (wmin * wmax).sqrt()).sqrt();
        let eps1_w1 = interp_log(w1, &w_true, &eps1_true);
        let eps1_ms = kk_mskk2_from_im(&w_meas, &eps2_meas, &omega_eval, w0, w1, eps1_w0, eps1_w1);

        results.push(ResultRow {
            label: label.to_string(),
            r_std: rms_rel(&eps1_std, &true_eval),
            r_ss: rms_rel(&eps1_ss, &true_eval),
            r_ms: rms_rel(&eps1_ms, &true_eval),
        });
    }

    println!("band,std_error,sskk_error,mskk2_error");
    for r in &results {
        println!("{},{:.6},{:.6},{:.6}", r.label, r.r_std, r.r_ss, r.r_ms);
    }

    Ok(())
}
