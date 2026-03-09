/// Result of a Wow! Signal zero-divisor filter application.
#[derive(Debug, Clone)]
pub struct WowFilterResult {
    pub p_value: f64,
    pub indistinguishable: bool,
    pub n_filtered: usize,
}

/// Filter zero-divisors based on the 1420 MHz Wow! Signal constraints.
pub fn wow_signal_filter(zds: &[(usize, usize, usize, usize, f64)], alpha: f64) -> WowFilterResult {
    let fundamental_freq = 1420.405751;
    let n_zds = zds.len();

    let filtered: Vec<_> = zds
        .iter()
        .filter(|&&(_, _, _, _, norm)| {
            let freq_proxy = 1.0 / (norm + 1e-15);
            (freq_proxy - fundamental_freq).abs() / fundamental_freq < 0.1
        })
        .collect();

    let n_filtered = filtered.len();
    let p_value = n_filtered as f64 / n_zds.max(1) as f64;

    WowFilterResult {
        p_value,
        indistinguishable: p_value > alpha,
        n_filtered,
    }
}
