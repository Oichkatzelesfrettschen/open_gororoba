//! Balloon ride (L9f): fixed-S, increasing-N emanation table sequence.
//!
//! A "balloon ride" fixes a strut constant S and observes its ET as we
//! ascend through CD levels N, N+1, N+2, ... . This reveals the
//! recursive structure:
//!
//!   - Mandala struts (all regime-address bits = 0) maintain 100% fill
//!     at every level.
//!   - Sky struts gain one extra regime-address prefix per level
//!     (always `[0]`).
//!   - DMZ growth ratio converges to 4 as K grows (addressable cells
//!     ~4x per doubling).
//!   - Theorem 11 embedding holds at every transition.
//!   - Fill ratio for sky struts is monotonically non-decreasing.
//!
//! The minimum valid level for strut S is the smallest N where
//! G = 2^(N-1) > S.

use super::{
    regime_address::regime_address, strut_spectroscopy::is_sky_strut,
    strutted_et::create_strutted_et,
};

/// One step of a balloon ride: the ET data for strut S at level N.
#[derive(Debug, Clone)]
pub struct BalloonRideStep {
    /// CD level (dimension = 2^N).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Tone-row parameters: G = 2^(N-1), K = G-2, X = G+S.
    pub g: usize,
    pub k: usize,
    pub x: usize,
    /// DMZ count in the K x K ET.
    pub dmz_count: usize,
    /// Total addressable cells = K * (K - 2).
    pub addressable: usize,
    /// Fill ratio = dmz_count / addressable.
    pub fill_ratio: f64,
    /// Regime address (recursive bit decomposition of S in base-G).
    pub regime_address: Vec<u8>,
    /// Whether S is a sky strut (at the sedenion level, S >= 8).
    pub is_sky: bool,
    /// DMZ growth ratio from previous level (0.0 for the first step).
    pub dmz_growth_ratio: f64,
}

/// Complete balloon ride result: a sequence of steps at increasing N.
#[derive(Debug, Clone)]
pub struct BalloonRide {
    /// Strut constant held fixed throughout the ride.
    pub s: usize,
    /// Sequence of steps at increasing N.
    pub steps: Vec<BalloonRideStep>,
    /// Whether fill ratio is monotonically non-decreasing across all steps.
    pub fill_monotone: bool,
    /// Whether all mandala levels have 100% fill (only meaningful if !is_sky).
    pub mandala_full_fill: bool,
}

/// Minimum valid CD level for a given strut constant S.
/// Returns the smallest N such that G = 2^(N-1) > S.
pub fn min_level_for_strut(s: usize) -> usize {
    // G = 2^(N-1) must be > S, so N-1 > log2(S), so N > log2(S) + 1.
    // For S=0, undefined; for S=1..7, N=4 (G=8>7); for S=8..15, N=5 (G=16>15).
    assert!(s >= 1, "Strut constant must be >= 1");
    let bits = u32::BITS - (s as u32).leading_zeros();
    (bits as usize) + 1
}

/// Perform a balloon ride: compute ETs for fixed S at levels n_start..=n_end.
///
/// Panics if n_start < min_level_for_strut(s).
pub fn balloon_ride(s: usize, n_start: usize, n_end: usize) -> BalloonRide {
    let min_n = min_level_for_strut(s);
    assert!(
        n_start >= min_n,
        "n_start={} < min_level_for_strut({})={}",
        n_start,
        s,
        min_n
    );
    assert!(n_end >= n_start, "n_end must be >= n_start");

    let is_sky = is_sky_strut(s);
    let mut steps = Vec::with_capacity(n_end - n_start + 1);
    let mut prev_dmz: Option<usize> = None;

    for n in n_start..=n_end {
        let et = create_strutted_et(n, s);
        let g = et.tone_row.g;
        let k = et.tone_row.k;
        let x = et.tone_row.x;
        let addressable = k * (k - 2);
        let fill_ratio = et.dmz_count as f64 / addressable as f64;
        let addr = regime_address(n, s);

        let dmz_growth_ratio = match prev_dmz {
            Some(pd) if pd > 0 => et.dmz_count as f64 / pd as f64,
            _ => 0.0,
        };

        steps.push(BalloonRideStep {
            n,
            s,
            g,
            k,
            x,
            dmz_count: et.dmz_count,
            addressable,
            fill_ratio,
            regime_address: addr,
            is_sky,
            dmz_growth_ratio,
        });

        prev_dmz = Some(et.dmz_count);
    }

    // Check fill monotonicity: fill_ratio[i] <= fill_ratio[i+1]
    let fill_monotone = steps
        .windows(2)
        .all(|w| w[0].fill_ratio <= w[1].fill_ratio + 1e-12);

    // Mandala full fill: all steps have fill_ratio == 1.0 (within tolerance)
    let mandala_full_fill = steps
        .iter()
        .all(|step| (step.fill_ratio - 1.0).abs() < 1e-12);

    BalloonRide {
        s,
        steps,
        fill_monotone,
        mandala_full_fill,
    }
}
