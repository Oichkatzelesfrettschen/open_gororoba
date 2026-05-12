//! Spectroscopy bands (L9g): fixed-N, all-S band structure of the
//! strutted emanation table.
//!
//! At CD level N, strut constants S in [1, G) form "bands" of width 8
//! (the sedenion generator group size). Band b contains
//! S = 8b+1 .. min(8b+8, G-1).
//!
//! Band 0 always contains the mandala struts (S=1..7) and sedenion
//! generators. Higher bands contain sky struts and may contain one
//! generator (power of 2).
//!
//! Within each band, struts share structural similarities:
//!   - Same number of regime address prefixes (nesting depth)
//!   - Similar (often identical) DMZ counts
//!   - Compatible hide/fill involution partners
//!
//! The "flip-book" is a compact representation of how the DMZ pattern
//! varies across all struts in a band: a vector of (S, dmz_count,
//! regime_address) triples enabling quick comparison.

use super::regime_address::regime_address;
use super::strut_spectroscopy::{StrutClass, classify_strut};
use super::strutted_et::create_strutted_et;

/// Dominant behavior in a spectroscopy band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandBehavior {
    /// All struts in the band are full-fill (mandala or generator).
    FullFill,
    /// Band contains a single DMZ regime (all sky struts share one DMZ count).
    UniformSky,
    /// Band contains multiple DMZ regimes (mixed behavior).
    MixedRegime,
}

/// A single frame in the flip-book: one strut's summary within a band.
#[derive(Debug, Clone)]
pub struct FlipBookFrame {
    /// Strut constant.
    pub s: usize,
    /// Classification: Generator, Mandala, or Sky.
    pub class: StrutClass,
    /// DMZ count.
    pub dmz_count: usize,
    /// Fill ratio.
    pub fill_ratio: f64,
    /// Regime address.
    pub regime_address: Vec<u8>,
    /// Effective box-kite count (dmz_count / 24).
    pub effective_bk_count: usize,
}

/// Summary of one spectroscopy band (a group of 8 consecutive strut constants).
#[derive(Debug, Clone)]
pub struct SpectroscopyBand {
    /// Band index (0, 1, 2, ...).
    pub band_index: usize,
    /// Range of S values: [s_lo, s_hi] inclusive.
    pub s_lo: usize,
    pub s_hi: usize,
    /// Number of struts in this band.
    pub n_struts: usize,
    /// Count by class.
    pub n_generators: usize,
    pub n_mandala: usize,
    pub n_sky: usize,
    /// DMZ range across the band.
    pub dmz_min: usize,
    pub dmz_max: usize,
    /// Number of distinct DMZ counts (regimes) in this band.
    pub n_regimes: usize,
    /// Number of distinct regime addresses in this band.
    pub n_distinct_addresses: usize,
    /// Dominant behavior.
    pub behavior: BandBehavior,
    /// Whether all struts in the band are full-fill.
    pub all_full_fill: bool,
    /// Flip-book: ordered frames for each strut in the band.
    pub frames: Vec<FlipBookFrame>,
}

/// Complete spectroscopy result for a CD level.
#[derive(Debug, Clone)]
pub struct SpectroscopyResult {
    /// CD level.
    pub n: usize,
    /// Dimension = 2^N.
    pub dim: usize,
    /// Generator G = 2^(N-1).
    pub g: usize,
    /// Number of valid struts.
    pub n_struts: usize,
    /// Number of bands.
    pub n_bands: usize,
    /// Bands.
    pub bands: Vec<SpectroscopyBand>,
    /// Global: number of distinct DMZ counts across ALL struts.
    pub n_global_regimes: usize,
    /// Global: expected regime count = 2^(N-4) (de Marrais formula).
    pub expected_regime_count: usize,
}

/// Compute the full spectroscopy band analysis for CD level N.
///
/// Groups all strut constants S in [1, G) into bands of width 8,
/// classifies each band's dominant behavior, and builds a flip-book
/// of per-strut summaries.
pub fn spectroscopy_bands(n: usize) -> SpectroscopyResult {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);
    let dim = 1usize << n;
    let n_struts = g - 1;
    let n_bands = n_struts.div_ceil(8);
    let expected_regime_count = if n >= 4 { 1usize << (n - 4) } else { 1 };

    let mut bands = Vec::with_capacity(n_bands);
    let mut global_dmz_set = std::collections::BTreeSet::new();

    for band_idx in 0..n_bands {
        let s_lo = band_idx * 8 + 1;
        let s_hi = ((band_idx + 1) * 8).min(g - 1);

        let mut frames = Vec::new();
        let mut n_gen = 0usize;
        let mut n_man = 0usize;
        let mut n_sky = 0usize;
        let mut dmz_min = usize::MAX;
        let mut dmz_max = 0usize;
        let mut dmz_set = std::collections::BTreeSet::new();
        let mut addr_set = std::collections::BTreeSet::new();
        let mut all_full = true;

        for s in s_lo..=s_hi {
            let class = classify_strut(n, s);
            match class {
                StrutClass::Generator => n_gen += 1,
                StrutClass::Mandala => n_man += 1,
                StrutClass::Sky => n_sky += 1,
            }

            let et = create_strutted_et(n, s);
            let fill_ratio = if et.total_possible > 0 {
                et.dmz_count as f64 / et.total_possible as f64
            } else {
                0.0
            };
            let addr = regime_address(n, s);

            if et.dmz_count < dmz_min {
                dmz_min = et.dmz_count;
            }
            if et.dmz_count > dmz_max {
                dmz_max = et.dmz_count;
            }
            dmz_set.insert(et.dmz_count);
            global_dmz_set.insert(et.dmz_count);
            addr_set.insert(addr.clone());

            if et.dmz_count != et.total_possible {
                all_full = false;
            }

            frames.push(FlipBookFrame {
                s,
                class,
                dmz_count: et.dmz_count,
                fill_ratio,
                regime_address: addr,
                effective_bk_count: et.dmz_count / 24,
            });
        }

        let n_regimes = dmz_set.len();
        let behavior = if all_full {
            BandBehavior::FullFill
        } else if n_regimes == 1 {
            BandBehavior::UniformSky
        } else {
            BandBehavior::MixedRegime
        };

        bands.push(SpectroscopyBand {
            band_index: band_idx,
            s_lo,
            s_hi,
            n_struts: s_hi - s_lo + 1,
            n_generators: n_gen,
            n_mandala: n_man,
            n_sky,
            dmz_min,
            dmz_max,
            n_regimes,
            n_distinct_addresses: addr_set.len(),
            behavior,
            all_full_fill: all_full,
            frames,
        });
    }

    SpectroscopyResult {
        n,
        dim,
        g,
        n_struts,
        n_bands: bands.len(),
        bands,
        n_global_regimes: global_dmz_set.len(),
        expected_regime_count,
    }
}
