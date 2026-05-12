//! ET Sparsity Spectroscopy (L4) and strut-class spectroscopy (L4b)
//! for the strutted emanation tables.
//!
//! Per-strut DMZ count, regime grouping, associative triplet count
//! Trip_N, and the inherited / generator / sky classification of
//! strut constants live here.
//!
//! Key facts:
//! - DMZ counts are always divisible by 24 (each box-kite contributes
//!   24 directed DMZ cells).
//! - Trip_N = (2^N - 1)(2^N - 2)/6 = C(2^N-1, 2)/3.
//! - Sky struts (S > 8 and not a power of two) drive the meta-fractal
//!   skybox structure.

use std::collections::HashMap;

use super::strutted_et::create_strutted_et;

/// Per-strut DMZ count for spectroscopy analysis.
#[derive(Debug, Clone)]
pub struct StrutSpectrum {
    /// The power-of-2 exponent (dim = 2^n).
    pub n: usize,
    /// Strut constant.
    pub s: usize,
    /// Number of DMZ (filled) cells in this strut's ET.
    pub dmz_count: usize,
    /// Total possible cells (excluding diagonal and strut-opposite blanks).
    pub total_possible: usize,
    /// Fill ratio: dmz_count / total_possible.
    pub fill_ratio: f64,
}

/// Compute the DMZ spectrum across all valid strut constants for a given N.
///
/// De Marrais observes regime structure:
/// - N=4 (sedenions): 1 regime (all 7 struts yield same DMZ count)
/// - N=5 (pathions): 2 regimes (168 and 72 DMZ cells)
/// - N=6 (chingons): 4 regimes (840, 456, 168, 552)
/// - N=7: 8 regimes
///
/// DMZ counts are always divisible by 24.
pub fn et_sparsity_spectroscopy(n: usize) -> Vec<StrutSpectrum> {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);

    let mut spectra = Vec::new();
    for s in 1..g {
        let et = create_strutted_et(n, s);
        let fill_ratio = if et.total_possible > 0 {
            et.dmz_count as f64 / et.total_possible as f64
        } else {
            0.0
        };
        spectra.push(StrutSpectrum {
            n,
            s,
            dmz_count: et.dmz_count,
            total_possible: et.total_possible,
            fill_ratio,
        });
    }

    spectra
}

/// Group strut constants by DMZ count (regime detection).
///
/// Returns a map from DMZ count to the list of strut constants that yield it.
pub fn et_regimes(n: usize) -> HashMap<usize, Vec<usize>> {
    let spectra = et_sparsity_spectroscopy(n);
    let mut regimes: HashMap<usize, Vec<usize>> = HashMap::new();
    for sp in &spectra {
        regimes.entry(sp.dmz_count).or_default().push(sp.s);
    }
    regimes
}

/// Compute the associative triplet count (Trip_N) for 2^N-ions.
///
/// Formula: Trip_N = (2^N - 1)(2^N - 2) / 6 = C(2^N - 1, 2) / 3.
///
/// - N=2 (quaternions): 1
/// - N=3 (octonions): 7
/// - N=4 (sedenions): 35
/// - N=5 (pathions): 155
/// - N=6 (chingons): 651
/// - N=7 (routons): 2667
pub fn trip_count(n: usize) -> usize {
    let d = 1usize << n; // 2^N
    (d - 1) * (d - 2) / 6
}

/// The Trip-Count Two-Step: for inherited struts (S < 8), the full-fill ET
/// decomposes into exactly Trip_{N-2} complete box-kites.
///
/// De Marrais (arXiv:0704.0026, Section 2): "The maximum number of Box-Kites
/// that can fill a 2^N-ion ET = Trip_{N-2}."
///
/// This follows because the full-fill total_possible = K(K-2) where K = 2^{N-1} - 2,
/// and each box-kite contributes exactly 24 directed DMZ cells to the ET.
pub fn trip_count_two_step(n: usize) -> usize {
    assert!(n >= 4, "Trip-Count Two-Step requires N >= 4 (sedenions)");
    trip_count(n - 2)
}

/// Classify whether a strut constant S at doubling level N generates a "Sky"
/// (meta-fractal skybox structure) per de Marrais (arXiv:0704.0112, 2007).
///
/// A Sky occurs when S > 8 AND S is not a power of 2.  Powers of 2 are
/// generator-inherited struts that always yield 100% ET fill.  S <= 8 struts
/// are "sand mandala" struts inherited from lower doublings.
///
/// Note: the Complex Systems (2006) abstract erroneously states "< 8";
/// all other de Marrais sources consistently say "> 8".
pub fn is_sky_strut(s: usize) -> bool {
    s > 8 && !s.is_power_of_two()
}

/// Classify whether a strut constant S at level N is "inherited" from a
/// lower doubling and therefore guaranteed to have full ET fill (DMZ = total_possible).
///
/// A strut is inherited if it is a power of 2 (generator of some sub-doubling).
/// At level N, the inherited full-fill struts are: 1, 2, 4, ..., G/2, where
/// G = 2^(N-1).  Additionally, S = 1..7 are always full-fill at any N.
pub fn is_inherited_full_fill_strut(n: usize, s: usize) -> bool {
    // Powers of 2 less than G are always full fill
    if s.is_power_of_two() && s < (1usize << (n - 1)) {
        return true;
    }
    // S = 1..7 (sedenion struts) are always full fill at any N
    s <= 7
}

/// Classification of a strut constant within a Cayley-Dickson level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrutClass {
    /// Generator-inherited: S is a power of 2. Always full-fill.
    Generator,
    /// Mandala-inherited: S <= 7, inherited from sedenion level. Full-fill.
    Mandala,
    /// Sky: S > 8 and not a power of 2. Sparse fill (Sand Mandala pattern).
    Sky,
}

/// Detailed spectroscopy result for a single strut constant.
#[derive(Debug, Clone)]
pub struct StrutSpectroscopyEntry {
    /// Strut constant.
    pub s: usize,
    /// Classification.
    pub class: StrutClass,
    /// DMZ count.
    pub dmz_count: usize,
    /// Total possible cells.
    pub total_possible: usize,
    /// Fill ratio.
    pub fill_ratio: f64,
    /// Number of "effective box-kites": dmz_count / 24 (exact when divisible).
    pub effective_bk_count: usize,
    /// Whether this strut has full fill (DMZ = total_possible).
    pub is_full_fill: bool,
}

/// Classify a strut constant at level N.
pub fn classify_strut(n: usize, s: usize) -> StrutClass {
    let g = 1usize << (n - 1);
    assert!(s >= 1 && s < g, "S must be in [1, G)");
    if s.is_power_of_two() {
        StrutClass::Generator
    } else if s <= 7 {
        StrutClass::Mandala
    } else {
        StrutClass::Sky
    }
}

/// Compute detailed spectroscopy for all strut constants at level N.
///
/// For each strut S in [1, G), returns its classification, DMZ count,
/// effective box-kite count, and whether it has full fill.
pub fn strut_spectroscopy(n: usize) -> Vec<StrutSpectroscopyEntry> {
    assert!(n >= 4, "Need at least sedenions");
    let g = 1usize << (n - 1);

    let mut entries = Vec::new();
    for s in 1..g {
        let et = create_strutted_et(n, s);
        let fill_ratio = if et.total_possible > 0 {
            et.dmz_count as f64 / et.total_possible as f64
        } else {
            0.0
        };
        entries.push(StrutSpectroscopyEntry {
            s,
            class: classify_strut(n, s),
            dmz_count: et.dmz_count,
            total_possible: et.total_possible,
            fill_ratio,
            effective_bk_count: et.dmz_count / 24,
            is_full_fill: et.dmz_count == et.total_possible,
        });
    }
    entries
}
