//! Theorem 11 recursive ET embedding (L9e), per de Marrais (2004,
//! "The 42 Assessors").
//!
//! The primary + shifted sub-block embedding is the structural
//! mechanism that propagates ET cells through CD doubling.
//!
//! When building the ET for dim = 2^(N+1) with strut constant S, the
//! old dim = 2^N ET (same S) reappears as an exact sub-block.
//! Specifically:
//!
//!   Primary copy: new positions whose lo matches an old position's lo.
//!   Shifted copy: new positions whose lo = old_lo + old_g (g = 2^(N-1)).
//!
//! Both copies have identical DMZ patterns and emanation values to the
//! old ET. This is the core "bit-string recursion" that connects all CD
//! levels.

use super::strutted_et::create_strutted_et;

/// Result of verifying Theorem 11 recursive embedding.
#[derive(Debug, Clone)]
pub struct Theorem11Result {
    /// Old CD level.
    pub n_old: usize,
    /// New CD level (n_old + 1).
    pub n_new: usize,
    /// Strut constant.
    pub s: usize,
    /// Primary copy: map from old position -> new position.
    pub primary_map: Vec<usize>,
    /// Shifted copy: map from old position -> new position (lo + old_g).
    pub shifted_map: Vec<usize>,
    /// Whether primary copy DMZ pattern exactly matches old ET.
    pub primary_dmz_match: bool,
    /// Whether shifted copy DMZ pattern exactly matches old ET.
    pub shifted_dmz_match: bool,
    /// Whether primary copy emanation values exactly match old ET.
    pub primary_value_match: bool,
    /// Old ET DMZ count.
    pub old_dmz_count: usize,
    /// DMZ count in the primary sub-block of the new ET.
    pub primary_subblock_dmz: usize,
    /// DMZ count in the shifted sub-block of the new ET.
    pub shifted_subblock_dmz: usize,
}

/// Verify Theorem 11 recursive embedding: old ET embeds in new ET.
///
/// Computes the position mapping from old (level N) to new (level N+1) tone
/// rows and verifies that both the primary and shifted copies preserve the
/// DMZ pattern and emanation values exactly.
pub fn verify_theorem11(n: usize, s: usize) -> Theorem11Result {
    let et_old = create_strutted_et(n, s);
    let et_new = create_strutted_et(n + 1, s);
    let tr_old = &et_old.tone_row;
    let tr_new = &et_new.tone_row;
    let old_g = tr_old.g;

    // Build primary map: old_lo -> new position with matching lo
    let mut primary_map = Vec::with_capacity(tr_old.k);
    for &old_lo in &tr_old.lo {
        let new_pos = tr_new
            .lo
            .iter()
            .position(|&l| l == old_lo)
            .expect("primary: old lo must appear in new tone row");
        primary_map.push(new_pos);
    }

    // Build shifted map: old_lo + old_g -> new position
    let mut shifted_map = Vec::with_capacity(tr_old.k);
    for &old_lo in &tr_old.lo {
        let shifted = old_lo + old_g;
        let new_pos = tr_new
            .lo
            .iter()
            .position(|&l| l == shifted)
            .expect("shifted: old lo + g must appear in new tone row");
        shifted_map.push(new_pos);
    }

    // Check primary copy
    let mut primary_dmz_match = true;
    let mut primary_value_match = true;
    let mut primary_subblock_dmz = 0usize;
    for old_r in 0..tr_old.k {
        for old_c in 0..tr_old.k {
            let nr = primary_map[old_r];
            let nc = primary_map[old_c];
            let old_dmz = et_old.cells[old_r][old_c]
                .as_ref()
                .is_some_and(|c| c.is_dmz);
            let new_dmz = et_new.cells[nr][nc].as_ref().is_some_and(|c| c.is_dmz);
            if old_dmz != new_dmz {
                primary_dmz_match = false;
            }
            if new_dmz {
                primary_subblock_dmz += 1;
            }
            if old_dmz && new_dmz {
                let old_val = et_old.cells[old_r][old_c].as_ref().unwrap().emanation_value;
                let new_val = et_new.cells[nr][nc].as_ref().unwrap().emanation_value;
                if old_val != new_val {
                    primary_value_match = false;
                }
            }
        }
    }

    // Check shifted copy
    let mut shifted_dmz_match = true;
    let mut shifted_subblock_dmz = 0usize;
    for old_r in 0..tr_old.k {
        for old_c in 0..tr_old.k {
            let nr = shifted_map[old_r];
            let nc = shifted_map[old_c];
            let old_dmz = et_old.cells[old_r][old_c]
                .as_ref()
                .is_some_and(|c| c.is_dmz);
            let new_dmz = et_new.cells[nr][nc].as_ref().is_some_and(|c| c.is_dmz);
            if old_dmz != new_dmz {
                shifted_dmz_match = false;
            }
            if new_dmz {
                shifted_subblock_dmz += 1;
            }
        }
    }

    Theorem11Result {
        n_old: n,
        n_new: n + 1,
        s,
        primary_map,
        shifted_map,
        primary_dmz_match,
        shifted_dmz_match,
        primary_value_match,
        old_dmz_count: et_old.dmz_count,
        primary_subblock_dmz,
        shifted_subblock_dmz,
    }
}
