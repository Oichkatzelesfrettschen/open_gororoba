//! Phonon mode analysis by point-group symmetry.
//!
//! Extracted from `crystal_symmetry.rs` as part of the #139 PH-MOD split.
//! Defines `PhononMode` plus `phonon_modes_by_symmetry()` which enumerates
//! acoustic + optical branches for n atoms in the unit cell:
//!
//!   total modes = 3n
//!     - 3 acoustic (low frequency, near omega = 0)
//!     - 3(n-1) optical (higher frequency)
//!
//! For Oh (cubic) the acoustic modes transform as T1g; for C6v (hexagonal)
//! they decompose as A1 + 2E. Other point groups fall through to a generic
//! A1/A2/A3 labelling. Optical-mode labels follow the same per-point-group
//! table.

use super::PointGroup;

/// Phonon mode information (index, irrep symbol, estimated frequency, type).
#[derive(Debug, Clone)]
pub struct PhononMode {
    /// Mode index (0 to 3n-1).
    pub index: usize,
    /// Irreducible representation symbol (e.g. "A1", "T2g", "E").
    pub irrep: String,
    /// Estimated frequency in GHz.
    pub frequency_ghz: f64,
    /// Mode type: `"acoustic"` or `"optical"`.
    pub mode_type: String,
}

/// Analyse phonon modes by symmetry for a given point group and number of
/// atoms in the unit cell. Returns 3n modes (3 acoustic + 3(n-1) optical).
pub fn phonon_modes_by_symmetry(point_group: PointGroup, n_atoms: usize) -> Vec<PhononMode> {
    let _total_modes = 3 * n_atoms;
    let mut modes = Vec::new();

    match point_group {
        PointGroup::Oh => {
            for i in 0..3 {
                modes.push(PhononMode {
                    index: i,
                    irrep: "T1g".to_string(),
                    frequency_ghz: 0.5 + 0.1 * i as f64,
                    mode_type: "acoustic".to_string(),
                });
            }
        }
        PointGroup::C6v => {
            modes.push(PhononMode {
                index: 0,
                irrep: "A1".to_string(),
                frequency_ghz: 0.3,
                mode_type: "acoustic".to_string(),
            });
            modes.push(PhononMode {
                index: 1,
                irrep: "E".to_string(),
                frequency_ghz: 0.4,
                mode_type: "acoustic".to_string(),
            });
            modes.push(PhononMode {
                index: 2,
                irrep: "E".to_string(),
                frequency_ghz: 0.4,
                mode_type: "acoustic".to_string(),
            });
        }
        _ => {
            for i in 0..3 {
                modes.push(PhononMode {
                    index: i,
                    irrep: format!("A{}", i + 1),
                    frequency_ghz: 0.5,
                    mode_type: "acoustic".to_string(),
                });
            }
        }
    }

    let n_optical = 3 * (n_atoms - 1);
    for i in 0..n_optical {
        let irrep_idx = i % 3;
        let irrep = match point_group {
            PointGroup::Oh => match irrep_idx {
                0 => "A1g".to_string(),
                1 => "T1g".to_string(),
                _ => "T2g".to_string(),
            },
            PointGroup::C6v => match irrep_idx {
                0 => "A1".to_string(),
                _ => "E".to_string(),
            },
            _ => format!("B{}", irrep_idx + 1),
        };
        modes.push(PhononMode {
            index: 3 + i,
            irrep,
            frequency_ghz: 5.0 + 1.0 * (i as f64),
            mode_type: "optical".to_string(),
        });
    }

    modes
}
