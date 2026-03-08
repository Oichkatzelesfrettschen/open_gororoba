/// Dissociative Field Theory (DFT) Force Mapping.
///
/// Maps Cayley-Dickson algebraic layers to physical force sectors
/// as proposed in the 2025-2026 DFT synthesis.
pub enum DftForceSector {
    Gravity,               // 1D (Reals)
    TemporalFlux,          // 2D (Complex)
    Spacetime,             // 4D (Quaternions)
    Magnetism,             // 8D (Octonions)
    Electricity,           // 16D (Sedenions)
    WeakNuclear,           // 32D (Pathions)
    StrongNuclear,         // 64D (Chingons)
    LargeScaleHomogeneity, // 256D (Voudons)
}

impl DftForceSector {
    pub fn from_dim(dim: usize) -> Option<Self> {
        match dim {
            1 => Some(Self::Gravity),
            2 => Some(Self::TemporalFlux),
            4 => Some(Self::Spacetime),
            8 => Some(Self::Magnetism),
            16 => Some(Self::Electricity),
            32 => Some(Self::WeakNuclear),
            64 => Some(Self::StrongNuclear),
            256 => Some(Self::LargeScaleHomogeneity),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Gravity => "1D (Reals): Scalar Gravity / Mass",
            Self::TemporalFlux => "2D (Complex): Temporal Flux / Phase",
            Self::Spacetime => "4D (Quaternions): Spacetime Metric (GR)",
            Self::Magnetism => "8D (Octonions): Magnetism / Strong Force (Color)",
            Self::Electricity => "16D (Sedenions): Electricity / Weak Isospin",
            Self::WeakNuclear => "32D (Pathions): Weak Nuclear Force / Intentionality",
            Self::StrongNuclear => "64D (Chingons): Strong Nuclear Force / Non-conservative Drag",
            Self::LargeScaleHomogeneity => "256D (Voudons): Global Mean Frustration / Dark Energy",
        }
    }
}

/// Maps a basis element index `i` to its corresponding force sector
/// based on the dimension doubling point it first appears in.
pub fn map_basis_to_sector(i: usize) -> DftForceSector {
    if i == 0 {
        return DftForceSector::Gravity;
    }
    if i < 2 {
        return DftForceSector::TemporalFlux;
    }
    if i < 4 {
        return DftForceSector::Spacetime;
    }
    if i < 8 {
        return DftForceSector::Magnetism;
    }
    if i < 16 {
        return DftForceSector::Electricity;
    }
    if i < 32 {
        return DftForceSector::WeakNuclear;
    }
    if i < 64 {
        return DftForceSector::StrongNuclear;
    }
    DftForceSector::LargeScaleHomogeneity
}
