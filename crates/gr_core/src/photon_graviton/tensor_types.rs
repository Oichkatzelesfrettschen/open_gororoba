//! Fixed-rank tensor types and kinematic validation for the Ward campaign.
//!
//! The source amplitudes use four Euclidean Lorentz indices. Keeping those
//! ranks in the type system prevents a tensor contraction from becoming an
//! undocumented scalar projection. The bilinear contractions in this module
//! use the Euclidean delta metric without complex conjugation; conjugation is
//! used only when a component norm is requested.

use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;

pub const LORENTZ_DIMENSION: usize = 4;

pub type ComplexFourVector = SVector<Complex64, LORENTZ_DIMENSION>;
pub type ComplexLorentzMatrix = SMatrix<Complex64, LORENTZ_DIMENSION, LORENTZ_DIMENSION>;

const RANK_THREE_COMPONENTS: usize = LORENTZ_DIMENSION * LORENTZ_DIMENSION * LORENTZ_DIMENSION;

/// A fixed 4x4x4 tensor with the source ordering `(mu, nu, alpha)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexRankThreeTensor {
    components: [Complex64; RANK_THREE_COMPONENTS],
}

impl Default for ComplexRankThreeTensor {
    fn default() -> Self {
        Self::zero()
    }
}

impl ComplexRankThreeTensor {
    pub fn zero() -> Self {
        Self {
            components: [Complex64::new(0.0, 0.0); RANK_THREE_COMPONENTS],
        }
    }

    pub fn from_fn(mut function: impl FnMut(usize, usize, usize) -> Complex64) -> Self {
        let mut tensor = Self::zero();
        for mu in 0..LORENTZ_DIMENSION {
            for nu in 0..LORENTZ_DIMENSION {
                for alpha in 0..LORENTZ_DIMENSION {
                    tensor.set(mu, nu, alpha, function(mu, nu, alpha));
                }
            }
        }
        tensor
    }

    pub fn get(&self, mu: usize, nu: usize, alpha: usize) -> Complex64 {
        self.components[Self::index(mu, nu, alpha)]
    }

    pub fn set(&mut self, mu: usize, nu: usize, alpha: usize, value: Complex64) {
        let index = Self::index(mu, nu, alpha);
        self.components[index] = value;
    }

    pub fn components(&self) -> &[Complex64; RANK_THREE_COMPONENTS] {
        &self.components
    }

    pub fn contract_photon(&self, photon_momentum: &ComplexFourVector) -> ComplexLorentzMatrix {
        let mut residual = ComplexLorentzMatrix::zeros();
        for mu in 0..LORENTZ_DIMENSION {
            for nu in 0..LORENTZ_DIMENSION {
                let mut component = Complex64::new(0.0, 0.0);
                for alpha in 0..LORENTZ_DIMENSION {
                    component += self.get(mu, nu, alpha) * photon_momentum[alpha];
                }
                residual[(mu, nu)] = component;
            }
        }
        residual
    }

    pub fn contract_graviton(
        &self,
        graviton_variation: &ComplexLorentzMatrix,
    ) -> ComplexFourVector {
        let mut residual = ComplexFourVector::zeros();
        for alpha in 0..LORENTZ_DIMENSION {
            let mut component = Complex64::new(0.0, 0.0);
            for mu in 0..LORENTZ_DIMENSION {
                for nu in 0..LORENTZ_DIMENSION {
                    component += graviton_variation[(mu, nu)] * self.get(mu, nu, alpha);
                }
            }
            residual[alpha] = component;
        }
        residual
    }

    pub fn symmetrized_graviton_indices(&self) -> Self {
        Self::from_fn(|mu, nu, alpha| {
            (self.get(mu, nu, alpha) + self.get(nu, mu, alpha)) * Complex64::new(0.5, 0.0)
        })
    }

    fn index(mu: usize, nu: usize, alpha: usize) -> usize {
        assert!(mu < LORENTZ_DIMENSION);
        assert!(nu < LORENTZ_DIMENSION);
        assert!(alpha < LORENTZ_DIMENSION);
        (mu * LORENTZ_DIMENSION + nu) * LORENTZ_DIMENSION + alpha
    }
}

/// Whether the photon momentum is on shell or carries declared virtuality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellMode {
    OnShell,
    OffShell,
}

impl ShellMode {
    pub fn is_on_shell(self) -> bool {
        matches!(self, Self::OnShell)
    }
}

/// Momentum conservation rule for a Ward fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MomentumRule {
    ConstantBackgroundConversion,
    DeclaredSum(ComplexFourVector),
}

impl MomentumRule {
    fn expected_sum(self) -> ComplexFourVector {
        match self {
            Self::ConstantBackgroundConversion => ComplexFourVector::zeros(),
            Self::DeclaredSum(sum) => sum,
        }
    }
}

/// Whether a source expression is bare or has received its stated subtraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenormalizationState {
    Unrenormalized,
    Renormalized,
}

/// Diagram label retained in per-diagram evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagram {
    Irreducible,
    Tadpole,
    External,
}

impl Diagram {
    pub const ALL: [Self; 3] = [Self::Irreducible, Self::Tadpole, Self::External];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Irreducible => "irreducible",
            Self::Tadpole => "tadpole",
            Self::External => "external",
        }
    }
}

/// Absolute and normalized tolerances for a retained residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualTolerance {
    pub absolute: f64,
    pub normalized: f64,
}

impl ResidualTolerance {
    pub fn new(absolute: f64, normalized: f64) -> Result<Self, KinematicsError> {
        if !absolute.is_finite() || absolute < 0.0 {
            return Err(KinematicsError::InvalidTolerance);
        }
        if !normalized.is_finite() || normalized < 0.0 {
            return Err(KinematicsError::InvalidTolerance);
        }
        Ok(Self {
            absolute,
            normalized,
        })
    }

    pub fn accepts(self, absolute_norm: f64, normalized_norm: f64) -> bool {
        absolute_norm <= self.absolute && normalized_norm <= self.normalized
    }
}

/// Kinematic data required by every tensor Ward evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WardKinematics {
    pub k: ComplexFourVector,
    pub k0: ComplexFourVector,
    pub epsilon: ComplexFourVector,
    pub epsilon0: ComplexLorentzMatrix,
    pub zeta0: ComplexFourVector,
    pub field_strength: ComplexLorentzMatrix,
    pub declared_virtuality: Complex64,
    pub shell_mode: ShellMode,
    pub momentum_rule: MomentumRule,
    pub require_zeta_transversality: bool,
    pub validation_tolerance: f64,
}

impl WardKinematics {
    // Keep the source-required vectors and matrices explicit in the constructor signature.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        k: ComplexFourVector,
        k0: ComplexFourVector,
        epsilon: ComplexFourVector,
        epsilon0: ComplexLorentzMatrix,
        zeta0: ComplexFourVector,
        field_strength: ComplexLorentzMatrix,
        declared_virtuality: Complex64,
        shell_mode: ShellMode,
        momentum_rule: MomentumRule,
        require_zeta_transversality: bool,
        validation_tolerance: f64,
    ) -> Result<Self, KinematicsError> {
        let kinematics = Self {
            k,
            k0,
            epsilon,
            epsilon0,
            zeta0,
            field_strength,
            declared_virtuality,
            shell_mode,
            momentum_rule,
            require_zeta_transversality,
            validation_tolerance,
        };
        kinematics.validate()?;
        Ok(kinematics)
    }

    pub fn validate(&self) -> Result<(), KinematicsError> {
        if !self.validation_tolerance.is_finite() || self.validation_tolerance <= 0.0 {
            return Err(KinematicsError::InvalidValidationTolerance);
        }
        if !self.declared_virtuality.re.is_finite() || !self.declared_virtuality.im.is_finite() {
            return Err(KinematicsError::NonFiniteComponent("declared virtuality"));
        }
        if !vector_is_finite(&self.k)
            || !vector_is_finite(&self.k0)
            || !vector_is_finite(&self.epsilon)
            || !vector_is_finite(&self.zeta0)
            || !matrix_is_finite(&self.epsilon0)
            || !matrix_is_finite(&self.field_strength)
        {
            return Err(KinematicsError::NonFiniteComponent("kinematic tensor"));
        }
        if !matrix_is_antisymmetric(&self.field_strength, self.validation_tolerance) {
            return Err(KinematicsError::FieldStrengthNotAntisymmetric);
        }
        if !matrix_is_symmetric(&self.epsilon0, self.validation_tolerance) {
            return Err(KinematicsError::GravitonPolarizationNotSymmetric);
        }
        let momentum_defect = self.k + self.k0 - self.momentum_rule.expected_sum();
        if vector_norm(&momentum_defect) > self.validation_tolerance {
            return Err(KinematicsError::MomentumRuleViolation);
        }
        let virtuality_defect = bilinear_dot(&self.k, &self.k) - self.declared_virtuality;
        if virtuality_defect.norm() > self.validation_tolerance {
            return Err(KinematicsError::VirtualityMismatch);
        }
        if self.shell_mode.is_on_shell() {
            if self.declared_virtuality.norm() > self.validation_tolerance {
                return Err(KinematicsError::OnShellVirtualityNonzero);
            }
            if bilinear_dot(&self.epsilon, &self.k).norm() > self.validation_tolerance {
                return Err(KinematicsError::PhotonPolarizationNotTransverse);
            }
            if self.require_zeta_transversality
                && bilinear_dot(&self.zeta0, &self.k0).norm() > self.validation_tolerance
            {
                return Err(KinematicsError::DiffeomorphismVectorNotTransverse);
            }
        }
        Ok(())
    }

    pub fn graviton_variation(&self) -> ComplexLorentzMatrix {
        let mut variation = ComplexLorentzMatrix::zeros();
        for mu in 0..LORENTZ_DIMENSION {
            for nu in 0..LORENTZ_DIMENSION {
                variation[(mu, nu)] = self.k0[mu] * self.zeta0[nu] + self.zeta0[mu] * self.k0[nu];
            }
        }
        variation
    }

    pub fn photon_field_strength(&self) -> ComplexLorentzMatrix {
        let mut photon_field = ComplexLorentzMatrix::zeros();
        for mu in 0..LORENTZ_DIMENSION {
            for nu in 0..LORENTZ_DIMENSION {
                photon_field[(mu, nu)] =
                    self.k[mu] * self.epsilon[nu] - self.epsilon[mu] * self.k[nu];
            }
        }
        photon_field
    }
}

/// Rank-preserving electromagnetic gauge residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeWardResidual {
    pub diagram: Diagram,
    pub shell_mode: ShellMode,
    pub renormalization_state: RenormalizationState,
    pub contracted_components: ComplexLorentzMatrix,
    pub conditioning_scale: f64,
    pub absolute_norm: f64,
    pub normalized_norm: f64,
    pub tolerance: ResidualTolerance,
    pub passes: bool,
}

impl GaugeWardResidual {
    pub fn from_tensor(
        tensor: &ComplexRankThreeTensor,
        kinematics: &WardKinematics,
        diagram: Diagram,
        renormalization_state: RenormalizationState,
        conditioning_scale: f64,
        tolerance: ResidualTolerance,
    ) -> Result<Self, KinematicsError> {
        if !conditioning_scale.is_finite() || conditioning_scale <= 0.0 {
            return Err(KinematicsError::InvalidConditioningScale);
        }
        let contracted_components = tensor.contract_photon(&kinematics.k);
        let absolute_norm = matrix_frobenius_norm(&contracted_components);
        let normalized_norm = absolute_norm / conditioning_scale;
        Ok(Self {
            diagram,
            shell_mode: kinematics.shell_mode,
            renormalization_state,
            contracted_components,
            conditioning_scale,
            absolute_norm,
            normalized_norm,
            tolerance,
            passes: tolerance.accepts(absolute_norm, normalized_norm),
        })
    }
}

/// Rank-preserving gravitational residual with the complete lower-point split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GravitationalWardResidual {
    pub shell_mode: ShellMode,
    pub renormalization_state: RenormalizationState,
    pub lhs_components: ComplexFourVector,
    pub one_photon_rhs_components: ComplexFourVector,
    pub two_photon_rhs_components: ComplexFourVector,
    pub lower_point_rhs_components: ComplexFourVector,
    pub defect_components: ComplexFourVector,
    pub conditioning_scale: f64,
    pub absolute_defect: f64,
    pub normalized_defect: f64,
    pub tolerance: ResidualTolerance,
    pub passes: bool,
}

impl GravitationalWardResidual {
    pub fn from_components(
        lhs_components: ComplexFourVector,
        one_photon_rhs_components: ComplexFourVector,
        two_photon_rhs_components: ComplexFourVector,
        shell_mode: ShellMode,
        renormalization_state: RenormalizationState,
        conditioning_scale: f64,
        tolerance: ResidualTolerance,
    ) -> Result<Self, KinematicsError> {
        if !conditioning_scale.is_finite() || conditioning_scale <= 0.0 {
            return Err(KinematicsError::InvalidConditioningScale);
        }
        let lower_point_rhs_components = one_photon_rhs_components + two_photon_rhs_components;
        let defect_components = lhs_components - lower_point_rhs_components;
        let absolute_defect = vector_norm(&defect_components);
        let normalized_defect = absolute_defect / conditioning_scale;
        Ok(Self {
            shell_mode,
            renormalization_state,
            lhs_components,
            one_photon_rhs_components,
            two_photon_rhs_components,
            lower_point_rhs_components,
            defect_components,
            conditioning_scale,
            absolute_defect,
            normalized_defect,
            tolerance,
            passes: tolerance.accepts(absolute_defect, normalized_defect),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KinematicsError {
    InvalidValidationTolerance,
    InvalidTolerance,
    InvalidConditioningScale,
    NonFiniteComponent(&'static str),
    FieldStrengthNotAntisymmetric,
    GravitonPolarizationNotSymmetric,
    MomentumRuleViolation,
    VirtualityMismatch,
    OnShellVirtualityNonzero,
    PhotonPolarizationNotTransverse,
    DiffeomorphismVectorNotTransverse,
}

impl std::fmt::Display for KinematicsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidValidationTolerance => "invalid kinematic validation tolerance",
            Self::InvalidTolerance => "invalid residual tolerance",
            Self::InvalidConditioningScale => "invalid residual conditioning scale",
            Self::NonFiniteComponent(name) => name,
            Self::FieldStrengthNotAntisymmetric => "field strength is not antisymmetric",
            Self::GravitonPolarizationNotSymmetric => "graviton polarization is not symmetric",
            Self::MomentumRuleViolation => "momentum rule is violated",
            Self::VirtualityMismatch => "declared virtuality disagrees with k dot k",
            Self::OnShellVirtualityNonzero => "on-shell virtuality is nonzero",
            Self::PhotonPolarizationNotTransverse => {
                "on-shell photon polarization is not transverse"
            }
            Self::DiffeomorphismVectorNotTransverse => {
                "on-shell diffeomorphism vector is not transverse"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for KinematicsError {}

pub fn bilinear_dot(left: &ComplexFourVector, right: &ComplexFourVector) -> Complex64 {
    let mut result = Complex64::new(0.0, 0.0);
    for index in 0..LORENTZ_DIMENSION {
        result += left[index] * right[index];
    }
    result
}

pub fn vector_norm(vector: &ComplexFourVector) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

pub fn matrix_frobenius_norm(matrix: &ComplexLorentzMatrix) -> f64 {
    matrix.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn vector_is_finite(vector: &ComplexFourVector) -> bool {
    vector
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
}

fn matrix_is_finite(matrix: &ComplexLorentzMatrix) -> bool {
    matrix
        .iter()
        .all(|component| component.re.is_finite() && component.im.is_finite())
}

fn matrix_is_symmetric(matrix: &ComplexLorentzMatrix, tolerance: f64) -> bool {
    for mu in 0..LORENTZ_DIMENSION {
        for nu in 0..LORENTZ_DIMENSION {
            if (matrix[(mu, nu)] - matrix[(nu, mu)]).norm() > tolerance {
                return false;
            }
        }
    }
    true
}

fn matrix_is_antisymmetric(matrix: &ComplexLorentzMatrix, tolerance: f64) -> bool {
    for mu in 0..LORENTZ_DIMENSION {
        for nu in 0..LORENTZ_DIMENSION {
            if (matrix[(mu, nu)] + matrix[(nu, mu)]).norm() > tolerance {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_vector() -> ComplexFourVector {
        ComplexFourVector::zeros()
    }

    fn valid_kinematics() -> WardKinematics {
        let mut field_strength = ComplexLorentzMatrix::zeros();
        field_strength[(0, 1)] = Complex64::new(0.1, 0.0);
        field_strength[(1, 0)] = Complex64::new(-0.1, 0.0);
        WardKinematics::new(
            zero_vector(),
            zero_vector(),
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            zero_vector(),
            field_strength,
            Complex64::new(0.0, 0.0),
            ShellMode::OnShell,
            MomentumRule::ConstantBackgroundConversion,
            true,
            1e-12,
        )
        .expect("fixture must satisfy the declared kinematic invariants")
    }

    #[test]
    fn rank_three_contractions_preserve_shapes() {
        let tensor = ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
            Complex64::new((mu * 16 + nu * 4 + alpha) as f64, 0.0)
        });
        let photon = ComplexFourVector::from_element(Complex64::new(1.0, 0.0));
        let graviton = ComplexLorentzMatrix::from_element(Complex64::new(1.0, 0.0));
        let gauge = tensor.contract_photon(&photon);
        let gravitational = tensor.contract_graviton(&graviton);
        assert_eq!(gauge.shape(), (4, 4));
        assert_eq!(gravitational.shape(), (4, 1));
        assert_eq!(gauge[(0, 0)], Complex64::new(6.0, 0.0));
        assert_eq!(gravitational[0], Complex64::new(480.0, 0.0));
    }

    #[test]
    fn kinematics_rejects_non_antisymmetric_field() {
        let mut field_strength = ComplexLorentzMatrix::zeros();
        field_strength[(0, 1)] = Complex64::new(1.0, 0.0);
        let result = WardKinematics::new(
            zero_vector(),
            zero_vector(),
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            zero_vector(),
            field_strength,
            Complex64::new(0.0, 0.0),
            ShellMode::OnShell,
            MomentumRule::ConstantBackgroundConversion,
            false,
            1e-12,
        );
        assert_eq!(result, Err(KinematicsError::FieldStrengthNotAntisymmetric));
    }

    #[test]
    fn kinematics_rejects_nonsymmetric_graviton_polarization() {
        let mut epsilon0 = ComplexLorentzMatrix::zeros();
        epsilon0[(0, 1)] = Complex64::new(1.0, 0.0);
        let result = WardKinematics::new(
            zero_vector(),
            zero_vector(),
            zero_vector(),
            epsilon0,
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            Complex64::new(0.0, 0.0),
            ShellMode::OnShell,
            MomentumRule::ConstantBackgroundConversion,
            false,
            1e-12,
        );
        assert_eq!(
            result,
            Err(KinematicsError::GravitonPolarizationNotSymmetric)
        );
    }

    #[test]
    fn kinematics_rejects_virtuality_mismatch() {
        let mut k = zero_vector();
        k[0] = Complex64::new(1.0, 0.0);
        let result = WardKinematics::new(
            k,
            -k,
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            Complex64::new(0.0, 0.0),
            ShellMode::OffShell,
            MomentumRule::ConstantBackgroundConversion,
            false,
            1e-12,
        );
        assert_eq!(result, Err(KinematicsError::VirtualityMismatch));
    }

    #[test]
    fn kinematics_accepts_declared_off_shell_virtuality() {
        let mut k = zero_vector();
        k[0] = Complex64::new(1.0, 0.0);
        let result = WardKinematics::new(
            k,
            -k,
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            zero_vector(),
            ComplexLorentzMatrix::zeros(),
            Complex64::new(1.0, 0.0),
            ShellMode::OffShell,
            MomentumRule::ConstantBackgroundConversion,
            false,
            1e-12,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn residuals_retain_components_before_norms() {
        let kinematics = valid_kinematics();
        let tensor = ComplexRankThreeTensor::from_fn(|mu, nu, alpha| {
            if mu == 1 && nu == 2 && alpha == 0 {
                Complex64::new(2.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let tolerance = ResidualTolerance::new(1e-12, 1e-12).expect("valid tolerance");
        let residual = GaugeWardResidual::from_tensor(
            &tensor,
            &kinematics,
            Diagram::Irreducible,
            RenormalizationState::Unrenormalized,
            2.0,
            tolerance,
        )
        .expect("valid scale");
        assert_eq!(
            residual.contracted_components[(1, 2)],
            Complex64::new(0.0, 0.0)
        );
        assert_eq!(residual.absolute_norm, 0.0);
        assert!(residual.passes);
    }
}
