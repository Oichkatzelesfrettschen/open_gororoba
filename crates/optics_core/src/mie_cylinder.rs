//! Concentric-cylinder Lorentz-Mie scattering with explicit polarization.
//!
//! The source uses TM for a nonzero H_z field. At a radial material interface
//! the continuous state is H_z and (1/epsilon)*dH_z/drho. The alternate TE
//! path uses E_z and (1/mu)*dE_z/drho. The validated solver keeps the complex
//! incoming and outgoing channel amplitudes and computes observables from the
//! source definitions.
//!
//! See Ruan and Fan, "Temporal coupled-mode theory for Fano resonance in light
//! scattering by a single obstacle", arXiv:0909.3323v2, Figures 3-5.

use crate::{
    bessel::{bessel_j, bessel_j_prime, bessel_y, bessel_y_prime},
    fano_tcmt::{
        ChannelCrossSections, CrossSections, FanoChannelError, FanoDrudeParams, try_drude_epsilon,
    },
};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// Longitudinal cylindrical polarization and its interface state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CylindricalPolarization {
    /// TM source convention with nonzero H_z.
    HzTm,
    /// TE convention with nonzero E_z.
    EzTe,
}

/// Material role retained in the source geometry ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialRole {
    /// Drude material in the Ruan-Fan MDM geometry.
    Metal,
    /// Nonmetal source or test material.
    Dielectric,
    /// Compatibility role for a generic finite layer.
    Generic,
}

/// A finite radial layer with explicit electromagnetic material data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylinderLayer {
    /// Outer radius of this layer.
    pub outer_radius: f64,
    /// Relative permittivity.
    pub epsilon: Complex64,
    /// Relative permeability.
    pub mu: Complex64,
    /// Source or test material role.
    pub material: MaterialRole,
}

impl CylinderLayer {
    /// Construct a nonmagnetic finite layer with an explicit material role.
    pub fn nonmagnetic(outer_radius: f64, epsilon: Complex64, material: MaterialRole) -> Self {
        Self {
            outer_radius,
            epsilon,
            mu: Complex64::new(1.0, 0.0),
            material,
        }
    }
}

/// Concentric geometry and its exterior boundary data.
#[derive(Debug, Clone)]
pub struct ConcentricCylinder {
    /// Layers from innermost to outermost shell.
    pub layers: Vec<CylinderLayer>,
    /// Exterior relative permittivity.
    pub eps_ext: Complex64,
    /// Exterior relative permeability.
    pub mu_ext: Complex64,
    /// Declared longitudinal polarization.
    pub polarization: CylindricalPolarization,
    /// Optional frequency-dependent Drude model for layers marked Metal.
    pub metal_drude: Option<FanoDrudeParams>,
}

/// Defect between the two continuous interface state components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceContinuityResidual {
    /// Difference in the longitudinal field component.
    pub field_defect: Complex64,
    /// Difference in the weighted radial flux component.
    pub flux_defect: Complex64,
    /// Largest component magnitude.
    pub max_component: f64,
}

/// Complex observables and interface diagnostics for one angular channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelResult {
    /// Angular momentum index l.
    pub l: i32,
    /// Scattering coefficient S_l.
    pub s_l: Complex64,
    /// Reflection coefficient R_l = 1 + 2*S_l.
    pub r_l: Complex64,
    /// Independently defined normalized channel observables.
    pub cross_sections: ChannelCrossSections,
    /// Absorption computed from the R_l flux defect.
    pub absorption_from_flux: f64,
    /// Extinction - scattering - S-based absorption.
    pub balance_defect: f64,
    /// Maximum state continuity residual over interfaces.
    pub interface_residual: InterfaceContinuityResidual,
    /// Reciprocal determinant conditioning indicator.
    pub conditioning_indicator: f64,
}

/// Aggregate observable residuals for a finite channel sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MieObservableResiduals {
    /// Extinction - scattering - S-based absorption.
    pub balance_defect: f64,
    /// Maximum S, R, and closed-form absorption disagreement.
    pub absorption_representation_defect: f64,
    /// Maximum R-flux and closed-form absorption disagreement.
    pub flux_representation_defect: f64,
}

/// Full Mie result for a declared finite channel range.
#[derive(Debug, Clone)]
pub struct MieResult {
    /// Per-channel complex results and residuals.
    pub channels: Vec<ChannelResult>,
    /// Total cross-sections normalized by 2*lambda/pi.
    pub cross_sections: CrossSections,
    /// Aggregate residuals retained before scalar verdicts.
    pub observable_residuals: MieObservableResiduals,
    /// Angular frequency used.
    pub omega: f64,
}

/// Error returned by the validated cylindrical solver.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MieError {
    #[error("the cylinder has no finite layers")]
    EmptyLayers,
    #[error("{field} must be finite")]
    NonFinite { field: &'static str },
    #[error("omega must be positive")]
    NonPositiveFrequency,
    #[error("l_max must be nonnegative")]
    NegativeLMax,
    #[error("layer radii must be positive and strictly increasing")]
    InvalidRadii,
    #[error("epsilon and mu must be nonzero")]
    ZeroMaterialParameter,
    #[error("the interface determinant is singular for l={l} at rho={rho}")]
    SingularInterface { l: i32, rho: f64 },
    #[error("the exterior incoming amplitude is singular for l={l}")]
    SingularExteriorAmplitude { l: i32 },
    #[error("the cylindrical calculation produced a non-finite value")]
    NonFiniteCalculation,
    #[error("layer index {index} is out of range")]
    LayerIndexOutOfRange { index: usize },
    #[error("layer {index} is not marked as metal")]
    NotMetalLayer { index: usize },
    #[error("Drude validation failed: {0}")]
    Drude(#[from] FanoChannelError),
}

/// 2x2 complex matrix stored in row-major order.
type Mat2 = [[Complex64; 2]; 2];

#[derive(Debug, Clone, Copy)]
struct MaterialParameters {
    epsilon: Complex64,
    mu: Complex64,
}

#[derive(Debug, Clone, Copy)]
struct RadialState {
    coefficients: [Complex64; 2],
    k: Complex64,
    material: MaterialParameters,
}

fn finite_complex(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat2_vec(matrix: &Mat2, vector: [Complex64; 2]) -> [Complex64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn material_weight(
    polarization: CylindricalPolarization,
    epsilon: Complex64,
    mu: Complex64,
) -> Result<Complex64, MieError> {
    let weight = match polarization {
        CylindricalPolarization::HzTm => Complex64::new(1.0, 0.0) / epsilon,
        CylindricalPolarization::EzTe => Complex64::new(1.0, 0.0) / mu,
    };
    if !finite_complex(weight) {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok(weight)
}

fn state_matrix(
    l: i32,
    k: Complex64,
    material: MaterialParameters,
    polarization: CylindricalPolarization,
    rho: f64,
) -> Result<Mat2, MieError> {
    let argument = k * rho;
    let weight = material_weight(polarization, material.epsilon, material.mu)?;
    let matrix = [
        [bessel_j(l, argument), bessel_y(l, argument)],
        [
            weight * k * bessel_j_prime(l, argument),
            weight * k * bessel_y_prime(l, argument),
        ],
    ];
    if matrix.iter().flatten().any(|value| !finite_complex(*value)) {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok(matrix)
}

fn interface_matrix(
    l: i32,
    k_in: Complex64,
    material_in: MaterialParameters,
    k_out: Complex64,
    material_out: MaterialParameters,
    polarization: CylindricalPolarization,
    rho: f64,
) -> Result<(Mat2, f64), MieError> {
    let inner = state_matrix(l, k_in, material_in, polarization, rho)?;
    let outer = state_matrix(l, k_out, material_out, polarization, rho)?;
    let determinant = outer[0][0] * outer[1][1] - outer[0][1] * outer[1][0];
    if determinant.norm_sqr() == 0.0 {
        return Err(MieError::SingularInterface { l, rho });
    }
    let inverse = [
        [outer[1][1] / determinant, -outer[0][1] / determinant],
        [-outer[1][0] / determinant, outer[0][0] / determinant],
    ];
    let transfer = mat2_mul(&inverse, &inner);
    if transfer
        .iter()
        .flatten()
        .any(|value| !finite_complex(*value))
    {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok((transfer, 1.0 / determinant.norm()))
}

fn state_residual(
    l: i32,
    rho: f64,
    inner_state: RadialState,
    outer_state: RadialState,
    polarization: CylindricalPolarization,
) -> Result<InterfaceContinuityResidual, MieError> {
    let inner_matrix = state_matrix(l, inner_state.k, inner_state.material, polarization, rho)?;
    let outer_matrix = state_matrix(l, outer_state.k, outer_state.material, polarization, rho)?;
    let inner_values = mat2_vec(&inner_matrix, inner_state.coefficients);
    let outer_values = mat2_vec(&outer_matrix, outer_state.coefficients);
    let field_defect = outer_values[0] - inner_values[0];
    let flux_defect = outer_values[1] - inner_values[1];
    let max_component = field_defect.norm().max(flux_defect.norm());
    if !max_component.is_finite() {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok(InterfaceContinuityResidual {
        field_defect,
        flux_defect,
        max_component,
    })
}

fn passive_sqrt(epsilon: Complex64) -> Result<Complex64, MieError> {
    if !finite_complex(epsilon) {
        return Err(MieError::NonFinite { field: "epsilon" });
    }
    let mut root = epsilon.sqrt();
    if root.im < 0.0 || (root.im == 0.0 && root.re < 0.0) {
        root = -root;
    }
    if !finite_complex(root) {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok(root)
}

/// Select the passive wavenumber branch for exp(-i*omega*t).
fn wavenumber(omega: f64, epsilon: Complex64) -> Result<Complex64, MieError> {
    let root = passive_sqrt(epsilon)?;
    let result = Complex64::new(omega, 0.0) * root;
    if !finite_complex(result) {
        return Err(MieError::NonFiniteCalculation);
    }
    Ok(result)
}

impl ConcentricCylinder {
    /// Validate geometry and dimensional material inputs at one frequency.
    pub fn validate(&self, omega: f64) -> Result<(), MieError> {
        if self.layers.is_empty() {
            return Err(MieError::EmptyLayers);
        }
        if !omega.is_finite() {
            return Err(MieError::NonFinite { field: "omega" });
        }
        if omega <= 0.0 {
            return Err(MieError::NonPositiveFrequency);
        }
        if !finite_complex(self.eps_ext) || !finite_complex(self.mu_ext) {
            return Err(MieError::NonFinite {
                field: "exterior material",
            });
        }
        if self.eps_ext.norm_sqr() == 0.0 || self.mu_ext.norm_sqr() == 0.0 {
            return Err(MieError::ZeroMaterialParameter);
        }
        let mut previous_radius = 0.0;
        for layer in &self.layers {
            if !layer.outer_radius.is_finite()
                || layer.outer_radius <= 0.0
                || layer.outer_radius <= previous_radius
                || !finite_complex(layer.epsilon)
                || !finite_complex(layer.mu)
            {
                return Err(MieError::InvalidRadii);
            }
            if layer.epsilon.norm_sqr() == 0.0 || layer.mu.norm_sqr() == 0.0 {
                return Err(MieError::ZeroMaterialParameter);
            }
            previous_radius = layer.outer_radius;
        }
        if let Some(drude) = self.metal_drude {
            drude.validate()?;
            for layer in &self.layers {
                if layer.material == MaterialRole::Metal {
                    let epsilon = try_drude_epsilon(&drude, omega)?;
                    if epsilon.norm_sqr() == 0.0 {
                        return Err(MieError::ZeroMaterialParameter);
                    }
                }
            }
        }
        Ok(())
    }

    fn material_at(&self, layer: CylinderLayer, omega: f64) -> Result<CylinderLayer, MieError> {
        if layer.material == MaterialRole::Metal
            && let Some(drude) = self.metal_drude
        {
            return Ok(CylinderLayer {
                epsilon: try_drude_epsilon(&drude, omega)?,
                ..layer
            });
        }
        Ok(layer)
    }
}

/// Compute one complex Mie channel and retain interface diagnostics.
pub fn try_scattering_channel(
    geom: &ConcentricCylinder,
    l: i32,
    omega: f64,
) -> Result<ChannelResult, MieError> {
    geom.validate(omega)?;
    let core = geom.material_at(geom.layers[0], omega)?;
    let k_core = wavenumber(omega, core.epsilon)?;
    let k_ext = wavenumber(omega, geom.eps_ext)?;
    let mut coefficients = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let mut maximum_interface_residual = InterfaceContinuityResidual {
        field_defect: Complex64::new(0.0, 0.0),
        flux_defect: Complex64::new(0.0, 0.0),
        max_component: 0.0,
    };
    let mut conditioning_indicator: f64 = 0.0;

    for index in 0..geom.layers.len() {
        let inner = if index == 0 {
            core
        } else {
            geom.material_at(geom.layers[index], omega)?
        };
        let outer = if index + 1 == geom.layers.len() {
            CylinderLayer {
                outer_radius: geom.layers[index].outer_radius,
                epsilon: geom.eps_ext,
                mu: geom.mu_ext,
                material: MaterialRole::Generic,
            }
        } else {
            geom.material_at(geom.layers[index + 1], omega)?
        };
        let k_in = if index == 0 {
            k_core
        } else {
            wavenumber(omega, inner.epsilon)?
        };
        let k_out = if index + 1 == geom.layers.len() {
            k_ext
        } else {
            wavenumber(omega, outer.epsilon)?
        };
        let before = coefficients;
        let (transfer, determinant_indicator) = interface_matrix(
            l,
            k_in,
            MaterialParameters {
                epsilon: inner.epsilon,
                mu: inner.mu,
            },
            k_out,
            MaterialParameters {
                epsilon: outer.epsilon,
                mu: outer.mu,
            },
            geom.polarization,
            geom.layers[index].outer_radius,
        )?;
        coefficients = mat2_vec(&transfer, before);
        let residual = state_residual(
            l,
            geom.layers[index].outer_radius,
            RadialState {
                coefficients: before,
                k: k_in,
                material: MaterialParameters {
                    epsilon: inner.epsilon,
                    mu: inner.mu,
                },
            },
            RadialState {
                coefficients,
                k: k_out,
                material: MaterialParameters {
                    epsilon: outer.epsilon,
                    mu: outer.mu,
                },
            },
            geom.polarization,
        )?;
        if residual.max_component > maximum_interface_residual.max_component {
            maximum_interface_residual = residual;
        }
        conditioning_indicator = conditioning_indicator.max(determinant_indicator);
    }

    let imaginary_unit = Complex64::i();
    let outgoing = coefficients[0] - imaginary_unit * coefficients[1];
    let incoming = coefficients[0] + imaginary_unit * coefficients[1];
    if incoming.norm_sqr() == 0.0 {
        return Err(MieError::SingularExteriorAmplitude { l });
    }
    let reflection = outgoing / incoming;
    let scattering = (reflection - 1.0) / 2.0;
    if !finite_complex(reflection) || !finite_complex(scattering) {
        return Err(MieError::NonFiniteCalculation);
    }
    let cross_sections = ChannelCrossSections {
        scattering: scattering.norm_sqr(),
        absorption: -(scattering.re + scattering.norm_sqr()),
        extinction: -scattering.re,
    };
    let absorption_from_flux = (1.0 - reflection.norm_sqr()) / 4.0;
    let balance_defect =
        cross_sections.extinction - cross_sections.scattering - cross_sections.absorption;
    Ok(ChannelResult {
        l,
        s_l: scattering,
        r_l: reflection,
        cross_sections,
        absorption_from_flux,
        balance_defect,
        interface_residual: maximum_interface_residual,
        conditioning_indicator,
    })
}

/// Compute one complex Mie channel with compatibility return type.
pub fn scattering_coefficient_l(geom: &ConcentricCylinder, l: i32, omega: f64) -> Complex64 {
    try_scattering_channel(geom, l, omega)
        .unwrap_or_else(|error| panic!("invalid cylindrical channel input: {error}"))
        .s_l
}

/// Compute all channels in the declared range with explicit validation.
pub fn try_mie_scattering(
    geom: &ConcentricCylinder,
    omega: f64,
    l_max: i32,
) -> Result<MieResult, MieError> {
    if l_max < 0 {
        return Err(MieError::NegativeLMax);
    }
    let mut channels = Vec::new();
    let mut scattering = 0.0;
    let mut absorption = 0.0;
    let mut extinction = 0.0;
    let mut absorption_from_flux = 0.0;
    let mut absorption_representation_defect: f64 = 0.0;
    let mut flux_representation_defect: f64 = 0.0;
    for l in -l_max..=l_max {
        let channel = try_scattering_channel(geom, l, omega)?;
        scattering += channel.cross_sections.scattering;
        absorption += channel.cross_sections.absorption;
        extinction += channel.cross_sections.extinction;
        absorption_from_flux += channel.absorption_from_flux;
        absorption_representation_defect = absorption_representation_defect
            .max((channel.cross_sections.absorption - channel.absorption_from_flux).abs());
        flux_representation_defect = flux_representation_defect
            .max((channel.absorption_from_flux - channel.cross_sections.absorption).abs());
        channels.push(channel);
    }
    let cross_sections = CrossSections {
        c_sct: scattering,
        c_abs: absorption,
        c_ext: extinction,
    };
    Ok(MieResult {
        channels,
        cross_sections,
        observable_residuals: MieObservableResiduals {
            balance_defect: extinction - scattering - absorption,
            absorption_representation_defect,
            flux_representation_defect: (absorption_from_flux - absorption)
                .abs()
                .max(flux_representation_defect),
        },
        omega,
    })
}

/// Compute all channels with compatibility return type.
pub fn mie_scattering(geom: &ConcentricCylinder, omega: f64, l_max: i32) -> MieResult {
    try_mie_scattering(geom, omega, l_max)
        .unwrap_or_else(|error| panic!("invalid cylindrical geometry: {error}"))
}

/// Compute a validated frequency sweep.
pub fn try_mie_sweep(
    geom: &ConcentricCylinder,
    omegas: &[f64],
    l_max: i32,
) -> Result<Vec<MieResult>, MieError> {
    omegas
        .iter()
        .map(|&omega| try_mie_scattering(geom, omega, l_max))
        .collect()
}

/// Compute a compatibility frequency sweep.
pub fn mie_sweep(geom: &ConcentricCylinder, omegas: &[f64], l_max: i32) -> Vec<MieResult> {
    try_mie_sweep(geom, omegas, l_max)
        .unwrap_or_else(|error| panic!("invalid cylindrical geometry: {error}"))
}

fn source_mdm_geometry(
    drude: &FanoDrudeParams,
    inner_radius_over_lambda_p: f64,
    dielectric_radius_over_lambda_p: f64,
    outer_radius_over_lambda_p: f64,
) -> ConcentricCylinder {
    let lambda_p = 2.0 * PI / drude.omega_p;
    let dielectric = Complex64::new(12.96, 0.0);
    let metal_placeholder = Complex64::new(1.0, 0.0);
    ConcentricCylinder {
        layers: vec![
            CylinderLayer::nonmagnetic(
                inner_radius_over_lambda_p * lambda_p,
                metal_placeholder,
                MaterialRole::Metal,
            ),
            CylinderLayer::nonmagnetic(
                dielectric_radius_over_lambda_p * lambda_p,
                dielectric,
                MaterialRole::Dielectric,
            ),
            CylinderLayer::nonmagnetic(
                outer_radius_over_lambda_p * lambda_p,
                metal_placeholder,
                MaterialRole::Metal,
            ),
        ],
        eps_ext: Complex64::new(1.0, 0.0),
        mu_ext: Complex64::new(1.0, 0.0),
        polarization: CylindricalPolarization::HzTm,
        metal_drude: Some(*drude),
    }
}

/// Build the source Figure 4 metal-dielectric-metal geometry.
pub fn ruan_fan_mdm_fig4(drude: &FanoDrudeParams) -> ConcentricCylinder {
    source_mdm_geometry(drude, 0.285, 1.0, 1.5)
}

/// Build the source Figure 5 metal-dielectric-metal geometry.
pub fn ruan_fan_mdm_fig5(drude: &FanoDrudeParams) -> ConcentricCylinder {
    source_mdm_geometry(drude, 0.36, 0.73, 1.0)
}

/// Update one explicitly marked metal layer with a Drude value.
pub fn try_update_metal_epsilon(
    geom: &mut ConcentricCylinder,
    metal_layer_idx: usize,
    drude: &FanoDrudeParams,
    omega: f64,
) -> Result<(), MieError> {
    if metal_layer_idx >= geom.layers.len() {
        return Err(MieError::LayerIndexOutOfRange {
            index: metal_layer_idx,
        });
    }
    if geom.layers[metal_layer_idx].material != MaterialRole::Metal {
        return Err(MieError::NotMetalLayer {
            index: metal_layer_idx,
        });
    }
    geom.layers[metal_layer_idx].epsilon = try_drude_epsilon(drude, omega)?;
    Ok(())
}

/// Compatibility wrapper for updating one metal layer.
pub fn update_metal_epsilon(
    geom: &mut ConcentricCylinder,
    metal_layer_idx: usize,
    drude: &FanoDrudeParams,
    omega: f64,
) {
    try_update_metal_epsilon(geom, metal_layer_idx, drude, omega)
        .unwrap_or_else(|error| panic!("invalid metal update: {error}"));
}

/// Compute a source MDM sweep using all role-tagged metal layers.
pub fn mie_mdm_sweep(
    base_geom: &ConcentricCylinder,
    metal_layer_idx: usize,
    drude: &FanoDrudeParams,
    omegas: &[f64],
    l_max: i32,
) -> Vec<MieResult> {
    let has_role_tagged_metal = base_geom
        .layers
        .iter()
        .any(|layer| layer.material == MaterialRole::Metal);
    omegas
        .iter()
        .map(|&omega| {
            let mut geometry = base_geom.clone();
            if has_role_tagged_metal {
                geometry.metal_drude = Some(*drude);
            } else {
                update_metal_epsilon(&mut geometry, metal_layer_idx, drude, omega);
            }
            mie_scattering(&geometry, omega, l_max)
        })
        .collect()
}

/// Heuristic characterization extractor retained outside validated paths.
///
/// The source uses complex roots and a uniform metallic-cylinder background.
/// This function deliberately does not claim that method. It returns None when
/// the sweep cannot supply a positive measured half width; it has no range
/// based fallback.
pub fn extract_fano_params(
    omegas: &[f64],
    results: &[MieResult],
    l: i32,
) -> Option<crate::fano_tcmt::FanoChannel> {
    if omegas.len() != results.len() || omegas.is_empty() {
        return None;
    }
    let mut max_s2 = 0.0_f64;
    let mut peak_idx = None;
    for (index, result) in results.iter().enumerate() {
        if let Some(channel) = result.channels.iter().find(|channel| channel.l == l) {
            let scattering = channel.s_l.norm_sqr();
            if scattering > max_s2 {
                max_s2 = scattering;
                peak_idx = Some(index);
            }
        }
    }
    let peak_idx = peak_idx?;
    let omega_0 = omegas[peak_idx];
    let half_max = max_s2 / 2.0;
    let mut gamma = None;
    for index in (peak_idx + 1)..omegas.len() {
        let current = results[index]
            .channels
            .iter()
            .find(|channel| channel.l == l)?
            .s_l
            .norm_sqr();
        if current < half_max {
            let previous = results[index - 1]
                .channels
                .iter()
                .find(|channel| channel.l == l)?
                .s_l
                .norm_sqr();
            let denominator = current - previous;
            if denominator == 0.0 {
                return None;
            }
            let fraction = (half_max - previous) / denominator;
            let half_frequency = omegas[index - 1] + fraction * (omegas[index] - omegas[index - 1]);
            let width = half_frequency - omega_0;
            if width > 0.0 && width.is_finite() {
                gamma = Some(width);
            }
            break;
        }
    }
    let gamma = gamma?;
    let far_idx = if peak_idx > omegas.len() / 2 {
        0
    } else {
        omegas.len() - 1
    };
    let phase = results[far_idx]
        .channels
        .iter()
        .find(|channel| channel.l == l)?
        .r_l
        .arg();
    let peak_channel = results[peak_idx]
        .channels
        .iter()
        .find(|channel| channel.l == l)?;
    let absorption = -(peak_channel.s_l.re + peak_channel.s_l.norm_sqr());
    let ratio = if absorption > 0.0 {
        let discriminant = (1.0 - 2.0 * absorption).max(0.0);
        (1.0 - 2.0 * absorption + discriminant.sqrt()) / (2.0 * absorption)
    } else {
        0.0
    };
    Some(crate::fano_tcmt::FanoChannel {
        omega_0,
        gamma,
        gamma_0: ratio * gamma,
        phi: phase,
        l,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bessel::{
        bessel_j, bessel_j_prime, hankel_1, hankel_1_prime, hankel_2, hankel_2_prime,
    };

    fn geometry(layers: Vec<CylinderLayer>) -> ConcentricCylinder {
        ConcentricCylinder {
            layers,
            eps_ext: Complex64::new(1.0, 0.0),
            mu_ext: Complex64::new(1.0, 0.0),
            polarization: CylindricalPolarization::HzTm,
            metal_drude: None,
        }
    }

    #[test]
    fn homogeneous_no_scattering() {
        let geom = geometry(vec![CylinderLayer::nonmagnetic(
            1.0,
            Complex64::new(1.0, 0.0),
            MaterialRole::Dielectric,
        )]);
        for l in 0..=3 {
            let result = try_scattering_channel(&geom, l, 2.0).expect("valid homogeneous layer");
            assert!(result.s_l.norm() < 1e-10);
            assert!(result.interface_residual.max_component < 1e-10);
        }
    }

    #[test]
    fn single_hz_dielectric_matches_direct_formula() {
        let epsilon = Complex64::new(4.0, 0.0);
        let radius = 1.0;
        let omega = 2.0;
        let geom = geometry(vec![CylinderLayer::nonmagnetic(
            radius,
            epsilon,
            MaterialRole::Dielectric,
        )]);
        let k0 = Complex64::new(omega, 0.0);
        let k = k0 * epsilon.sqrt();
        let x0 = k0 * radius;
        let x = k * radius;
        let q_inside = k / epsilon;
        let q_outside = k0;
        for l in 0..=2 {
            let result = try_scattering_channel(&geom, l, omega).expect("valid dielectric");
            let numerator = q_outside * bessel_j(l, x) * hankel_2_prime(l, x0)
                - q_inside * bessel_j_prime(l, x) * hankel_2(l, x0);
            let denominator = q_inside * bessel_j_prime(l, x) * hankel_1(l, x0)
                - q_outside * bessel_j(l, x) * hankel_1_prime(l, x0);
            let direct_reflection = numerator / denominator;
            let direct_scattering = (direct_reflection - 1.0) / 2.0;
            assert!((result.s_l - direct_scattering).norm() < 1e-8);
        }
    }

    #[test]
    fn lossless_channel_is_flux_normalized() {
        let geom = geometry(vec![CylinderLayer::nonmagnetic(
            0.5,
            Complex64::new(9.0, 0.0),
            MaterialRole::Dielectric,
        )]);
        let result = try_mie_scattering(&geom, 3.0, 3).expect("valid lossless geometry");
        for channel in &result.channels {
            assert!((channel.r_l.norm() - 1.0).abs() < 1e-8);
            assert!(channel.absorption_from_flux.abs() < 1e-8);
            assert!(channel.interface_residual.max_component < 1e-8);
        }
    }

    #[test]
    fn passive_channel_has_nonnegative_absorption() {
        let geom = geometry(vec![CylinderLayer::nonmagnetic(
            0.8,
            Complex64::new(4.0, 0.5),
            MaterialRole::Dielectric,
        )]);
        let result = try_mie_scattering(&geom, 2.5, 4).expect("valid passive geometry");
        for channel in &result.channels {
            assert!(channel.r_l.norm() <= 1.0 + 1e-8);
            assert!(channel.cross_sections.absorption >= -1e-10);
            assert!(channel.absorption_from_flux >= -1e-10);
        }
        assert!(result.observable_residuals.balance_defect.abs() < 1e-12);
    }

    #[test]
    fn channel_symmetry() {
        let geom = geometry(vec![
            CylinderLayer::nonmagnetic(0.5, Complex64::new(4.0, 0.1), MaterialRole::Dielectric),
            CylinderLayer::nonmagnetic(1.0, Complex64::new(2.0, 0.0), MaterialRole::Dielectric),
        ]);
        for l in 1..=3 {
            let positive = scattering_coefficient_l(&geom, l, 2.0);
            let negative = scattering_coefficient_l(&geom, -l, 2.0);
            assert!((positive - negative).norm() < 1e-10);
        }
    }

    #[test]
    fn source_geometry_has_metal_dielectric_metal_order() {
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.001,
        };
        let geometry = ruan_fan_mdm_fig4(&drude);
        assert_eq!(geometry.polarization, CylindricalPolarization::HzTm);
        assert_eq!(geometry.layers[0].material, MaterialRole::Metal);
        assert_eq!(geometry.layers[1].material, MaterialRole::Dielectric);
        assert_eq!(geometry.layers[2].material, MaterialRole::Metal);
        assert_eq!(geometry.layers[1].epsilon, Complex64::new(12.96, 0.0));
    }

    #[test]
    fn material_order_mutation_changes_the_channel_response() {
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.001,
        };
        let source_geometry = ruan_fan_mdm_fig4(&drude);
        let source =
            try_scattering_channel(&source_geometry, 0, 0.1552).expect("valid source MDM geometry");
        let mut swapped_geometry = source_geometry.clone();
        swapped_geometry.layers[0].material = MaterialRole::Dielectric;
        swapped_geometry.layers[0].epsilon = Complex64::new(12.96, 0.0);
        swapped_geometry.layers[1].material = MaterialRole::Metal;
        swapped_geometry.layers[1].epsilon = Complex64::new(1.0, 0.0);
        let swapped =
            try_scattering_channel(&swapped_geometry, 0, 0.1552).expect("valid mutated geometry");
        assert!((source.s_l - swapped.s_l).norm() > 1e-6);
    }

    #[test]
    fn source_mdm_resonance_has_nonzero_channel_response() {
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.0,
        };
        let geometry = ruan_fan_mdm_fig4(&drude);
        let omegas: Vec<f64> = (0..25)
            .map(|index| 0.14 + 0.03 * index as f64 / 24.0)
            .collect();
        let results = mie_mdm_sweep(&geometry, 1, &drude, &omegas, 0);
        let maximum = results
            .iter()
            .flat_map(|result| result.channels.iter())
            .map(|channel| channel.s_l.norm_sqr())
            .fold(0.0, f64::max);
        assert!(maximum > 1e-6);
    }

    #[test]
    fn source_mdm_lossless_channels_are_flux_normalized() {
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.0,
        };
        let geometry = ruan_fan_mdm_fig4(&drude);
        for omega in [0.145, 0.1552, 0.17] {
            let result = try_mie_scattering(&geometry, omega, 2).expect("valid lossless MDM");
            for channel in result.channels {
                assert!((channel.r_l.norm() - 1.0).abs() < 1e-7);
                assert!(channel.cross_sections.absorption.abs() < 1e-7);
            }
        }
    }

    #[test]
    fn source_mdm_passive_channels_are_contractive() {
        let drude = FanoDrudeParams {
            omega_p: 1.0,
            gamma_d: 0.001,
        };
        let geometry = ruan_fan_mdm_fig5(&drude);
        for omega in [0.22, 0.226, 0.233] {
            let result = try_mie_scattering(&geometry, omega, 2).expect("valid passive MDM");
            for channel in result.channels {
                assert!(channel.r_l.norm() <= 1.0 + 1e-7);
                assert!(channel.cross_sections.absorption >= -1e-8);
                assert!(channel.absorption_from_flux >= -1e-8);
            }
        }
    }

    #[test]
    fn interface_polarization_changes_weighted_state() {
        let layers = vec![CylinderLayer {
            outer_radius: 0.8,
            epsilon: Complex64::new(4.0, 0.0),
            mu: Complex64::new(2.0, 0.0),
            material: MaterialRole::Dielectric,
        }];
        let hz = ConcentricCylinder {
            layers: layers.clone(),
            eps_ext: Complex64::new(1.0, 0.0),
            mu_ext: Complex64::new(1.0, 0.0),
            polarization: CylindricalPolarization::HzTm,
            metal_drude: None,
        };
        let ez = ConcentricCylinder {
            polarization: CylindricalPolarization::EzTe,
            ..hz.clone()
        };
        let hz_result = try_scattering_channel(&hz, 0, 2.5).expect("valid Hz geometry");
        let ez_result = try_scattering_channel(&ez, 0, 2.5).expect("valid Ez geometry");
        assert!((hz_result.s_l - ez_result.s_l).norm() > 1e-8);
        assert!(hz_result.interface_residual.max_component < 1e-8);
        assert!(ez_result.interface_residual.max_component < 1e-8);
    }

    #[test]
    fn passive_branch_has_nonnegative_imaginary_wavenumber() {
        let root = passive_sqrt(Complex64::new(-3.0, 0.2)).expect("valid passive material");
        assert!(root.im >= 0.0);
        assert!(root.re >= 0.0);
    }

    #[test]
    fn invalid_geometry_is_rejected_without_clamping() {
        let invalid = geometry(vec![CylinderLayer::nonmagnetic(
            0.0,
            Complex64::new(1.0, 0.0),
            MaterialRole::Dielectric,
        )]);
        assert!(matches!(
            try_mie_scattering(&invalid, 1.0, 0),
            Err(MieError::InvalidRadii)
        ));
    }

    #[test]
    fn old_extinction_bookkeeping_is_not_the_observable() {
        let geom = geometry(vec![CylinderLayer::nonmagnetic(
            0.8,
            Complex64::new(4.0, 0.5),
            MaterialRole::Dielectric,
        )]);
        let result = try_mie_scattering(&geom, 2.5, 2).expect("valid passive geometry");
        let old_absorption: f64 = result.channels.iter().map(|channel| -channel.s_l.re).sum();
        assert!((old_absorption - result.cross_sections.c_abs).abs() > 1e-8);
        assert!(
            (result.cross_sections.c_ext
                - result.cross_sections.c_sct
                - result.cross_sections.c_abs)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn direct_hankel_basis_is_referenced_in_tests() {
        let argument = Complex64::new(2.0, 0.3);
        let value = hankel_1(0, argument) + hankel_2(0, argument);
        assert!((value - 2.0 * bessel_j(0, argument)).norm() < 1e-10);
    }
}
