//! Planar Lifshitz interactions for stratified local optical media.
//!
//! A [`Multilayer`] is ordered from the vacuum gap into the body. Each
//! finite-thickness [`Layer`] borrows a [`DrudeLorentzParams`] model, and the
//! stack terminates in a material or ideal-conductor half-space. The
//! zero-temperature integrals cover the complete dimensionless
//! `q in [0, infinity)` and `mu in [0, 1]` domain. The finite-temperature
//! pressure implements the local-Drude Matsubara prescription.

use std::{error::Error, f64::consts::PI, fmt, num::NonZeroUsize};

use gauss_quad::GaussLegendre;

use crate::{C, DrudeLorentzParams, E_CHARGE, EV_TO_RADS, HBAR_EV_S, K_B_EV, ScatteringModel};

const HBAR_J_S: f64 = HBAR_EV_S * E_CHARGE;
const K_B_J_K: f64 = K_B_EV * E_CHARGE;

/// A hypothetical ultraviolet oscillator that completes a finite-band
/// high-frequency background.
///
/// [`DrudeLorentzParams::eps_inf`] is replaced, for this calculation only, by
/// a unit vacuum background plus an oscillator of strength `eps_inf - 1`.
/// The replacement preserves the model's static background and makes the
/// completed response approach one at infinite imaginary frequency. The
/// borrowed source model remains unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighFrequencyCompletion {
    /// Hypothetical oscillator resonance energy in eV.
    pub resonance_ev: f64,
    /// Hypothetical oscillator damping energy in eV.
    pub damping_ev: f64,
}

impl HighFrequencyCompletion {
    /// Construct a high-frequency background completion.
    pub const fn new(resonance_ev: f64, damping_ev: f64) -> Self {
        Self {
            resonance_ev,
            damping_ev,
        }
    }
}

/// An immutable optical response and its calculation-specific completion.
#[derive(Debug, Clone, Copy)]
pub struct LifshitzModel<'a> {
    /// Source Drude-Lorentz model.
    pub params: &'a DrudeLorentzParams,
    /// Optional hypothetical completion of the constant high-frequency term.
    pub high_frequency_completion: Option<HighFrequencyCompletion>,
}

impl<'a> LifshitzModel<'a> {
    /// Borrow a Drude-Lorentz model without a high-frequency completion.
    pub const fn new(params: &'a DrudeLorentzParams) -> Self {
        Self {
            params,
            high_frequency_completion: None,
        }
    }

    /// Return a calculation-specific model with a hypothetical UV completion.
    pub const fn with_high_frequency_completion(
        mut self,
        completion: HighFrequencyCompletion,
    ) -> Self {
        self.high_frequency_completion = Some(completion);
        self
    }

    fn epsilon_imaginary(self, xi: f64) -> f64 {
        let uncompleted = self.params.epsilon_imaginary(xi);
        let Some(completion) = self.high_frequency_completion else {
            return uncompleted;
        };

        let resonance = completion.resonance_ev * EV_TO_RADS;
        let damping = completion.damping_ev * EV_TO_RADS;
        let background_strength = self.params.eps_inf - 1.0;
        let completed_background = 1.0
            + background_strength * resonance * resonance
                / (resonance * resonance + xi * xi + damping * xi);
        uncompleted - self.params.eps_inf + completed_background
    }
}

impl<'a> From<&'a DrudeLorentzParams> for LifshitzModel<'a> {
    fn from(params: &'a DrudeLorentzParams) -> Self {
        Self::new(params)
    }
}

/// One finite layer, ordered from the vacuum gap toward the substrate.
#[derive(Debug, Clone, Copy)]
pub struct Layer<'a> {
    /// Local optical response of the layer.
    pub model: LifshitzModel<'a>,
    /// Physical layer thickness in meters. Zero thickness is permitted.
    pub thickness_m: f64,
}

impl<'a> Layer<'a> {
    /// Construct a layer from an unmodified Drude-Lorentz model.
    pub const fn new(params: &'a DrudeLorentzParams, thickness_m: f64) -> Self {
        Self {
            model: LifshitzModel::new(params),
            thickness_m,
        }
    }

    /// Construct a layer from a calculation-specific optical response.
    pub const fn from_model(model: LifshitzModel<'a>, thickness_m: f64) -> Self {
        Self { model, thickness_m }
    }
}

/// Semi-infinite termination of a multilayer stack.
#[derive(Debug, Clone, Copy)]
pub enum HalfSpace<'a> {
    /// A local Drude-Lorentz material.
    Material(LifshitzModel<'a>),
    /// An ideal conductor used as an analytic and numerical reference.
    IdealConductor,
}

impl<'a> From<&'a DrudeLorentzParams> for HalfSpace<'a> {
    fn from(params: &'a DrudeLorentzParams) -> Self {
        Self::Material(LifshitzModel::new(params))
    }
}

/// A planar body viewed from the vacuum gap.
#[derive(Debug, Clone)]
pub struct Multilayer<'a> {
    /// Finite layers ordered from the vacuum gap toward the substrate.
    pub layers: Vec<Layer<'a>>,
    /// Semi-infinite substrate below the finite layers.
    pub substrate: HalfSpace<'a>,
}

impl<'a> Multilayer<'a> {
    /// Construct a finite stack over a semi-infinite substrate.
    pub fn new(layers: Vec<Layer<'a>>, substrate: HalfSpace<'a>) -> Self {
        Self { layers, substrate }
    }

    /// Construct a material half-space without finite coating layers.
    pub fn half_space(params: &'a DrudeLorentzParams) -> Self {
        Self::new(Vec::new(), HalfSpace::from(params))
    }

    /// Construct an ideal-conductor half-space for reference calculations.
    pub fn ideal_conductor() -> Self {
        Self::new(Vec::new(), HalfSpace::IdealConductor)
    }
}

/// Gauss-Legendre orders for the zero-temperature polar integral.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroTemperatureOptions {
    /// Order for the transformed `q in [0, infinity)` integral.
    pub radial_order: usize,
    /// Order for the complete `mu in [0, 1]` angular integral.
    pub angular_order: usize,
}

impl Default for ZeroTemperatureOptions {
    fn default() -> Self {
        Self {
            radial_order: 96,
            angular_order: 48,
        }
    }
}

/// Gauss-Legendre and Matsubara limits for finite-temperature pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteTemperatureOptions {
    /// Order for each transformed semi-infinite radial integral.
    pub radial_order: usize,
    /// Number of positive Matsubara modes; the zero mode is always included.
    pub matsubara_terms: usize,
}

impl Default for FiniteTemperatureOptions {
    fn default() -> Self {
        Self {
            radial_order: 64,
            matsubara_terms: 500,
        }
    }
}

/// Input or response failure from a multilayer Lifshitz calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifshitzError {
    /// A scalar input lies outside its finite physical domain.
    InvalidInput(&'static str),
    /// A layer has a non-finite or negative thickness.
    InvalidLayerThickness { side: &'static str, index: usize },
    /// A material parameter cannot define a finite passive local response.
    InvalidMaterial(&'static str),
    /// The finite-temperature path received a non-local or plasma response.
    UnsupportedThermalModel,
    /// Numerical evaluation produced a non-finite result.
    NonFiniteResult,
}

impl fmt::Display for LifshitzError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid Lifshitz input: {message}"),
            Self::InvalidLayerThickness { side, index } => {
                write!(formatter, "{side} layer {index} has invalid thickness")
            }
            Self::InvalidMaterial(message) => {
                write!(formatter, "invalid Drude-Lorentz material: {message}")
            }
            Self::UnsupportedThermalModel => write!(
                formatter,
                "finite-temperature pressure requires local Drude conductors or dielectrics"
            ),
            Self::NonFiniteResult => write!(formatter, "Lifshitz integral is non-finite"),
        }
    }
}

impl Error for LifshitzError {}

#[derive(Debug, Clone, Copy)]
enum StaticResponse {
    Dielectric(f64),
    Conductor,
}

#[derive(Debug, Clone, Copy)]
enum Polarization {
    TransverseElectric,
    TransverseMagnetic,
}

fn validate_finite_nonnegative(value: f64, message: &'static str) -> Result<(), LifshitzError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(LifshitzError::InvalidMaterial(message))
    }
}

fn validate_scattering(scattering: &ScatteringModel) -> Result<(), LifshitzError> {
    match scattering {
        ScatteringModel::Constant { gamma_ev } => {
            validate_finite_nonnegative(*gamma_ev, "constant scattering rate")
        }
        ScatteringModel::LinearInOmega { gamma_0_ev, alpha } => {
            validate_finite_nonnegative(*gamma_0_ev, "linear scattering intercept")?;
            validate_finite_nonnegative(*alpha, "linear scattering slope")
        }
        ScatteringModel::PowerLaw {
            gamma_0_ev,
            omega_scale_ev,
            exponent,
        } => {
            validate_finite_nonnegative(*gamma_0_ev, "power-law scattering rate")?;
            if !omega_scale_ev.is_finite() || *omega_scale_ev <= 0.0 {
                return Err(LifshitzError::InvalidMaterial(
                    "power-law scattering frequency scale",
                ));
            }
            validate_finite_nonnegative(*exponent, "power-law scattering exponent")
        }
        ScatteringModel::DrudeSmith {
            gamma_ev,
            backscatter_c,
        } => {
            validate_finite_nonnegative(*gamma_ev, "Drude-Smith scattering rate")?;
            if backscatter_c.is_finite() && (-1.0..=0.0).contains(backscatter_c) {
                Ok(())
            } else {
                Err(LifshitzError::InvalidMaterial(
                    "Drude-Smith backscatter coefficient",
                ))
            }
        }
        ScatteringModel::Tabulated { omega_ev, gamma_ev } => {
            if omega_ev.is_empty() || omega_ev.len() != gamma_ev.len() {
                return Err(LifshitzError::InvalidMaterial(
                    "tabulated scattering dimensions",
                ));
            }
            if omega_ev
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
                || omega_ev.windows(2).any(|pair| pair[1] <= pair[0])
                || gamma_ev
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(LifshitzError::InvalidMaterial(
                    "tabulated scattering values",
                ));
            }
            Ok(())
        }
    }
}

fn validate_model(model: LifshitzModel<'_>) -> Result<(), LifshitzError> {
    if !model.params.eps_inf.is_finite() || model.params.eps_inf < 1.0 {
        return Err(LifshitzError::InvalidMaterial(
            "high-frequency permittivity",
        ));
    }
    if let Some(drude) = model.params.drude {
        validate_finite_nonnegative(drude.omega_p_ev, "Drude plasma energy")?;
        validate_finite_nonnegative(drude.gamma_ev, "Drude damping energy")?;
        if !drude.eps_inf.is_finite() || drude.eps_inf < 1.0 {
            return Err(LifshitzError::InvalidMaterial(
                "standalone Drude high-frequency permittivity",
            ));
        }
    }
    if let Some(extended) = &model.params.extended_drude {
        validate_finite_nonnegative(extended.omega_p_ev, "extended-Drude plasma energy")?;
        if !extended.eps_inf.is_finite() || extended.eps_inf < 1.0 {
            return Err(LifshitzError::InvalidMaterial(
                "extended-Drude high-frequency permittivity",
            ));
        }
        validate_scattering(&extended.scattering)?;
    }
    for oscillator in &model.params.oscillators {
        validate_finite_nonnegative(oscillator.strength, "oscillator strength")?;
        if !oscillator.omega_0_ev.is_finite() || oscillator.omega_0_ev <= 0.0 {
            return Err(LifshitzError::InvalidMaterial(
                "oscillator resonance energy",
            ));
        }
        validate_finite_nonnegative(oscillator.gamma_ev, "oscillator damping energy")?;
    }
    if let Some(completion) = model.high_frequency_completion {
        if !completion.resonance_ev.is_finite() || completion.resonance_ev <= 0.0 {
            return Err(LifshitzError::InvalidMaterial(
                "UV completion resonance energy",
            ));
        }
        validate_finite_nonnegative(completion.damping_ev, "UV completion damping energy")?;
    }
    Ok(())
}

fn validate_stack(
    stack: &Multilayer<'_>,
    side: &'static str,
    local_drude_only: bool,
) -> Result<(), LifshitzError> {
    for (index, layer) in stack.layers.iter().enumerate() {
        if !layer.thickness_m.is_finite() || layer.thickness_m < 0.0 {
            return Err(LifshitzError::InvalidLayerThickness { side, index });
        }
        validate_model(layer.model)?;
        if local_drude_only {
            validate_local_drude_model(layer.model)?;
        }
    }
    match stack.substrate {
        HalfSpace::Material(model) => {
            validate_model(model)?;
            if local_drude_only {
                validate_local_drude_model(model)?;
            }
        }
        HalfSpace::IdealConductor if local_drude_only => {
            return Err(LifshitzError::UnsupportedThermalModel);
        }
        HalfSpace::IdealConductor => {}
    }
    Ok(())
}

fn validate_local_drude_model(model: LifshitzModel<'_>) -> Result<(), LifshitzError> {
    if model.params.extended_drude.is_some() {
        return Err(LifshitzError::UnsupportedThermalModel);
    }
    if model
        .params
        .drude
        .is_some_and(|drude| drude.omega_p_ev > 0.0 && drude.gamma_ev <= 0.0)
    {
        return Err(LifshitzError::UnsupportedThermalModel);
    }
    Ok(())
}

fn validate_geometry(
    gap_m: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    local_drude_only: bool,
) -> Result<(), LifshitzError> {
    if !gap_m.is_finite() || gap_m <= 0.0 {
        return Err(LifshitzError::InvalidInput(
            "gap must be finite and positive",
        ));
    }
    validate_stack(left, "left", local_drude_only)?;
    validate_stack(right, "right", local_drude_only)
}

fn quadrature(order: usize, message: &'static str) -> Result<GaussLegendre, LifshitzError> {
    let degree = NonZeroUsize::new(order).ok_or(LifshitzError::InvalidInput(message))?;
    Ok(GaussLegendre::new(degree))
}

fn kappa(permittivity: f64, xi_over_c: f64, k_parallel: f64) -> f64 {
    (permittivity * xi_over_c * xi_over_c + k_parallel * k_parallel).sqrt()
}

fn interface_reflection(
    incident_permittivity: f64,
    transmitted_permittivity: f64,
    incident_kappa: f64,
    transmitted_kappa: f64,
    polarization: Polarization,
) -> f64 {
    match polarization {
        Polarization::TransverseElectric => {
            (incident_kappa - transmitted_kappa) / (incident_kappa + transmitted_kappa)
        }
        Polarization::TransverseMagnetic => {
            let numerator = transmitted_permittivity * incident_kappa
                - incident_permittivity * transmitted_kappa;
            let denominator = transmitted_permittivity * incident_kappa
                + incident_permittivity * transmitted_kappa;
            numerator / denominator
        }
    }
}

fn surface_reflections(stack: &Multilayer<'_>, xi: f64, k_parallel: f64) -> (f64, f64) {
    let xi_over_c = xi / C;
    if stack.layers.is_empty() {
        return match stack.substrate {
            HalfSpace::IdealConductor => (1.0, -1.0),
            HalfSpace::Material(model) => {
                let permittivity = model.epsilon_imaginary(xi);
                let vacuum_kappa = kappa(1.0, xi_over_c, k_parallel);
                let material_kappa = kappa(permittivity, xi_over_c, k_parallel);
                (
                    interface_reflection(
                        1.0,
                        permittivity,
                        vacuum_kappa,
                        material_kappa,
                        Polarization::TransverseMagnetic,
                    ),
                    interface_reflection(
                        1.0,
                        permittivity,
                        vacuum_kappa,
                        material_kappa,
                        Polarization::TransverseElectric,
                    ),
                )
            }
        };
    }

    let layer_permittivities: Vec<f64> = stack
        .layers
        .iter()
        .map(|layer| layer.model.epsilon_imaginary(xi))
        .collect();
    let last_index = stack.layers.len() - 1;
    let last_permittivity = layer_permittivities[last_index];
    let last_kappa = kappa(last_permittivity, xi_over_c, k_parallel);
    let (mut effective_tm, mut effective_te) = match stack.substrate {
        HalfSpace::IdealConductor => (1.0, -1.0),
        HalfSpace::Material(model) => {
            let substrate_permittivity = model.epsilon_imaginary(xi);
            let substrate_kappa = kappa(substrate_permittivity, xi_over_c, k_parallel);
            (
                interface_reflection(
                    last_permittivity,
                    substrate_permittivity,
                    last_kappa,
                    substrate_kappa,
                    Polarization::TransverseMagnetic,
                ),
                interface_reflection(
                    last_permittivity,
                    substrate_permittivity,
                    last_kappa,
                    substrate_kappa,
                    Polarization::TransverseElectric,
                ),
            )
        }
    };

    for layer_index in (0..stack.layers.len()).rev() {
        let layer = stack.layers[layer_index];
        let layer_permittivity = layer_permittivities[layer_index];
        let layer_kappa = kappa(layer_permittivity, xi_over_c, k_parallel);
        let incident_permittivity = if layer_index == 0 {
            1.0
        } else {
            layer_permittivities[layer_index - 1]
        };
        let incident_kappa = kappa(incident_permittivity, xi_over_c, k_parallel);
        let interface_tm = interface_reflection(
            incident_permittivity,
            layer_permittivity,
            incident_kappa,
            layer_kappa,
            Polarization::TransverseMagnetic,
        );
        let interface_te = interface_reflection(
            incident_permittivity,
            layer_permittivity,
            incident_kappa,
            layer_kappa,
            Polarization::TransverseElectric,
        );
        let propagation = (-2.0 * layer_kappa * layer.thickness_m).exp();
        effective_tm = (interface_tm + effective_tm * propagation)
            / (1.0 + interface_tm * effective_tm * propagation);
        effective_te = (interface_te + effective_te * propagation)
            / (1.0 + interface_te * effective_te * propagation);
    }

    (effective_tm, effective_te)
}

fn static_response(model: LifshitzModel<'_>) -> StaticResponse {
    if model
        .params
        .drude
        .is_some_and(|drude| drude.omega_p_ev > 0.0)
    {
        StaticResponse::Conductor
    } else {
        let permittivity = model.params.eps_inf
            + model
                .params
                .oscillators
                .iter()
                .map(|oscillator| oscillator.strength)
                .sum::<f64>();
        StaticResponse::Dielectric(permittivity)
    }
}

fn static_tm_interface(incident: StaticResponse, transmitted: StaticResponse) -> f64 {
    match (incident, transmitted) {
        (StaticResponse::Dielectric(left), StaticResponse::Dielectric(right)) => {
            (right - left) / (right + left)
        }
        (StaticResponse::Dielectric(_), StaticResponse::Conductor) => 1.0,
        (StaticResponse::Conductor, StaticResponse::Dielectric(_)) => -1.0,
        (StaticResponse::Conductor, StaticResponse::Conductor) => 0.0,
    }
}

fn zero_mode_tm_reflection(stack: &Multilayer<'_>, k_parallel: f64) -> f64 {
    let active_layers: Vec<&Layer<'_>> = stack
        .layers
        .iter()
        .filter(|layer| layer.thickness_m > 0.0)
        .collect();
    if active_layers
        .first()
        .is_some_and(|layer| matches!(static_response(layer.model), StaticResponse::Conductor))
    {
        return 1.0;
    }

    if active_layers.is_empty() {
        return match stack.substrate {
            HalfSpace::IdealConductor => 1.0,
            HalfSpace::Material(model) => {
                static_tm_interface(StaticResponse::Dielectric(1.0), static_response(model))
            }
        };
    }

    let layer_responses: Vec<StaticResponse> = active_layers
        .iter()
        .map(|layer| static_response(layer.model))
        .collect();
    let last_index = active_layers.len() - 1;
    let substrate_response = match stack.substrate {
        HalfSpace::IdealConductor => StaticResponse::Conductor,
        HalfSpace::Material(model) => static_response(model),
    };
    let mut effective = static_tm_interface(layer_responses[last_index], substrate_response);

    for layer_index in (0..active_layers.len()).rev() {
        let incident_response = if layer_index == 0 {
            StaticResponse::Dielectric(1.0)
        } else {
            layer_responses[layer_index - 1]
        };
        let interface = static_tm_interface(incident_response, layer_responses[layer_index]);
        let propagation = (-2.0 * k_parallel * active_layers[layer_index].thickness_m).exp();
        effective =
            (interface + effective * propagation) / (1.0 + interface * effective * propagation);
    }
    effective
}

fn round_trip_fraction(reflection_product: f64, attenuation: f64) -> f64 {
    let round_trip = reflection_product * attenuation;
    round_trip / (1.0 - round_trip)
}

fn transformed_q(unit_node: f64) -> (f64, f64) {
    let complement = 1.0 - unit_node;
    (unit_node / complement, 1.0 / (complement * complement))
}

fn zero_temperature_integrals(
    gap_m: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    options: ZeroTemperatureOptions,
) -> Result<(f64, f64), LifshitzError> {
    validate_geometry(gap_m, left, right, false)?;
    let radial_quadrature = quadrature(
        options.radial_order,
        "radial quadrature order must be non-zero",
    )?;
    let angular_quadrature = quadrature(
        options.angular_order,
        "angular quadrature order must be non-zero",
    )?;

    let pressure_integral = radial_quadrature.integrate(0.0, 1.0, |unit_node| {
        let (q, jacobian) = transformed_q(unit_node);
        let attenuation = (-2.0 * q).exp();
        let angular_integral = angular_quadrature.integrate(0.0, 1.0, |mu| {
            let xi = C * q * mu / gap_m;
            let k_parallel = q * (1.0 - mu * mu).sqrt() / gap_m;
            let (left_tm, left_te) = surface_reflections(left, xi, k_parallel);
            let (right_tm, right_te) = surface_reflections(right, xi, k_parallel);
            round_trip_fraction(left_tm * right_tm, attenuation)
                + round_trip_fraction(left_te * right_te, attenuation)
        });
        q.powi(3) * jacobian * angular_integral
    });

    let energy_integral = radial_quadrature.integrate(0.0, 1.0, |unit_node| {
        let (q, jacobian) = transformed_q(unit_node);
        let attenuation = (-2.0 * q).exp();
        let angular_integral = angular_quadrature.integrate(0.0, 1.0, |mu| {
            let xi = C * q * mu / gap_m;
            let k_parallel = q * (1.0 - mu * mu).sqrt() / gap_m;
            let (left_tm, left_te) = surface_reflections(left, xi, k_parallel);
            let (right_tm, right_te) = surface_reflections(right, xi, k_parallel);
            (-(left_tm * right_tm) * attenuation).ln_1p()
                + (-(left_te * right_te) * attenuation).ln_1p()
        });
        q * q * jacobian * angular_integral
    });

    if pressure_integral.is_finite() && energy_integral.is_finite() {
        Ok((pressure_integral, energy_integral))
    } else {
        Err(LifshitzError::NonFiniteResult)
    }
}

/// Compute zero-temperature planar pressure in Pa.
///
/// Negative pressure denotes attraction. The transformed radial quadrature
/// covers `q in [0, infinity)` without a finite momentum cutoff.
pub fn zero_temperature_pressure(
    gap_m: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    options: ZeroTemperatureOptions,
) -> Result<f64, LifshitzError> {
    let (pressure_integral, _) = zero_temperature_integrals(gap_m, left, right, options)?;
    Ok(-HBAR_J_S * C * pressure_integral / (2.0 * PI * PI * gap_m.powi(4)))
}

/// Compute zero-temperature planar energy per unit area in J/m^2.
pub fn zero_temperature_energy_per_area(
    gap_m: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    options: ZeroTemperatureOptions,
) -> Result<f64, LifshitzError> {
    let (_, energy_integral) = zero_temperature_integrals(gap_m, left, right, options)?;
    Ok(HBAR_J_S * C * energy_integral / (4.0 * PI * PI * gap_m.powi(3)))
}

fn zero_mode_pressure_integral(
    gap_m: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    radial_quadrature: &GaussLegendre,
) -> f64 {
    radial_quadrature.integrate(0.0, 1.0, |unit_node| {
        let (q, jacobian) = transformed_q(unit_node);
        let k_parallel = q / gap_m;
        let reflection_product =
            zero_mode_tm_reflection(left, k_parallel) * zero_mode_tm_reflection(right, k_parallel);
        q * q * jacobian * round_trip_fraction(reflection_product, (-2.0 * q).exp())
    })
}

/// Compute the half-weight local-Drude zero-mode pressure in Pa.
///
/// A positive-thickness conducting outer layer receives the exact static
/// response `r_TM = 1` and `r_TE = 0`, independent of buried layers.
/// Ideal-conductor half-spaces are rejected because their nonzero TE zero mode
/// lies outside the local-Drude prescription implemented by this function.
pub fn local_drude_zero_mode_pressure(
    gap_m: f64,
    temperature_k: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    radial_order: usize,
) -> Result<f64, LifshitzError> {
    validate_geometry(gap_m, left, right, true)?;
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(LifshitzError::InvalidInput(
            "temperature must be finite and positive",
        ));
    }
    let radial_quadrature = quadrature(radial_order, "radial quadrature order must be non-zero")?;
    let integral = zero_mode_pressure_integral(gap_m, left, right, &radial_quadrature);
    let pressure = -K_B_J_K * temperature_k * integral / (2.0 * PI * gap_m.powi(3));
    if pressure.is_finite() {
        Ok(pressure)
    } else {
        Err(LifshitzError::NonFiniteResult)
    }
}

/// Compute local-Drude finite-temperature planar pressure in Pa.
///
/// The Matsubara sum includes the half-weight zero mode and
/// `options.matsubara_terms` positive modes. Extended-Drude and zero-damping
/// plasma responses and ideal-conductor half-spaces are rejected because they
/// do not satisfy this function's local-Drude zero-mode contract.
pub fn local_drude_finite_temperature_pressure(
    gap_m: f64,
    temperature_k: f64,
    left: &Multilayer<'_>,
    right: &Multilayer<'_>,
    options: FiniteTemperatureOptions,
) -> Result<f64, LifshitzError> {
    validate_geometry(gap_m, left, right, true)?;
    if !temperature_k.is_finite() || temperature_k <= 0.0 {
        return Err(LifshitzError::InvalidInput(
            "temperature must be finite and positive",
        ));
    }
    let radial_quadrature = quadrature(
        options.radial_order,
        "radial quadrature order must be non-zero",
    )?;
    let mut dimensionless_sum =
        0.5 * zero_mode_pressure_integral(gap_m, left, right, &radial_quadrature);
    let matsubara_spacing = 2.0 * PI * K_B_J_K * temperature_k / HBAR_J_S;

    for mode_index in 1..=options.matsubara_terms {
        let xi = mode_index as f64 * matsubara_spacing;
        let minimum_q = xi * gap_m / C;
        let mode_integral = radial_quadrature.integrate(0.0, 1.0, |unit_node| {
            let (offset, jacobian) = transformed_q(unit_node);
            let q = minimum_q + offset;
            let k_parallel = (q * q - minimum_q * minimum_q).sqrt() / gap_m;
            let (left_tm, left_te) = surface_reflections(left, xi, k_parallel);
            let (right_tm, right_te) = surface_reflections(right, xi, k_parallel);
            let attenuation = (-2.0 * q).exp();
            q * q
                * jacobian
                * (round_trip_fraction(left_tm * right_tm, attenuation)
                    + round_trip_fraction(left_te * right_te, attenuation))
        });
        dimensionless_sum += mode_integral;
    }

    let pressure = -K_B_J_K * temperature_k * dimensionless_sum / (PI * gap_m.powi(3));
    if pressure.is_finite() {
        Ok(pressure)
    } else {
        Err(LifshitzError::NonFiniteResult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DrudeParams, ExtendedDrudeParams, OpticalLorentzOscillator};

    fn dielectric(eps_inf: f64, strength: f64) -> DrudeLorentzParams {
        DrudeLorentzParams {
            drude: None,
            oscillators: vec![OpticalLorentzOscillator {
                strength,
                omega_0_ev: 5.0,
                gamma_ev: 0.1,
            }],
            eps_inf,
            extended_drude: None,
        }
    }

    fn local_conductor() -> DrudeLorentzParams {
        DrudeLorentzParams {
            drude: Some(DrudeParams {
                omega_p_ev: 9.0,
                gamma_ev: 0.035,
                eps_inf: 1.0,
            }),
            oscillators: Vec::new(),
            eps_inf: 1.0,
            extended_drude: None,
        }
    }

    #[test]
    fn validation_rejects_sub_vacuum_asymptotic_permittivity() {
        let material = dielectric(0.9, 1.0);
        assert_eq!(
            validate_model(LifshitzModel::new(&material)),
            Err(LifshitzError::InvalidMaterial(
                "high-frequency permittivity"
            ))
        );
    }

    #[test]
    fn validation_rejects_negative_singleton_scattering_frequency() {
        let material = DrudeLorentzParams {
            drude: None,
            oscillators: Vec::new(),
            eps_inf: 1.0,
            extended_drude: Some(ExtendedDrudeParams {
                omega_p_ev: 1.0,
                scattering: ScatteringModel::Tabulated {
                    omega_ev: vec![-1.0],
                    gamma_ev: vec![0.1],
                },
                eps_inf: 1.0,
            }),
        };
        assert_eq!(
            validate_model(LifshitzModel::new(&material)),
            Err(LifshitzError::InvalidMaterial(
                "tabulated scattering values"
            ))
        );
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        ((actual - expected) / expected).abs()
    }

    #[test]
    fn ideal_conductor_limit_matches_planar_results() {
        let gap_m = 200.0e-9;
        let ideal = Multilayer::ideal_conductor();
        let options = ZeroTemperatureOptions {
            radial_order: 128,
            angular_order: 24,
        };
        let pressure = zero_temperature_pressure(gap_m, &ideal, &ideal, options).unwrap();
        let energy = zero_temperature_energy_per_area(gap_m, &ideal, &ideal, options).unwrap();
        let expected_pressure = -PI.powi(2) * HBAR_J_S * C / (240.0 * gap_m.powi(4));
        let expected_energy = -PI.powi(2) * HBAR_J_S * C / (720.0 * gap_m.powi(3));

        assert!(relative_error(pressure, expected_pressure) < 2.0e-9);
        assert!(relative_error(energy, expected_energy) < 2.0e-9);
    }

    #[test]
    fn zero_thickness_layer_equals_direct_substrate() {
        let coating = dielectric(2.0, 1.0);
        let substrate = dielectric(3.0, 2.0);
        let reference = Multilayer::half_space(&substrate);
        let coated = Multilayer::new(vec![Layer::new(&coating, 0.0)], HalfSpace::from(&substrate));
        let opposing = Multilayer::ideal_conductor();
        let options = ZeroTemperatureOptions {
            radial_order: 64,
            angular_order: 32,
        };
        let direct =
            zero_temperature_energy_per_area(150.0e-9, &reference, &opposing, options).unwrap();
        let with_zero_layer =
            zero_temperature_energy_per_area(150.0e-9, &coated, &opposing, options).unwrap();

        assert!(relative_error(with_zero_layer, direct) < 2.0e-13);
    }

    #[test]
    fn swapping_identical_layers_preserves_energy() {
        let repeated = dielectric(2.5, 1.5);
        let substrate = dielectric(4.0, 0.5);
        let first = Multilayer::new(
            vec![
                Layer::new(&repeated, 20.0e-9),
                Layer::new(&repeated, 70.0e-9),
            ],
            HalfSpace::from(&substrate),
        );
        let swapped = Multilayer::new(
            vec![
                Layer::new(&repeated, 70.0e-9),
                Layer::new(&repeated, 20.0e-9),
            ],
            HalfSpace::from(&substrate),
        );
        let opposing = Multilayer::ideal_conductor();
        let options = ZeroTemperatureOptions {
            radial_order: 64,
            angular_order: 32,
        };
        let first_energy =
            zero_temperature_energy_per_area(180.0e-9, &first, &opposing, options).unwrap();
        let swapped_energy =
            zero_temperature_energy_per_area(180.0e-9, &swapped, &opposing, options).unwrap();

        assert!(relative_error(first_energy, swapped_energy) < 2.0e-13);
    }

    #[test]
    fn pressure_matches_negative_energy_derivative() {
        let left_material = dielectric(2.0, 2.5);
        let right_material = dielectric(3.0, 1.0);
        let left = Multilayer::half_space(&left_material);
        let right = Multilayer::half_space(&right_material);
        let gap_m = 220.0e-9;
        let step_m = gap_m * 2.0e-4;
        let options = ZeroTemperatureOptions {
            radial_order: 80,
            angular_order: 40,
        };
        let pressure = zero_temperature_pressure(gap_m, &left, &right, options).unwrap();
        let energy_below =
            zero_temperature_energy_per_area(gap_m - step_m, &left, &right, options).unwrap();
        let energy_above =
            zero_temperature_energy_per_area(gap_m + step_m, &left, &right, options).unwrap();
        let negative_derivative = -(energy_above - energy_below) / (2.0 * step_m);

        assert!(relative_error(pressure, negative_derivative) < 2.0e-5);
    }

    #[test]
    fn common_conducting_cap_cancels_zero_mode_differential() {
        let conductor = local_conductor();
        let buried_a = dielectric(2.0, 1.0);
        let buried_b = dielectric(6.0, 3.0);
        let left_a = Multilayer::new(
            vec![Layer::new(&conductor, 10.0e-9)],
            HalfSpace::from(&buried_a),
        );
        let left_b = Multilayer::new(
            vec![Layer::new(&conductor, 10.0e-9)],
            HalfSpace::from(&buried_b),
        );
        let right = Multilayer::half_space(&conductor);
        let pressure_a =
            local_drude_zero_mode_pressure(200.0e-9, 300.0, &left_a, &right, 96).unwrap();
        let pressure_b =
            local_drude_zero_mode_pressure(200.0e-9, 300.0, &left_b, &right, 96).unwrap();

        assert_eq!(pressure_a - pressure_b, 0.0);
    }

    #[test]
    fn zero_thickness_conductor_does_not_screen_static_response() {
        let conductor = local_conductor();
        let substrate = dielectric(3.0, 2.0);
        let direct = Multilayer::half_space(&substrate);
        let zero_coated = Multilayer::new(
            vec![Layer::new(&conductor, 0.0)],
            HalfSpace::from(&substrate),
        );
        let opposing = Multilayer::half_space(&conductor);
        let direct_pressure =
            local_drude_zero_mode_pressure(200.0e-9, 300.0, &direct, &opposing, 96).unwrap();
        let zero_coated_pressure =
            local_drude_zero_mode_pressure(200.0e-9, 300.0, &zero_coated, &opposing, 96).unwrap();

        assert_eq!(zero_coated_pressure, direct_pressure);
    }

    #[test]
    fn local_drude_thermal_apis_reject_ideal_conductor_half_spaces() {
        let conductor = local_conductor();
        let material = Multilayer::half_space(&conductor);
        let ideal = Multilayer::ideal_conductor();

        assert_eq!(
            local_drude_zero_mode_pressure(200.0e-9, 300.0, &material, &ideal, 16),
            Err(LifshitzError::UnsupportedThermalModel)
        );
        assert_eq!(
            local_drude_finite_temperature_pressure(
                200.0e-9,
                300.0,
                &ideal,
                &material,
                FiniteTemperatureOptions {
                    radial_order: 16,
                    matsubara_terms: 1,
                },
            ),
            Err(LifshitzError::UnsupportedThermalModel)
        );
    }

    #[test]
    fn uv_completion_is_calculation_scoped() {
        let material = dielectric(4.0, 1.0);
        let original_eps_inf = material.eps_inf;
        let completed = LifshitzModel::new(&material)
            .with_high_frequency_completion(HighFrequencyCompletion::new(20.0, 1.0));
        let high_frequency = 1.0e20;

        assert_eq!(material.eps_inf, original_eps_inf);
        assert!(completed.epsilon_imaginary(high_frequency) < 1.000_001);
        assert!(material.epsilon_imaginary(high_frequency) > 3.999_999);
    }
}
