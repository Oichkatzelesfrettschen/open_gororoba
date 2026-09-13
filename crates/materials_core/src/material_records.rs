//! Condition-bound material, specimen, measurement, and quantity records.
//!
//! Chemical formulae and common names identify material families. They do not
//! identify a phase, specimen, measurement, or model. These types provide an
//! extensible boundary around those identities without growing one universal
//! material struct.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// Stable identifier in the material evidence graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecordId(pub String);

impl RecordId {
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err("record identifier must be nonempty".to_owned());
        }
        Ok(Self(value))
    }

    fn validate(&self, label: &str) -> Result<(), String> {
        require_nonempty(label, &self.0)
    }
}

/// Broad quantity family used for discovery without constraining payload shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyFamily {
    Structure,
    Mechanical,
    Thermal,
    ElectricalElectronic,
    OpticalPhotonic,
    MagneticSpin,
    ChemicalElectrochemical,
    SurfaceInterface,
    RadiationNuclear,
    BiologicalToxicological,
    SafetyOutgassing,
    LifeCycleSupplyChain,
}

/// Evidence class for a populated value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    ExperimentalDirect,
    ExperimentalFitted,
    Computed,
    InferredProxy,
}

/// Record that produced a quantity without conflating acquisition and models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "origin")]
pub enum QuantityOrigin {
    Measurement { measurement_id: RecordId },
    ModelRun { model_run_id: RecordId },
}

/// Evidence details required by each admission class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "class")]
pub enum EvidenceBasis {
    ExperimentalDirect,
    ExperimentalFitted {
        input_quantity_ids: Vec<RecordId>,
        fit_model: String,
        fit_model_version: String,
        residual_artifact_id: RecordId,
        parameter_covariance: Uncertainty,
    },
    Computed,
    InferredProxy {
        rationale: String,
    },
}

impl EvidenceBasis {
    pub const fn class(&self) -> EvidenceClass {
        match self {
            Self::ExperimentalDirect => EvidenceClass::ExperimentalDirect,
            Self::ExperimentalFitted { .. } => EvidenceClass::ExperimentalFitted,
            Self::Computed => EvidenceClass::Computed,
            Self::InferredProxy { .. } => EvidenceClass::InferredProxy,
        }
    }
}

/// Typed reason that a quantity has no observed numerical payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Missingness {
    NotMeasured,
    BelowDetectionLimit { upper_bound: f64, unit: String },
    NotApplicable,
    Withheld { reason: String },
    Unknown,
}

/// Shape of an observed quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "representation")]
pub enum QuantityPayload {
    Scalar {
        value: f64,
    },
    Vector {
        values: Vec<f64>,
        basis: String,
    },
    Tensor {
        values_row_major: Vec<f64>,
        rows: usize,
        columns: usize,
        basis: String,
    },
    Spectrum {
        abscissa: Vec<f64>,
        ordinate: Vec<f64>,
        abscissa_unit: String,
    },
}

/// Observed payload or typed absence. The variants cannot be confused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum QuantityObservation {
    Observed { payload: QuantityPayload },
    Missing { reason: Missingness },
}

/// Uncertainty attached to an observed quantity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Uncertainty {
    Standard {
        value: f64,
        unit: String,
    },
    Interval {
        lower: f64,
        upper: f64,
        unit: String,
    },
    Covariance {
        dimension: usize,
        values_row_major: Vec<f64>,
        unit_squared: String,
    },
    NotReported {
        rationale: String,
    },
}

/// Material family identity independent of a particular state or specimen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Material {
    pub material_id: RecordId,
    pub preferred_name: String,
    pub formula: String,
}

/// Thermodynamic, structural, and history-dependent state of a material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialState {
    pub state_id: RecordId,
    pub material_id: RecordId,
    pub phase_id: String,
    pub temperature_k: Option<f64>,
    pub pressure_pa: Option<f64>,
    pub atmosphere: Option<String>,
    pub phase_fraction: Option<f64>,
    pub orientation: Option<String>,
    pub strain: Option<Vec<f64>>,
    pub applied_fields: BTreeMap<String, String>,
    pub time_since_processing_s: Option<f64>,
    pub history: Vec<String>,
}

/// Physical specimen carrying preparation and geometry metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Specimen {
    pub specimen_id: RecordId,
    pub state_id: RecordId,
    pub purity_assay: Option<String>,
    pub composition_assay: Option<String>,
    pub synthesis_or_deposition: String,
    pub anneal_history: Option<String>,
    pub thickness_m: Option<f64>,
    pub substrate_specimen_id: Option<RecordId>,
    pub adhesion_layers: Vec<String>,
    pub grain_size_m: Option<f64>,
    pub texture: Option<String>,
    pub porosity_fraction: Option<f64>,
    pub roughness_rms_m: Option<f64>,
    pub geometry: String,
}

/// Source-byte and transformation provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub source_id: String,
    pub doi_or_stable_id: String,
    pub locator: String,
    pub license: String,
    pub retrieval_date: String,
    pub source_sha256: String,
    pub parser_version: String,
    pub transformation_lineage: Vec<String>,
}

/// Measurement acquisition and processing identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub measurement_id: RecordId,
    pub specimen_id: RecordId,
    pub method: String,
    pub instrument: String,
    pub calibration_or_standard: String,
    pub operator_or_facility: String,
    pub raw_artifact_id: RecordId,
    pub processing_recipe_id: RecordId,
    pub repeat_count: usize,
    pub detection_limit: Option<(f64, String)>,
    pub provenance: Provenance,
}

/// Extensible quantity row attached to an acquisition or model run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantityValue {
    pub quantity_id: RecordId,
    pub origin: QuantityOrigin,
    pub family: PropertyFamily,
    pub quantity_kind: String,
    pub unit: String,
    pub tensor_component_or_basis: Option<String>,
    pub observation: QuantityObservation,
    pub uncertainty: Uncertainty,
    pub conditions: BTreeMap<String, String>,
    pub applicability_range: Option<String>,
    pub evidence: EvidenceBasis,
}

/// Versioned calculation whose output cannot satisfy direct-measurement scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRun {
    pub model_run_id: RecordId,
    pub model_name: String,
    pub model_version: String,
    pub code_artifact_id: RecordId,
    pub input_quantity_ids: Vec<RecordId>,
    pub specimen_id: RecordId,
    pub geometry: String,
    pub constitutive_model: String,
    pub conditions: BTreeMap<String, String>,
    pub state_history: Vec<String>,
    pub parameters: BTreeMap<String, String>,
    pub convergence_settings: BTreeMap<String, String>,
    pub validation_status: String,
}

/// Derived value with explicit measured inputs and propagated uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedValue {
    pub derived_value_id: RecordId,
    pub model_run_id: RecordId,
    pub input_quantity_ids: Vec<RecordId>,
    pub output: QuantityValue,
    pub propagated_uncertainty: Uncertainty,
}

/// Referentially validated material evidence graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialEvidenceGraph {
    pub materials: Vec<Material>,
    pub states: Vec<MaterialState>,
    pub specimens: Vec<Specimen>,
    pub measurements: Vec<Measurement>,
    pub quantities: Vec<QuantityValue>,
    pub model_runs: Vec<ModelRun>,
    pub derived_values: Vec<DerivedValue>,
}

fn require_nonempty(label: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be nonempty"))
    } else {
        Ok(())
    }
}

impl QuantityPayload {
    fn value_count(&self) -> usize {
        match self {
            Self::Scalar { .. } => 1,
            Self::Vector { values, .. } => values.len(),
            Self::Tensor {
                values_row_major, ..
            } => values_row_major.len(),
            Self::Spectrum { ordinate, .. } => ordinate.len(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Scalar { value } => {
                if !value.is_finite() {
                    return Err("scalar quantity must be finite".to_owned());
                }
            }
            Self::Vector { values, basis } => {
                require_nonempty("vector basis", basis)?;
                if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                    return Err("vector quantity requires finite values".to_owned());
                }
            }
            Self::Tensor {
                values_row_major,
                rows,
                columns,
                basis,
            } => {
                require_nonempty("tensor basis", basis)?;
                let Some(element_count) = rows.checked_mul(*columns) else {
                    return Err("tensor dimensions must match finite values".to_owned());
                };
                if *rows == 0
                    || *columns == 0
                    || values_row_major.len() != element_count
                    || values_row_major.iter().any(|value| !value.is_finite())
                {
                    return Err("tensor dimensions must match finite values".to_owned());
                }
            }
            Self::Spectrum {
                abscissa,
                ordinate,
                abscissa_unit,
            } => {
                require_nonempty("spectrum abscissa unit", abscissa_unit)?;
                if abscissa.is_empty()
                    || abscissa.len() != ordinate.len()
                    || abscissa.iter().any(|value| !value.is_finite())
                    || ordinate.iter().any(|value| !value.is_finite())
                {
                    return Err("spectrum axes must have equal nonzero finite length".to_owned());
                }
            }
        }
        Ok(())
    }
}

impl Uncertainty {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Standard { value, unit } => {
                require_nonempty("uncertainty unit", unit)?;
                if !value.is_finite() || *value < 0.0 {
                    return Err("standard uncertainty must be finite and nonnegative".to_owned());
                }
            }
            Self::Interval { lower, upper, unit } => {
                require_nonempty("uncertainty unit", unit)?;
                if !lower.is_finite() || !upper.is_finite() || lower > upper {
                    return Err("uncertainty interval is invalid".to_owned());
                }
            }
            Self::Covariance {
                dimension,
                values_row_major,
                unit_squared,
            } => {
                require_nonempty("covariance unit", unit_squared)?;
                let Some(element_count) = dimension.checked_mul(*dimension) else {
                    return Err("covariance dimensions must match finite values".to_owned());
                };
                if *dimension == 0
                    || values_row_major.len() != element_count
                    || values_row_major.iter().any(|value| !value.is_finite())
                {
                    return Err("covariance dimensions must match finite values".to_owned());
                }
                let matrix = DMatrix::from_row_slice(*dimension, *dimension, values_row_major);
                let scale = values_row_major
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max);
                let tolerance = 64.0 * f64::EPSILON * scale * (*dimension as f64);
                for row in 0..*dimension {
                    for column in (row + 1)..*dimension {
                        if (matrix[(row, column)] - matrix[(column, row)]).abs() > tolerance {
                            return Err("covariance matrix must be symmetric".to_owned());
                        }
                    }
                }
                let symmetric = (&matrix + matrix.transpose()) * 0.5;
                if symmetric
                    .symmetric_eigen()
                    .eigenvalues
                    .iter()
                    .any(|eigenvalue| *eigenvalue < -tolerance)
                {
                    return Err("covariance matrix must be positive semidefinite".to_owned());
                }
            }
            Self::NotReported { rationale } => {
                require_nonempty("uncertainty rationale", rationale)?;
            }
        }
        Ok(())
    }

    fn validate_quantity_unit(&self, quantity_unit: &str) -> Result<(), String> {
        match self {
            Self::Standard { unit, .. } | Self::Interval { unit, .. } => {
                if unit != quantity_unit {
                    return Err("uncertainty unit must match quantity unit".to_owned());
                }
            }
            Self::Covariance { unit_squared, .. } => {
                let expected_unit_squared = if quantity_unit == "1" {
                    "1".to_owned()
                } else {
                    format!("{quantity_unit}^2")
                };
                if unit_squared != &expected_unit_squared {
                    return Err(
                        "covariance unit must match the squared quantity unit".to_owned(),
                    );
                }
            }
            Self::NotReported { .. } => {}
        }
        Ok(())
    }
}

impl QuantityValue {
    pub fn validate(&self) -> Result<(), String> {
        self.quantity_id.validate("quantity identifier")?;
        require_nonempty("quantity kind", &self.quantity_kind)?;
        require_nonempty("quantity unit", &self.unit)?;
        self.uncertainty.validate()?;
        self.uncertainty.validate_quantity_unit(&self.unit)?;
        match &self.observation {
            QuantityObservation::Observed { payload } => {
                payload.validate()?;
                match payload {
                    QuantityPayload::Vector { basis, .. }
                    | QuantityPayload::Tensor { basis, .. } => {
                        if let Some(declared_basis) = &self.tensor_component_or_basis {
                            require_nonempty("quantity basis metadata", declared_basis)?;
                            if declared_basis != basis {
                                return Err(
                                    "quantity basis metadata must match the payload basis"
                                        .to_owned(),
                                );
                            }
                        }
                    }
                    QuantityPayload::Scalar { .. } | QuantityPayload::Spectrum { .. } => {
                        if let Some(component_or_basis) = &self.tensor_component_or_basis {
                            require_nonempty(
                                "quantity tensor component or basis",
                                component_or_basis,
                            )?;
                        }
                    }
                }
                if self.conditions.is_empty() {
                    return Err("observed quantity requires conditions".to_owned());
                }
                for (condition, value) in &self.conditions {
                    require_nonempty("condition name", condition)?;
                    require_nonempty("condition value", value)?;
                }
                let applicability_range = self
                    .applicability_range
                    .as_deref()
                    .ok_or_else(|| {
                        "observed quantity requires an applicability range".to_owned()
                    })?;
                require_nonempty("applicability range", applicability_range)?;
                if let Uncertainty::Covariance { dimension, .. } = &self.uncertainty
                    && *dimension != payload.value_count()
                {
                    return Err(
                        "covariance dimension must match the observed payload".to_owned(),
                    );
                }
            }
            QuantityObservation::Missing {
                reason: Missingness::BelowDetectionLimit { upper_bound, unit },
            } => {
                require_nonempty("detection-limit unit", unit)?;
                if !upper_bound.is_finite() || *upper_bound < 0.0 {
                    return Err("detection limit must be finite and nonnegative".to_owned());
                }
                if unit != &self.unit {
                    return Err("detection-limit unit must match quantity unit".to_owned());
                }
            }
            QuantityObservation::Missing {
                reason: Missingness::Withheld { reason },
            } => require_nonempty("withholding reason", reason)?,
            QuantityObservation::Missing { .. } => {}
        }
        if matches!(self.observation, QuantityObservation::Missing { .. })
            && self.evidence.class() == EvidenceClass::ExperimentalFitted
        {
            return Err("missing values cannot be experimental fits".to_owned());
        }
        match (&self.origin, &self.evidence) {
            (QuantityOrigin::Measurement { .. }, EvidenceBasis::ExperimentalDirect) => {}
            (
                QuantityOrigin::Measurement { .. },
                EvidenceBasis::ExperimentalFitted {
                    input_quantity_ids,
                    fit_model,
                    fit_model_version,
                    residual_artifact_id,
                    parameter_covariance,
                },
            ) => {
                if input_quantity_ids.is_empty() {
                    return Err("experimental fit requires direct input quantities".to_owned());
                }
                let mut unique_input_ids = BTreeSet::new();
                for input_quantity_id in input_quantity_ids {
                    input_quantity_id.validate("fit input quantity identifier")?;
                    if !unique_input_ids.insert(input_quantity_id) {
                        return Err(format!(
                            "experimental fit repeats input quantity {}",
                            input_quantity_id.0
                        ));
                    }
                }
                require_nonempty("fit model", fit_model)?;
                require_nonempty("fit model version", fit_model_version)?;
                residual_artifact_id.validate("fit residual artifact identifier")?;
                parameter_covariance.validate()?;
                if !matches!(parameter_covariance, Uncertainty::Covariance { .. }) {
                    return Err("experimental fit requires parameter covariance".to_owned());
                }
            }
            (QuantityOrigin::ModelRun { .. }, EvidenceBasis::Computed) => {}
            (_, EvidenceBasis::InferredProxy { rationale }) => {
                require_nonempty("inferred-proxy rationale", rationale)?;
            }
            _ => {
                return Err(
                    "quantity origin does not match direct, fitted, or computed evidence class"
                        .to_owned(),
                );
            }
        }
        Ok(())
    }
}

impl Material {
    pub fn validate(&self) -> Result<(), String> {
        self.material_id.validate("material identifier")?;
        require_nonempty("material name", &self.preferred_name)?;
        require_nonempty("material formula", &self.formula)
    }
}

impl MaterialState {
    pub fn validate(&self) -> Result<(), String> {
        self.state_id.validate("state identifier")?;
        self.material_id.validate("material identifier")?;
        require_nonempty("phase identifier", &self.phase_id)?;
        for (label, value) in [
            ("state atmosphere", self.atmosphere.as_deref()),
            ("state orientation", self.orientation.as_deref()),
        ] {
            if let Some(value) = value {
                require_nonempty(label, value)?;
            }
        }
        for (name, value) in &self.applied_fields {
            require_nonempty("applied-field name", name)?;
            require_nonempty("applied-field value", value)?;
        }
        for history_entry in &self.history {
            require_nonempty("state-history entry", history_entry)?;
        }
        if self
            .temperature_k
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("temperature must be finite and nonnegative".to_owned());
        }
        if self
            .pressure_pa
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("pressure must be finite and nonnegative".to_owned());
        }
        if self
            .phase_fraction
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err("phase fraction must lie in [0, 1]".to_owned());
        }
        if self
            .time_since_processing_s
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("time since processing must be finite and nonnegative".to_owned());
        }
        if self.strain.as_ref().is_some_and(|values| {
            values.is_empty() || values.iter().any(|value| !value.is_finite())
        }) {
            return Err("strain requires finite components".to_owned());
        }
        Ok(())
    }

    fn validate_quantity_conditions(
        &self,
        quantity_id: &RecordId,
        conditions: &BTreeMap<String, String>,
    ) -> Result<(), String> {
        fn numeric_condition(
            state: &MaterialState,
            quantity_id: &RecordId,
            conditions: &BTreeMap<String, String>,
            key: &str,
            expected: Option<f64>,
        ) -> Result<(), String> {
            let Some(encoded) = conditions.get(key) else {
                return Ok(());
            };
            let observed = encoded.parse::<f64>().map_err(|_| {
                format!(
                    "quantity {} condition {key} must be a finite SI scalar",
                    quantity_id.0
                )
            })?;
            if !observed.is_finite() {
                return Err(format!(
                    "quantity {} condition {key} must be a finite SI scalar",
                    quantity_id.0
                ));
            }
            let expected = expected.ok_or_else(|| {
                format!(
                    "quantity {} condition {key} is absent from state {}",
                    quantity_id.0, state.state_id.0
                )
            })?;
            if observed != expected {
                return Err(format!(
                    "quantity {} condition {key} contradicts state {}",
                    quantity_id.0, state.state_id.0
                ));
            }
            Ok(())
        }

        for ambiguous in ["temperature", "pressure", "strain"] {
            if conditions.contains_key(ambiguous) {
                return Err(format!(
                    "quantity {} condition {ambiguous} must use an explicit SI key",
                    quantity_id.0
                ));
            }
        }
        numeric_condition(
            self,
            quantity_id,
            conditions,
            "temperature_k",
            self.temperature_k,
        )?;
        numeric_condition(
            self,
            quantity_id,
            conditions,
            "pressure_pa",
            self.pressure_pa,
        )?;
        numeric_condition(
            self,
            quantity_id,
            conditions,
            "phase_fraction",
            self.phase_fraction,
        )?;
        numeric_condition(
            self,
            quantity_id,
            conditions,
            "time_since_processing_s",
            self.time_since_processing_s,
        )?;
        for (key, expected) in [
            ("phase_id", Some(self.phase_id.as_str())),
            ("atmosphere", self.atmosphere.as_deref()),
            ("orientation", self.orientation.as_deref()),
        ] {
            if let Some(observed) = conditions.get(key)
                && Some(observed.as_str()) != expected
            {
                return Err(format!(
                    "quantity {} condition {key} contradicts state {}",
                    quantity_id.0, self.state_id.0
                ));
            }
        }
        for key in conditions.keys() {
            let Some(component) = key.strip_prefix("strain_component:") else {
                continue;
            };
            let index = component.parse::<usize>().map_err(|_| {
                format!(
                    "quantity {} condition {key} must use a zero-based integer component",
                    quantity_id.0
                )
            })?;
            let canonical_component = index.to_string();
            if component != canonical_component.as_str() {
                return Err(format!(
                    "quantity {} condition {key} must use a canonical zero-based integer component",
                    quantity_id.0
                ));
            }
            numeric_condition(
                self,
                quantity_id,
                conditions,
                key,
                self.strain
                    .as_ref()
                    .and_then(|components| components.get(index))
                    .copied(),
            )?;
        }
        for (key, observed) in conditions {
            let Some(field_name) = key.strip_prefix("applied_field:") else {
                continue;
            };
            if field_name.is_empty()
                || self.applied_fields.get(field_name) != Some(observed)
            {
                return Err(format!(
                    "quantity {} condition {key} contradicts state {}",
                    quantity_id.0, self.state_id.0
                ));
            }
        }
        Ok(())
    }
}

impl Specimen {
    pub fn validate(&self) -> Result<(), String> {
        self.specimen_id.validate("specimen identifier")?;
        self.state_id.validate("state identifier")?;
        require_nonempty("synthesis or deposition", &self.synthesis_or_deposition)?;
        require_nonempty("specimen geometry", &self.geometry)?;
        for (label, value) in [
            ("purity assay", self.purity_assay.as_deref()),
            ("composition assay", self.composition_assay.as_deref()),
            ("anneal history", self.anneal_history.as_deref()),
            ("specimen texture", self.texture.as_deref()),
        ] {
            if let Some(value) = value {
                require_nonempty(label, value)?;
            }
        }
        if let Some(substrate_specimen_id) = &self.substrate_specimen_id {
            substrate_specimen_id.validate("substrate specimen identifier")?;
        }
        for adhesion_layer in &self.adhesion_layers {
            require_nonempty("adhesion-layer entry", adhesion_layer)?;
        }
        for (label, value) in [
            ("thickness", self.thickness_m),
            ("grain size", self.grain_size_m),
            ("roughness", self.roughness_rms_m),
        ] {
            if value.is_some_and(|number| !number.is_finite() || number < 0.0) {
                return Err(format!("{label} must be finite and nonnegative"));
            }
        }
        if self
            .porosity_fraction
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err("porosity fraction must lie in [0, 1]".to_owned());
        }
        Ok(())
    }
}

impl Measurement {
    pub fn validate(&self) -> Result<(), String> {
        self.measurement_id.validate("measurement identifier")?;
        self.specimen_id.validate("specimen identifier")?;
        self.raw_artifact_id.validate("raw artifact identifier")?;
        self.processing_recipe_id
            .validate("processing recipe identifier")?;
        require_nonempty("measurement method", &self.method)?;
        require_nonempty("instrument", &self.instrument)?;
        require_nonempty("calibration or standard", &self.calibration_or_standard)?;
        require_nonempty("operator or facility", &self.operator_or_facility)?;
        if self.repeat_count == 0 {
            return Err("measurement repeat count must be positive".to_owned());
        }
        if let Some((detection_limit, unit)) = &self.detection_limit {
            require_nonempty("measurement detection-limit unit", unit)?;
            if !detection_limit.is_finite() || *detection_limit < 0.0 {
                return Err(
                    "measurement detection limit must be finite and nonnegative".to_owned(),
                );
            }
        }
        require_nonempty("source identifier", &self.provenance.source_id)?;
        require_nonempty(
            "source DOI or stable identifier",
            &self.provenance.doi_or_stable_id,
        )?;
        require_nonempty("source locator", &self.provenance.locator)?;
        require_nonempty("source license", &self.provenance.license)?;
        require_nonempty("retrieval date", &self.provenance.retrieval_date)?;
        require_nonempty("parser version", &self.provenance.parser_version)?;
        if self.provenance.transformation_lineage.is_empty()
            || self
                .provenance
                .transformation_lineage
                .iter()
                .any(|step| step.trim().is_empty())
        {
            return Err("transformation lineage requires nonempty steps".to_owned());
        }
        require_nonempty("source SHA-256", &self.provenance.source_sha256)?;
        if self.provenance.source_sha256.len() != 64
            || !self
                .provenance
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("source SHA-256 must contain 64 hexadecimal digits".to_owned());
        }
        Ok(())
    }
}

impl ModelRun {
    pub fn validate(&self) -> Result<(), String> {
        self.model_run_id.validate("model-run identifier")?;
        self.code_artifact_id.validate("code artifact identifier")?;
        self.specimen_id.validate("model-run specimen identifier")?;
        require_nonempty("model name", &self.model_name)?;
        require_nonempty("model version", &self.model_version)?;
        require_nonempty("model-run geometry", &self.geometry)?;
        require_nonempty("constitutive model", &self.constitutive_model)?;
        require_nonempty("validation status", &self.validation_status)?;
        if self.input_quantity_ids.is_empty() {
            return Err("model run requires at least one input quantity".to_owned());
        }
        let unique_input_ids: BTreeSet<_> = self.input_quantity_ids.iter().collect();
        if unique_input_ids.len() != self.input_quantity_ids.len() {
            return Err("model run repeats an input quantity".to_owned());
        }
        if self.conditions.is_empty() {
            return Err("model run requires bound conditions".to_owned());
        }
        for (name, value) in &self.conditions {
            require_nonempty("model-run condition name", name)?;
            require_nonempty("model-run condition value", value)?;
        }
        for history_entry in &self.state_history {
            require_nonempty("model-run state-history entry", history_entry)?;
        }
        if self.parameters.is_empty() {
            return Err("model run requires declared parameters".to_owned());
        }
        for (name, value) in &self.parameters {
            require_nonempty("model parameter name", name)?;
            require_nonempty("model parameter value", value)?;
        }
        if self.convergence_settings.is_empty() {
            return Err("model run requires convergence settings".to_owned());
        }
        for (name, value) in &self.convergence_settings {
            require_nonempty("convergence setting name", name)?;
            require_nonempty("convergence setting value", value)?;
        }
        Ok(())
    }
}

impl DerivedValue {
    pub fn validate(&self) -> Result<(), String> {
        self.derived_value_id.validate("derived-value identifier")?;
        self.model_run_id.validate("model-run identifier")?;
        if self.input_quantity_ids.is_empty() {
            return Err("derived value requires measured or admitted inputs".to_owned());
        }
        let unique_input_ids: BTreeSet<_> = self.input_quantity_ids.iter().collect();
        if unique_input_ids.len() != self.input_quantity_ids.len() {
            return Err("derived value repeats an input quantity".to_owned());
        }
        let QuantityOrigin::ModelRun { model_run_id } = &self.output.origin else {
            return Err("derived output must originate from a model run".to_owned());
        };
        if model_run_id != &self.model_run_id {
            return Err("derived output model-run identity mismatch".to_owned());
        }
        if !matches!(
            self.output.evidence,
            EvidenceBasis::Computed | EvidenceBasis::InferredProxy { .. }
        ) {
            return Err("derived output requires computed or inferred_proxy evidence".to_owned());
        }
        self.output.validate()?;
        self.propagated_uncertainty.validate()?;
        if self.output.uncertainty != self.propagated_uncertainty {
            return Err(
                "derived output uncertainty must equal propagated uncertainty".to_owned(),
            );
        }
        Ok(())
    }
}

impl MaterialEvidenceGraph {
    pub fn validate(&self) -> Result<(), String> {
        fn unique_ids<'a>(
            label: &str,
            ids: impl IntoIterator<Item = &'a RecordId>,
        ) -> Result<BTreeSet<&'a RecordId>, String> {
            let mut unique = BTreeSet::new();
            for id in ids {
                id.validate(label)?;
                if !unique.insert(id) {
                    return Err(format!("duplicate {label}: {}", id.0));
                }
            }
            Ok(unique)
        }

        for material in &self.materials {
            material.validate()?;
        }
        for state in &self.states {
            state.validate()?;
        }
        for specimen in &self.specimens {
            specimen.validate()?;
        }
        for measurement in &self.measurements {
            measurement.validate()?;
        }
        for quantity in &self.quantities {
            quantity.validate()?;
        }
        for model_run in &self.model_runs {
            model_run.validate()?;
        }
        for derived in &self.derived_values {
            derived.validate()?;
        }

        let material_ids = unique_ids(
            "material identifier",
            self.materials.iter().map(|record| &record.material_id),
        )?;
        let state_ids = unique_ids(
            "state identifier",
            self.states.iter().map(|record| &record.state_id),
        )?;
        let states_by_id: BTreeMap<_, _> = self
            .states
            .iter()
            .map(|state| (&state.state_id, state))
            .collect();
        let specimen_ids = unique_ids(
            "specimen identifier",
            self.specimens.iter().map(|record| &record.specimen_id),
        )?;
        let specimens_by_id: BTreeMap<_, _> = self
            .specimens
            .iter()
            .map(|specimen| (&specimen.specimen_id, specimen))
            .collect();
        unique_ids(
            "measurement identifier",
            self.measurements
                .iter()
                .map(|record| &record.measurement_id),
        )?;
        let measurements_by_id: BTreeMap<_, _> = self
            .measurements
            .iter()
            .map(|measurement| (&measurement.measurement_id, measurement))
            .collect();
        unique_ids(
            "quantity identifier",
            self.quantities
                .iter()
                .map(|record| &record.quantity_id)
                .chain(
                    self.derived_values
                        .iter()
                        .map(|record| &record.output.quantity_id),
                ),
        )?;
        let quantities_by_id: BTreeMap<_, _> = self
            .quantities
            .iter()
            .map(|quantity| (&quantity.quantity_id, quantity))
            .collect();
        unique_ids(
            "model-run identifier",
            self.model_runs.iter().map(|record| &record.model_run_id),
        )?;
        let model_runs_by_id: BTreeMap<_, _> = self
            .model_runs
            .iter()
            .map(|model_run| (&model_run.model_run_id, model_run))
            .collect();
        unique_ids(
            "derived-value identifier",
            self.derived_values
                .iter()
                .map(|record| &record.derived_value_id),
        )?;

        for state in &self.states {
            if !material_ids.contains(&state.material_id) {
                return Err(format!(
                    "state {} references unknown material {}",
                    state.state_id.0, state.material_id.0
                ));
            }
        }
        for specimen in &self.specimens {
            let mut substrate_chain = BTreeSet::new();
            let mut current_id = Some(&specimen.specimen_id);
            while let Some(specimen_id) = current_id {
                if !substrate_chain.insert(specimen_id) {
                    return Err(format!(
                        "specimen substrate references contain a cycle at {}",
                        specimen_id.0
                    ));
                }
                current_id = specimens_by_id
                    .get(specimen_id)
                    .and_then(|record| record.substrate_specimen_id.as_ref());
            }
        }
        for specimen in &self.specimens {
            if !state_ids.contains(&specimen.state_id) {
                return Err(format!(
                    "specimen {} references unknown state {}",
                    specimen.specimen_id.0, specimen.state_id.0
                ));
            }
            if let Some(substrate_id) = &specimen.substrate_specimen_id
                && !specimen_ids.contains(substrate_id)
            {
                return Err(format!(
                    "specimen {} references unknown substrate {}",
                    specimen.specimen_id.0, substrate_id.0
                ));
            }
        }
        for measurement in &self.measurements {
            if !specimen_ids.contains(&measurement.specimen_id) {
                return Err(format!(
                    "measurement {} references unknown specimen {}",
                    measurement.measurement_id.0, measurement.specimen_id.0
                ));
            }
        }
        for quantity in &self.quantities {
            match &quantity.origin {
                QuantityOrigin::Measurement { measurement_id } => {
                    let measurement = measurements_by_id.get(measurement_id).ok_or_else(|| {
                        format!(
                            "quantity {} references unknown measurement {}",
                            quantity.quantity_id.0, measurement_id.0
                        )
                    })?;
                    let specimen = specimens_by_id
                        .get(&measurement.specimen_id)
                        .ok_or_else(|| {
                            format!(
                                "measurement {} references unknown specimen {}",
                                measurement.measurement_id.0, measurement.specimen_id.0
                            )
                        })?;
                    let state = states_by_id.get(&specimen.state_id).ok_or_else(|| {
                        format!(
                            "specimen {} references unknown state {}",
                            specimen.specimen_id.0, specimen.state_id.0
                        )
                    })?;
                    state.validate_quantity_conditions(
                        &quantity.quantity_id,
                        &quantity.conditions,
                    )?;
                    if let QuantityObservation::Missing {
                        reason: Missingness::BelowDetectionLimit { upper_bound, unit },
                    } = &quantity.observation
                    {
                        let (measurement_limit, measurement_unit) = measurement
                            .detection_limit
                            .as_ref()
                            .ok_or_else(|| {
                                format!(
                                    "quantity {} claims a below-detection limit but measurement {} has no detection limit",
                                    quantity.quantity_id.0, measurement_id.0
                                )
                            })?;
                        if upper_bound.to_bits() != measurement_limit.to_bits()
                            || unit != measurement_unit
                        {
                            return Err(format!(
                                "quantity {} below-detection limit does not match measurement {}",
                                quantity.quantity_id.0, measurement_id.0
                            ));
                        }
                    }
                }
                QuantityOrigin::ModelRun { model_run_id } => {
                    return Err(format!(
                        "quantity {} from model run {} must be wrapped in a derived value",
                        quantity.quantity_id.0, model_run_id.0
                    ));
                }
            }
            if let EvidenceBasis::ExperimentalFitted {
                input_quantity_ids, ..
            } = &quantity.evidence
            {
                for input_quantity_id in input_quantity_ids {
                    let input_quantity =
                        quantities_by_id.get(input_quantity_id).ok_or_else(|| {
                            format!(
                                "fitted quantity {} references unknown input quantity {}",
                                quantity.quantity_id.0, input_quantity_id.0
                            )
                        })?;
                    if input_quantity.evidence.class() != EvidenceClass::ExperimentalDirect {
                        return Err(format!(
                            "fitted quantity {} input {} is not experimental_direct",
                            quantity.quantity_id.0, input_quantity_id.0
                        ));
                    }
                    if !matches!(
                        &input_quantity.observation,
                        QuantityObservation::Observed { .. }
                    ) {
                        return Err(format!(
                            "fitted quantity {} input {} is not an observed quantity",
                            quantity.quantity_id.0, input_quantity_id.0
                        ));
                    }
                }
            }
        }
        for model_run in &self.model_runs {
            let specimen = specimens_by_id.get(&model_run.specimen_id).ok_or_else(|| {
                format!(
                    "model run {} references unknown specimen {}",
                    model_run.model_run_id.0, model_run.specimen_id.0
                )
            })?;
            if model_run.geometry != specimen.geometry {
                return Err(format!(
                    "model run {} geometry does not match specimen {}",
                    model_run.model_run_id.0, specimen.specimen_id.0
                ));
            }
            let state = states_by_id.get(&specimen.state_id).ok_or_else(|| {
                format!(
                    "model run {} specimen {} references unknown state {}",
                    model_run.model_run_id.0, specimen.specimen_id.0, specimen.state_id.0
                )
            })?;
            if model_run.state_history != state.history {
                return Err(format!(
                    "model run {} state history does not match state {}",
                    model_run.model_run_id.0, state.state_id.0
                ));
            }
            for input_quantity_id in &model_run.input_quantity_ids {
                let input_quantity = quantities_by_id.get(input_quantity_id).ok_or_else(|| {
                    format!(
                        "model run {} references unknown input quantity {}",
                        model_run.model_run_id.0, input_quantity_id.0
                    )
                })?;
                if !matches!(
                    &input_quantity.observation,
                    QuantityObservation::Observed { .. }
                )
                    || !matches!(
                        input_quantity.evidence.class(),
                        EvidenceClass::ExperimentalDirect | EvidenceClass::ExperimentalFitted
                    )
                {
                    return Err(format!(
                        "model run {} input {} is not observed experimental evidence",
                        model_run.model_run_id.0, input_quantity_id.0
                    ));
                }
                let QuantityOrigin::Measurement { measurement_id } = &input_quantity.origin else {
                    return Err(format!(
                        "model run {} input {} does not originate from a measurement",
                        model_run.model_run_id.0, input_quantity_id.0
                    ));
                };
                let measurement = measurements_by_id.get(measurement_id).ok_or_else(|| {
                    format!(
                        "model run {} input {} references unknown measurement {}",
                        model_run.model_run_id.0, input_quantity_id.0, measurement_id.0
                    )
                })?;
                if measurement.specimen_id != model_run.specimen_id {
                    return Err(format!(
                        "model run {} specimen does not match input {} specimen",
                        model_run.model_run_id.0, input_quantity_id.0
                    ));
                }
                if input_quantity.conditions != model_run.conditions {
                    return Err(format!(
                        "model run {} conditions do not match input {} conditions",
                        model_run.model_run_id.0, input_quantity_id.0
                    ));
                }
            }
        }
        for derived in &self.derived_values {
            let model_run = model_runs_by_id.get(&derived.model_run_id).ok_or_else(|| {
                format!(
                    "derived value {} references unknown model run {}",
                    derived.derived_value_id.0, derived.model_run_id.0
                )
            })?;
            if derived.output.conditions != model_run.conditions {
                return Err(format!(
                    "derived value {} output conditions do not match model run {} conditions",
                    derived.derived_value_id.0, model_run.model_run_id.0
                ));
            }
            let declared_inputs: BTreeSet<_> = model_run.input_quantity_ids.iter().collect();
            let derived_inputs: BTreeSet<_> = derived.input_quantity_ids.iter().collect();
            if derived_inputs != declared_inputs {
                return Err(format!(
                    "derived value {} inputs do not equal model run {} inputs",
                    derived.derived_value_id.0, model_run.model_run_id.0
                ));
            }
            for input_quantity_id in &derived.input_quantity_ids {
                let input_quantity = quantities_by_id.get(input_quantity_id).ok_or_else(|| {
                    format!(
                        "derived value {} references unknown input quantity {}",
                        derived.derived_value_id.0, input_quantity_id.0
                    )
                })?;
                if !matches!(
                    &input_quantity.observation,
                    QuantityObservation::Observed { .. }
                )
                    || !matches!(
                        input_quantity.evidence.class(),
                        EvidenceClass::ExperimentalDirect | EvidenceClass::ExperimentalFitted
                    )
                {
                    return Err(format!(
                        "derived value {} input {} is not observed experimental evidence",
                        derived.derived_value_id.0, input_quantity_id.0
                    ));
                }
                if !declared_inputs.contains(input_quantity_id) {
                    return Err(format!(
                        "derived value {} input {} is not declared by model run {}",
                        derived.derived_value_id.0, input_quantity_id.0, model_run.model_run_id.0
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identifier(value: &str) -> RecordId {
        RecordId::new(value).unwrap()
    }

    #[test]
    fn one_formula_supports_distinct_gold_specimens() {
        let material = Material {
            material_id: identifier("material:au"),
            preferred_name: "Gold".to_owned(),
            formula: "Au".to_owned(),
        };
        let state = MaterialState {
            state_id: identifier("state:au:300k"),
            material_id: material.material_id,
            phase_id: "fcc".to_owned(),
            temperature_k: Some(300.0),
            pressure_pa: Some(101_325.0),
            atmosphere: Some("vacuum deposition chamber".to_owned()),
            phase_fraction: Some(1.0),
            orientation: None,
            strain: None,
            applied_fields: BTreeMap::new(),
            time_since_processing_s: None,
            history: Vec::new(),
        };
        let evaporated = Specimen {
            specimen_id: identifier("specimen:au:evaporated"),
            state_id: state.state_id.clone(),
            purity_assay: Some("source assay required".to_owned()),
            composition_assay: None,
            synthesis_or_deposition: "thermal evaporation".to_owned(),
            anneal_history: None,
            thickness_m: Some(200e-9),
            substrate_specimen_id: Some(identifier("specimen:glass")),
            adhesion_layers: Vec::new(),
            grain_size_m: None,
            texture: None,
            porosity_fraction: None,
            roughness_rms_m: None,
            geometry: "planar film".to_owned(),
        };
        let template_stripped = Specimen {
            specimen_id: identifier("specimen:au:template-stripped"),
            synthesis_or_deposition: "thermal evaporation followed by template stripping"
                .to_owned(),
            ..evaporated.clone()
        };
        assert_ne!(evaporated.specimen_id, template_stripped.specimen_id);
        assert_eq!(evaporated.state_id, template_stripped.state_id);
    }

    #[test]
    fn missingness_is_not_a_numeric_zero() {
        let missing = QuantityObservation::Missing {
            reason: Missingness::NotMeasured,
        };
        let zero = QuantityObservation::Observed {
            payload: QuantityPayload::Scalar { value: 0.0 },
        };
        assert_ne!(missing, zero);
    }

    fn scalar_quantity(origin: QuantityOrigin, evidence: EvidenceBasis) -> QuantityValue {
        QuantityValue {
            quantity_id: identifier("quantity:test"),
            origin,
            family: PropertyFamily::OpticalPhotonic,
            quantity_kind: "refractive_index_real".to_owned(),
            unit: "1".to_owned(),
            tensor_component_or_basis: None,
            observation: QuantityObservation::Observed {
                payload: QuantityPayload::Scalar { value: 0.82 },
            },
            uncertainty: Uncertainty::Standard {
                value: 0.01,
                unit: "1".to_owned(),
            },
            conditions: BTreeMap::from([("wavelength".to_owned(), "0.500 um".to_owned())]),
            applicability_range: Some("single reported wavelength".to_owned()),
            evidence,
        }
    }

    #[test]
    fn evidence_class_requires_matching_origin_and_fit_artifacts() {
        let direct = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        assert!(direct.validate().is_ok());

        let invalid_computed = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::Computed,
        );
        assert!(invalid_computed.validate().is_err());

        let invalid_fit = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalFitted {
                input_quantity_ids: vec![identifier("quantity:psi-delta")],
                fit_model: "Drude-Lorentz".to_owned(),
                fit_model_version: "1".to_owned(),
                residual_artifact_id: identifier("artifact:fit-residuals"),
                parameter_covariance: Uncertainty::Standard {
                    value: 0.01,
                    unit: "1".to_owned(),
                },
            },
        );
        assert!(invalid_fit.validate().is_err());
    }

    #[test]
    fn covariance_requires_symmetry_and_positive_semidefiniteness() {
        let valid_semidefinite = Uncertainty::Covariance {
            dimension: 2,
            values_row_major: vec![1.0, 1.0, 1.0, 1.0],
            unit_squared: "1".to_owned(),
        };
        assert!(valid_semidefinite.validate().is_ok());

        let asymmetric = Uncertainty::Covariance {
            dimension: 2,
            values_row_major: vec![1.0, 0.5, 0.25, 1.0],
            unit_squared: "1".to_owned(),
        };
        assert_eq!(
            asymmetric.validate().unwrap_err(),
            "covariance matrix must be symmetric"
        );

        let indefinite = Uncertainty::Covariance {
            dimension: 2,
            values_row_major: vec![1.0, 2.0, 2.0, 1.0],
            unit_squared: "1".to_owned(),
        };
        assert_eq!(
            indefinite.validate().unwrap_err(),
            "covariance matrix must be positive semidefinite"
        );

        let small_negative_variance = Uncertainty::Covariance {
            dimension: 1,
            values_row_major: vec![-1e-20],
            unit_squared: "m^2".to_owned(),
        };
        assert_eq!(
            small_negative_variance.validate().unwrap_err(),
            "covariance matrix must be positive semidefinite"
        );

        let small_asymmetry = Uncertainty::Covariance {
            dimension: 2,
            values_row_major: vec![1e-20, 1e-20, 0.0, 1e-20],
            unit_squared: "m^2".to_owned(),
        };
        assert_eq!(
            small_asymmetry.validate().unwrap_err(),
            "covariance matrix must be symmetric"
        );
    }

    #[test]
    fn uncertainty_units_match_quantity_units() {
        let origin = QuantityOrigin::Measurement {
            measurement_id: identifier("measurement:ellipsometry"),
        };
        let mut quantity = scalar_quantity(origin.clone(), EvidenceBasis::ExperimentalDirect);
        quantity.unit = "Pa".to_owned();
        quantity.uncertainty = Uncertainty::Standard {
            value: 0.01,
            unit: "K".to_owned(),
        };
        assert_eq!(
            quantity.validate().unwrap_err(),
            "uncertainty unit must match quantity unit"
        );

        let mut quantity = scalar_quantity(origin.clone(), EvidenceBasis::ExperimentalDirect);
        quantity.unit = "Pa".to_owned();
        quantity.uncertainty = Uncertainty::Interval {
            lower: 0.0,
            upper: 0.02,
            unit: "K".to_owned(),
        };
        assert_eq!(
            quantity.validate().unwrap_err(),
            "uncertainty unit must match quantity unit"
        );

        let mut quantity = scalar_quantity(origin, EvidenceBasis::ExperimentalDirect);
        quantity.unit = "Pa".to_owned();
        quantity.uncertainty = Uncertainty::Covariance {
            dimension: 1,
            values_row_major: vec![0.01],
            unit_squared: "Pa^2".to_owned(),
        };
        assert!(quantity.validate().is_ok());

        let Uncertainty::Covariance { unit_squared, .. } = &mut quantity.uncertainty else {
            unreachable!()
        };
        *unit_squared = "K^2".to_owned();
        assert_eq!(
            quantity.validate().unwrap_err(),
            "covariance unit must match the squared quantity unit"
        );
    }

    #[test]
    fn tensor_and_covariance_dimension_products_reject_overflow() {
        let tensor = QuantityPayload::Tensor {
            values_row_major: Vec::new(),
            rows: usize::MAX,
            columns: 2,
            basis: "Cartesian".to_owned(),
        };
        assert_eq!(
            tensor.validate().unwrap_err(),
            "tensor dimensions must match finite values"
        );

        let covariance = Uncertainty::Covariance {
            dimension: usize::MAX,
            values_row_major: Vec::new(),
            unit_squared: "Pa^2".to_owned(),
        };
        assert_eq!(
            covariance.validate().unwrap_err(),
            "covariance dimensions must match finite values"
        );
    }

    #[test]
    fn observed_quantities_require_conditions_applicability_and_matching_covariance() {
        let origin = QuantityOrigin::Measurement {
            measurement_id: identifier("measurement:ellipsometry"),
        };
        let mut quantity = scalar_quantity(origin.clone(), EvidenceBasis::ExperimentalDirect);
        quantity.conditions.clear();
        assert_eq!(
            quantity.validate().unwrap_err(),
            "observed quantity requires conditions"
        );

        let mut quantity = scalar_quantity(origin.clone(), EvidenceBasis::ExperimentalDirect);
        quantity.applicability_range = None;
        assert_eq!(
            quantity.validate().unwrap_err(),
            "observed quantity requires an applicability range"
        );

        let mut quantity = scalar_quantity(origin, EvidenceBasis::ExperimentalDirect);
        quantity.observation = QuantityObservation::Observed {
            payload: QuantityPayload::Vector {
                values: vec![0.1, 0.2, 0.3],
                basis: "Cartesian".to_owned(),
            },
        };
        quantity.tensor_component_or_basis = Some("Cartesian".to_owned());
        quantity.uncertainty = Uncertainty::Covariance {
            dimension: 2,
            values_row_major: vec![1.0, 0.0, 0.0, 1.0],
            unit_squared: "1".to_owned(),
        };
        assert_eq!(
            quantity.validate().unwrap_err(),
            "covariance dimension must match the observed payload"
        );
    }

    #[test]
    fn below_detection_limit_requires_a_nonnegative_bound() {
        let mut quantity = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        quantity.observation = QuantityObservation::Missing {
            reason: Missingness::BelowDetectionLimit {
                upper_bound: -1e-9,
                unit: "m".to_owned(),
            },
        };
        assert_eq!(
            quantity.validate().unwrap_err(),
            "detection limit must be finite and nonnegative"
        );
    }

    #[test]
    fn vector_and_tensor_basis_metadata_has_one_consistent_identity() {
        let origin = QuantityOrigin::Measurement {
            measurement_id: identifier("measurement:ellipsometry"),
        };
        let mut quantity = scalar_quantity(origin, EvidenceBasis::ExperimentalDirect);
        quantity.observation = QuantityObservation::Observed {
            payload: QuantityPayload::Vector {
                values: vec![0.1, 0.2, 0.3],
                basis: "Cartesian".to_owned(),
            },
        };

        assert!(quantity.validate().is_ok());
        quantity.tensor_component_or_basis = Some(" ".to_owned());
        assert_eq!(
            quantity.validate().unwrap_err(),
            "quantity basis metadata must be nonempty"
        );
        quantity.tensor_component_or_basis = Some("cylindrical".to_owned());
        assert_eq!(
            quantity.validate().unwrap_err(),
            "quantity basis metadata must match the payload basis"
        );
        quantity.tensor_component_or_basis = Some("Cartesian".to_owned());
        assert!(quantity.validate().is_ok());

        quantity.observation = QuantityObservation::Observed {
            payload: QuantityPayload::Tensor {
                values_row_major: vec![1.0, 0.0, 0.0, 1.0],
                rows: 2,
                columns: 2,
                basis: "crystallographic".to_owned(),
            },
        };
        quantity.tensor_component_or_basis = Some("Cartesian".to_owned());
        assert_eq!(
            quantity.validate().unwrap_err(),
            "quantity basis metadata must match the payload basis"
        );
        quantity.tensor_component_or_basis = Some("crystallographic".to_owned());
        assert!(quantity.validate().is_ok());
    }

    #[test]
    fn fitted_evidence_requires_valid_identifiers() {
        let fitted = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalFitted {
                input_quantity_ids: vec![identifier("quantity:psi-delta")],
                fit_model: "Drude-Lorentz".to_owned(),
                fit_model_version: "1".to_owned(),
                residual_artifact_id: RecordId(" ".to_owned()),
                parameter_covariance: Uncertainty::Covariance {
                    dimension: 1,
                    values_row_major: vec![0.01],
                    unit_squared: "1".to_owned(),
                },
            },
        );
        assert_eq!(
            fitted.validate().unwrap_err(),
            "fit residual artifact identifier must be nonempty"
        );
    }

    fn graph_with_quantities(quantities: Vec<QuantityValue>) -> MaterialEvidenceGraph {
        MaterialEvidenceGraph {
            materials: vec![Material {
                material_id: identifier("material:au"),
                preferred_name: "Gold".to_owned(),
                formula: "Au".to_owned(),
            }],
            states: vec![MaterialState {
                state_id: identifier("state:au:test"),
                material_id: identifier("material:au"),
                phase_id: "solid".to_owned(),
                temperature_k: Some(300.0),
                pressure_pa: None,
                atmosphere: None,
                phase_fraction: None,
                orientation: None,
                strain: None,
                applied_fields: BTreeMap::new(),
                time_since_processing_s: None,
                history: Vec::new(),
            }],
            specimens: vec![Specimen {
                specimen_id: identifier("specimen:au:test"),
                state_id: identifier("state:au:test"),
                purity_assay: None,
                composition_assay: None,
                synthesis_or_deposition: "reported preparation".to_owned(),
                anneal_history: None,
                thickness_m: None,
                substrate_specimen_id: None,
                adhesion_layers: Vec::new(),
                grain_size_m: None,
                texture: None,
                porosity_fraction: None,
                roughness_rms_m: None,
                geometry: "reported geometry".to_owned(),
            }],
            measurements: vec![Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
                specimen_id: identifier("specimen:au:test"),
                method: "ellipsometry".to_owned(),
                instrument: "reported instrument".to_owned(),
                calibration_or_standard: "reported calibration".to_owned(),
                operator_or_facility: "reported facility".to_owned(),
                raw_artifact_id: identifier("artifact:raw"),
                processing_recipe_id: identifier("recipe:fit"),
                repeat_count: 1,
                detection_limit: None,
                provenance: Provenance {
                    source_id: "source:test".to_owned(),
                    doi_or_stable_id: "doi:10.example/test".to_owned(),
                    locator: "table".to_owned(),
                    license: "reported terms".to_owned(),
                    retrieval_date: "2026-09-12".to_owned(),
                    source_sha256: "0".repeat(64),
                    parser_version: "fixture-v1".to_owned(),
                    transformation_lineage: vec!["source to fixture".to_owned()],
                },
            }],
            quantities,
            model_runs: Vec::new(),
            derived_values: Vec::new(),
        }
    }

    fn model_run(model_run_id: &str, input_quantity_ids: Vec<RecordId>) -> ModelRun {
        ModelRun {
            model_run_id: identifier(model_run_id),
            model_name: "Drude-Lorentz".to_owned(),
            model_version: "1".to_owned(),
            code_artifact_id: identifier("artifact:model-code"),
            input_quantity_ids,
            specimen_id: identifier("specimen:au:test"),
            geometry: "reported geometry".to_owned(),
            constitutive_model: "Drude-Lorentz permittivity".to_owned(),
            conditions: BTreeMap::from([(
                "wavelength".to_owned(),
                "0.500 um".to_owned(),
            )]),
            state_history: Vec::new(),
            parameters: BTreeMap::from([("oscillators".to_owned(), "3".to_owned())]),
            convergence_settings: BTreeMap::from([(
                "relative_tolerance".to_owned(),
                "1e-10".to_owned(),
            )]),
            validation_status: "fixture validated".to_owned(),
        }
    }

    fn graph_with_model_and_derived() -> MaterialEvidenceGraph {
        let mut input = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        input.quantity_id = identifier("quantity:epsilon");
        let mut graph = graph_with_quantities(vec![input.clone()]);
        graph.model_runs.push(model_run(
            "model:drude-lorentz:v1",
            vec![input.quantity_id.clone()],
        ));
        graph.derived_values.push(derived_value(
            "derived:reflectivity",
            "model:drude-lorentz:v1",
            vec![input.quantity_id],
            "quantity:reflectivity",
        ));
        graph
    }

    fn derived_value(
        derived_value_id: &str,
        model_run_id: &str,
        input_quantity_ids: Vec<RecordId>,
        output_quantity_id: &str,
    ) -> DerivedValue {
        let model_run_id = identifier(model_run_id);
        let mut output = scalar_quantity(
            QuantityOrigin::ModelRun {
                model_run_id: model_run_id.clone(),
            },
            EvidenceBasis::Computed,
        );
        output.quantity_id = identifier(output_quantity_id);
        DerivedValue {
            derived_value_id: identifier(derived_value_id),
            model_run_id,
            input_quantity_ids,
            output,
            propagated_uncertainty: Uncertainty::Standard {
                value: 0.01,
                unit: "1".to_owned(),
            },
        }
    }

    #[test]
    fn derived_output_accepts_matching_propagated_uncertainty() {
        let derived = derived_value(
            "derived:reflectivity",
            "model:drude-lorentz:v1",
            vec![identifier("quantity:epsilon")],
            "quantity:reflectivity",
        );

        assert!(derived.validate().is_ok());
    }

    #[test]
    fn derived_output_rejects_mismatched_propagated_uncertainty() {
        let mut derived = derived_value(
            "derived:reflectivity",
            "model:drude-lorentz:v1",
            vec![identifier("quantity:epsilon")],
            "quantity:reflectivity",
        );
        derived.propagated_uncertainty = Uncertainty::Standard {
            value: 0.02,
            unit: "1".to_owned(),
        };

        assert_eq!(
            derived.validate().unwrap_err(),
            "derived output uncertainty must equal propagated uncertainty"
        );
    }

    #[test]
    fn measurement_detection_limit_requires_finite_nonnegative_value_and_unit() {
        let mut graph = graph_with_quantities(Vec::new());
        let measurement = &mut graph.measurements[0];

        measurement.detection_limit = Some((f64::NAN, "m".to_owned()));
        assert_eq!(
            measurement.validate().unwrap_err(),
            "measurement detection limit must be finite and nonnegative"
        );

        measurement.detection_limit = Some((-1.0, "m".to_owned()));
        assert_eq!(
            measurement.validate().unwrap_err(),
            "measurement detection limit must be finite and nonnegative"
        );

        measurement.detection_limit = Some((0.0, " ".to_owned()));
        assert_eq!(
            measurement.validate().unwrap_err(),
            "measurement detection-limit unit must be nonempty"
        );
    }

    #[test]
    fn material_state_rejects_blank_optional_and_collection_metadata() {
        let state = graph_with_quantities(Vec::new())
            .states
            .into_iter()
            .next()
            .unwrap();

        let mut blank_atmosphere = state.clone();
        blank_atmosphere.atmosphere = Some(" ".to_owned());
        assert_eq!(
            blank_atmosphere.validate().unwrap_err(),
            "state atmosphere must be nonempty"
        );

        let mut blank_orientation = state.clone();
        blank_orientation.orientation = Some(String::new());
        assert_eq!(
            blank_orientation.validate().unwrap_err(),
            "state orientation must be nonempty"
        );

        let mut blank_field_name = state.clone();
        blank_field_name
            .applied_fields
            .insert(" ".to_owned(), "1 T".to_owned());
        assert_eq!(
            blank_field_name.validate().unwrap_err(),
            "applied-field name must be nonempty"
        );

        let mut blank_field_value = state.clone();
        blank_field_value
            .applied_fields
            .insert("magnetic_flux_density".to_owned(), String::new());
        assert_eq!(
            blank_field_value.validate().unwrap_err(),
            "applied-field value must be nonempty"
        );

        let mut blank_history = state;
        blank_history.history.push(" ".to_owned());
        assert_eq!(
            blank_history.validate().unwrap_err(),
            "state-history entry must be nonempty"
        );
    }

    #[test]
    fn specimen_rejects_blank_optional_and_collection_metadata() {
        let specimen = graph_with_quantities(Vec::new())
            .specimens
            .into_iter()
            .next()
            .unwrap();

        for (label, error) in [
            ("purity", "purity assay must be nonempty"),
            ("composition", "composition assay must be nonempty"),
            ("anneal", "anneal history must be nonempty"),
            ("texture", "specimen texture must be nonempty"),
        ] {
            let mut blank = specimen.clone();
            match label {
                "purity" => blank.purity_assay = Some(" ".to_owned()),
                "composition" => blank.composition_assay = Some(" ".to_owned()),
                "anneal" => blank.anneal_history = Some(" ".to_owned()),
                "texture" => blank.texture = Some(" ".to_owned()),
                _ => unreachable!(),
            }
            assert_eq!(blank.validate().unwrap_err(), error);
        }

        let mut blank_substrate = specimen.clone();
        blank_substrate.substrate_specimen_id = Some(RecordId(" ".to_owned()));
        assert_eq!(
            blank_substrate.validate().unwrap_err(),
            "substrate specimen identifier must be nonempty"
        );

        let mut blank_adhesion_layer = specimen;
        blank_adhesion_layer.adhesion_layers.push(String::new());
        assert_eq!(
            blank_adhesion_layer.validate().unwrap_err(),
            "adhesion-layer entry must be nonempty"
        );
    }

    #[test]
    fn below_detection_limit_matches_the_originating_measurement() {
        let mut quantity = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        quantity.quantity_id = identifier("quantity:below-detection");
        quantity.unit = "m".to_owned();
        quantity.uncertainty = Uncertainty::Standard {
            value: 0.0,
            unit: "m".to_owned(),
        };
        quantity.observation = QuantityObservation::Missing {
            reason: Missingness::BelowDetectionLimit {
                upper_bound: 1e-9,
                unit: "m".to_owned(),
            },
        };
        let mut graph = graph_with_quantities(vec![quantity]);

        assert_eq!(
            graph.validate().unwrap_err(),
            "quantity quantity:below-detection claims a below-detection limit but measurement measurement:ellipsometry has no detection limit"
        );

        graph.measurements[0].detection_limit = Some((2e-9, "m".to_owned()));
        assert_eq!(
            graph.validate().unwrap_err(),
            "quantity quantity:below-detection below-detection limit does not match measurement measurement:ellipsometry"
        );

        graph.measurements[0].detection_limit = Some((1e-9, "Pa".to_owned()));
        assert_eq!(
            graph.validate().unwrap_err(),
            "quantity quantity:below-detection below-detection limit does not match measurement measurement:ellipsometry"
        );

        graph.measurements[0].detection_limit = Some((1e-9, "m".to_owned()));
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn below_detection_limit_unit_matches_the_quantity_unit() {
        let mut quantity = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        quantity.observation = QuantityObservation::Missing {
            reason: Missingness::BelowDetectionLimit {
                upper_bound: 1.0,
                unit: "Pa".to_owned(),
            },
        };

        assert_eq!(
            quantity.validate().unwrap_err(),
            "detection-limit unit must match quantity unit"
        );
    }

    #[test]
    fn graph_requires_observed_experimental_model_inputs() {
        let mut inferred = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::InferredProxy {
                rationale: "proxy fixture".to_owned(),
            },
        );
        inferred.quantity_id = identifier("quantity:proxy");
        let mut graph = graph_with_quantities(vec![inferred.clone()]);
        graph.model_runs.push(model_run(
            "model:drude-lorentz:v1",
            vec![inferred.quantity_id.clone()],
        ));
        assert_eq!(
            graph.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 input quantity:proxy is not observed experimental evidence"
        );

        inferred.evidence = EvidenceBasis::ExperimentalDirect;
        inferred.observation = QuantityObservation::Missing {
            reason: Missingness::NotMeasured,
        };
        graph.quantities[0] = inferred;
        assert_eq!(
            graph.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 input quantity:proxy is not observed experimental evidence"
        );
    }

    #[test]
    fn derived_outputs_have_globally_unique_quantity_ids() {
        let mut duplicate_derived = graph_with_model_and_derived();
        duplicate_derived.derived_values.push(derived_value(
            "derived:reflectivity:alternate",
            "model:drude-lorentz:v1",
            vec![identifier("quantity:epsilon")],
            "quantity:reflectivity",
        ));
        assert_eq!(
            duplicate_derived.validate().unwrap_err(),
            "duplicate quantity identifier: quantity:reflectivity"
        );

        let mut direct = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        direct.quantity_id = identifier("quantity:input");
        let mut graph = graph_with_quantities(vec![direct.clone()]);
        graph.model_runs.push(model_run(
            "model:drude-lorentz:v1",
            vec![direct.quantity_id.clone()],
        ));
        graph.derived_values.push(derived_value(
            "derived:reflectivity",
            "model:drude-lorentz:v1",
            vec![direct.quantity_id.clone()],
            "quantity:input",
        ));
        assert_eq!(
            graph.validate().unwrap_err(),
            "duplicate quantity identifier: quantity:input"
        );
    }

    #[test]
    fn top_level_model_outputs_must_be_derived_values() {
        let mut graph = graph_with_model_and_derived();
        let model_output = graph.derived_values.remove(0).output;
        graph.quantities.push(model_output);

        assert_eq!(
            graph.validate().unwrap_err(),
            "quantity quantity:reflectivity from model run model:drude-lorentz:v1 must be wrapped in a derived value"
        );
    }

    #[test]
    fn derived_inputs_must_be_declared_by_the_linked_model_run() {
        let mut declared = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        declared.quantity_id = identifier("quantity:declared");
        let mut undeclared = declared.clone();
        undeclared.quantity_id = identifier("quantity:undeclared");
        let mut graph = graph_with_quantities(vec![declared.clone(), undeclared.clone()]);
        graph.model_runs.push(model_run(
            "model:drude-lorentz:v1",
            vec![declared.quantity_id.clone()],
        ));
        graph.derived_values.push(derived_value(
            "derived:reflectivity",
            "model:drude-lorentz:v1",
            vec![declared.quantity_id.clone()],
            "quantity:reflectivity",
        ));
        assert!(graph.validate().is_ok());

        graph.derived_values[0].input_quantity_ids[0] = undeclared.quantity_id.clone();
        assert_eq!(
            graph.validate().unwrap_err(),
            "derived value derived:reflectivity inputs do not equal model run model:drude-lorentz:v1 inputs"
        );
    }

    #[test]
    fn derived_inputs_equal_the_complete_model_input_set() {
        let mut graph = graph_with_model_and_derived();
        let mut second_input = graph.quantities[0].clone();
        second_input.quantity_id = identifier("quantity:temperature");
        graph.quantities.push(second_input.clone());
        graph.model_runs[0]
            .input_quantity_ids
            .push(second_input.quantity_id);

        assert_eq!(
            graph.validate().unwrap_err(),
            "derived value derived:reflectivity inputs do not equal model run model:drude-lorentz:v1 inputs"
        );
    }

    #[test]
    fn model_and_derived_inputs_reject_duplicates() {
        let mut graph = graph_with_model_and_derived();
        let input_id = graph.model_runs[0].input_quantity_ids[0].clone();
        graph.model_runs[0].input_quantity_ids.push(input_id);
        assert_eq!(
            graph.validate().unwrap_err(),
            "model run repeats an input quantity"
        );

        let mut graph = graph_with_model_and_derived();
        let input_id = graph.derived_values[0].input_quantity_ids[0].clone();
        graph.derived_values[0].input_quantity_ids.push(input_id);
        assert_eq!(
            graph.validate().unwrap_err(),
            "derived value repeats an input quantity"
        );
    }

    #[test]
    fn fitted_inputs_resolve_to_direct_graph_quantities() {
        let mut direct = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        direct.quantity_id = identifier("quantity:psi-delta");
        let mut fitted = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalFitted {
                input_quantity_ids: vec![direct.quantity_id.clone()],
                fit_model: "Drude-Lorentz".to_owned(),
                fit_model_version: "1".to_owned(),
                residual_artifact_id: identifier("artifact:fit-residuals"),
                parameter_covariance: Uncertainty::Covariance {
                    dimension: 1,
                    values_row_major: vec![0.01],
                    unit_squared: "1".to_owned(),
                },
            },
        );
        fitted.quantity_id = identifier("quantity:fitted-n");
        assert!(
            graph_with_quantities(vec![direct.clone(), fitted.clone()])
                .validate()
                .is_ok()
        );

        let mut missing_direct = direct.clone();
        missing_direct.observation = QuantityObservation::Missing {
            reason: Missingness::NotMeasured,
        };
        assert_eq!(
            graph_with_quantities(vec![missing_direct, fitted.clone()])
                .validate()
                .unwrap_err(),
            "fitted quantity quantity:fitted-n input quantity:psi-delta is not an observed quantity"
        );

        let EvidenceBasis::ExperimentalFitted {
            input_quantity_ids, ..
        } = &mut fitted.evidence
        else {
            unreachable!()
        };
        input_quantity_ids[0] = identifier("quantity:absent");
        assert!(
            graph_with_quantities(vec![direct, fitted])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn model_run_rejects_empty_parameter_and_convergence_entries() {
        let mut run = model_run("model:empty-map-entry", vec![identifier("quantity:input")]);
        run.parameters = BTreeMap::from([("".to_owned(), "".to_owned())]);
        assert_eq!(
            run.validate().unwrap_err(),
            "model parameter name must be nonempty"
        );

        let mut run = model_run("model:empty-map-entry", vec![identifier("quantity:input")]);
        run.convergence_settings = BTreeMap::from([("tolerance".to_owned(), " ".to_owned())]);
        assert_eq!(
            run.validate().unwrap_err(),
            "convergence setting value must be nonempty"
        );
    }

    #[test]
    fn model_run_requires_explicit_physical_bindings() {
        let mut run = model_run("model:missing-binding", vec![identifier("quantity:input")]);
        run.specimen_id = RecordId(" ".to_owned());
        assert_eq!(
            run.validate().unwrap_err(),
            "model-run specimen identifier must be nonempty"
        );

        let mut run = model_run("model:missing-binding", vec![identifier("quantity:input")]);
        run.geometry.clear();
        assert_eq!(
            run.validate().unwrap_err(),
            "model-run geometry must be nonempty"
        );

        let mut run = model_run("model:missing-binding", vec![identifier("quantity:input")]);
        run.constitutive_model.clear();
        assert_eq!(
            run.validate().unwrap_err(),
            "constitutive model must be nonempty"
        );

        let mut run = model_run("model:missing-binding", vec![identifier("quantity:input")]);
        run.conditions.clear();
        assert_eq!(
            run.validate().unwrap_err(),
            "model run requires bound conditions"
        );

        let mut run = model_run("model:missing-binding", vec![identifier("quantity:input")]);
        run.state_history = vec![" ".to_owned()];
        assert_eq!(
            run.validate().unwrap_err(),
            "model-run state-history entry must be nonempty"
        );
    }

    #[test]
    fn model_run_bindings_match_inputs_and_derived_outputs() {
        let graph = graph_with_model_and_derived();
        assert!(graph.validate().is_ok());

        let mut specimen_mismatch = graph.clone();
        let mut alternate_specimen = specimen_mismatch.specimens[0].clone();
        alternate_specimen.specimen_id = identifier("specimen:au:alternate");
        specimen_mismatch.specimens.push(alternate_specimen);
        specimen_mismatch.model_runs[0].specimen_id = identifier("specimen:au:alternate");
        assert_eq!(
            specimen_mismatch.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 specimen does not match input quantity:epsilon specimen"
        );

        let mut geometry_mismatch = graph.clone();
        geometry_mismatch.model_runs[0].geometry = "spherical".to_owned();
        assert_eq!(
            geometry_mismatch.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 geometry does not match specimen specimen:au:test"
        );

        let mut condition_mismatch = graph.clone();
        condition_mismatch.model_runs[0].conditions =
            BTreeMap::from([("wavelength".to_owned(), "0.600 um".to_owned())]);
        assert_eq!(
            condition_mismatch.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 conditions do not match input quantity:epsilon conditions"
        );

        let mut history_mismatch = graph.clone();
        history_mismatch.model_runs[0].state_history = vec!["annealed".to_owned()];
        assert_eq!(
            history_mismatch.validate().unwrap_err(),
            "model run model:drude-lorentz:v1 state history does not match state state:au:test"
        );

        let mut output_mismatch = graph;
        output_mismatch.derived_values[0].output.conditions =
            BTreeMap::from([("wavelength".to_owned(), "0.600 um".to_owned())]);
        assert_eq!(
            output_mismatch.validate().unwrap_err(),
            "derived value derived:reflectivity output conditions do not match model run model:drude-lorentz:v1 conditions"
        );
    }

    #[test]
    fn measurement_quantity_conditions_match_material_state() {
        let mut quantity = scalar_quantity(
            QuantityOrigin::Measurement {
                measurement_id: identifier("measurement:ellipsometry"),
            },
            EvidenceBasis::ExperimentalDirect,
        );
        quantity
            .conditions
            .insert("temperature_k".to_owned(), "300".to_owned());
        let graph = graph_with_quantities(vec![quantity]);
        assert!(graph.validate().is_ok());

        let mut strain_match = graph.clone();
        strain_match.states[0].strain = Some(vec![0.001, -0.002]);
        strain_match.quantities[0]
            .conditions
            .insert("strain_component:1".to_owned(), "-0.002".to_owned());
        assert!(strain_match.validate().is_ok());

        strain_match.quantities[0]
            .conditions
            .insert("strain_component:1".to_owned(), "0.002".to_owned());
        assert_eq!(
            strain_match.validate().unwrap_err(),
            "quantity quantity:test condition strain_component:1 contradicts state state:au:test"
        );

        let mut temperature_mismatch = graph.clone();
        temperature_mismatch.quantities[0]
            .conditions
            .insert("temperature_k".to_owned(), "100".to_owned());
        assert_eq!(
            temperature_mismatch.validate().unwrap_err(),
            "quantity quantity:test condition temperature_k contradicts state state:au:test"
        );

        let mut ambiguous_temperature = graph.clone();
        ambiguous_temperature.quantities[0]
            .conditions
            .insert("temperature".to_owned(), "100 K".to_owned());
        assert_eq!(
            ambiguous_temperature.validate().unwrap_err(),
            "quantity quantity:test condition temperature must use an explicit SI key"
        );

        let mut ambiguous_strain = graph;
        ambiguous_strain.quantities[0]
            .conditions
            .insert("strain".to_owned(), "0.001".to_owned());
        assert_eq!(
            ambiguous_strain.validate().unwrap_err(),
            "quantity quantity:test condition strain must use an explicit SI key"
        );
    }

    #[test]
    fn evidence_graph_rejects_unknown_specimen_reference() {
        let measurement = Measurement {
            measurement_id: identifier("measurement:missing-specimen"),
            specimen_id: identifier("specimen:absent"),
            method: "spectroscopic ellipsometry".to_owned(),
            instrument: "instrument identity retained in source record".to_owned(),
            calibration_or_standard: "source calibration procedure".to_owned(),
            operator_or_facility: "source laboratory".to_owned(),
            raw_artifact_id: identifier("artifact:raw"),
            processing_recipe_id: identifier("recipe:ellipsometry"),
            repeat_count: 1,
            detection_limit: None,
            provenance: Provenance {
                source_id: "source:test".to_owned(),
                doi_or_stable_id: "doi:10.example/test".to_owned(),
                locator: "Table 1".to_owned(),
                license: "source terms recorded".to_owned(),
                retrieval_date: "2026-09-12".to_owned(),
                source_sha256: "0".repeat(64),
                parser_version: "manual-transcription-v1".to_owned(),
                transformation_lineage: vec!["source table to typed fixture".to_owned()],
            },
        };
        let graph = MaterialEvidenceGraph {
            materials: Vec::new(),
            states: Vec::new(),
            specimens: Vec::new(),
            measurements: vec![measurement],
            quantities: Vec::new(),
            model_runs: Vec::new(),
            derived_values: Vec::new(),
        };
        assert!(graph.validate().is_err());
    }

    #[test]
    fn evidence_graph_rejects_specimen_substrate_cycles() {
        let graph = graph_with_quantities(Vec::new());

        let mut self_cycle = graph.clone();
        self_cycle.specimens[0].substrate_specimen_id =
            Some(self_cycle.specimens[0].specimen_id.clone());
        assert_eq!(
            self_cycle.validate().unwrap_err(),
            "specimen substrate references contain a cycle at specimen:au:test"
        );

        let mut two_specimen_cycle = graph;
        let mut substrate = two_specimen_cycle.specimens[0].clone();
        substrate.specimen_id = identifier("specimen:au:substrate");
        substrate.substrate_specimen_id = Some(identifier("specimen:au:test"));
        two_specimen_cycle.specimens[0].substrate_specimen_id =
            Some(substrate.specimen_id.clone());
        two_specimen_cycle.specimens.push(substrate);
        assert_eq!(
            two_specimen_cycle.validate().unwrap_err(),
            "specimen substrate references contain a cycle at specimen:au:test"
        );
    }

    #[test]
    fn derived_output_cannot_claim_direct_measurement() {
        let quantity = QuantityValue {
            quantity_id: identifier("quantity:derived"),
            origin: QuantityOrigin::ModelRun {
                model_run_id: identifier("model:drude-lorentz:v1"),
            },
            family: PropertyFamily::OpticalPhotonic,
            quantity_kind: "reflectivity".to_owned(),
            unit: "1".to_owned(),
            tensor_component_or_basis: None,
            observation: QuantityObservation::Observed {
                payload: QuantityPayload::Scalar { value: 0.5 },
            },
            uncertainty: Uncertainty::NotReported {
                rationale: "model sensitivity remains open".to_owned(),
            },
            conditions: BTreeMap::new(),
            applicability_range: Some("0.1 to 6 eV".to_owned()),
            evidence: EvidenceBasis::ExperimentalDirect,
        };
        let derived = DerivedValue {
            derived_value_id: identifier("derived:reflectivity"),
            model_run_id: identifier("model:drude-lorentz:v1"),
            input_quantity_ids: vec![identifier("quantity:epsilon")],
            output: quantity,
            propagated_uncertainty: Uncertainty::NotReported {
                rationale: "input covariance unavailable".to_owned(),
            },
        };
        assert!(derived.validate().is_err());
    }
}
