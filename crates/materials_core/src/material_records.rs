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
                if *rows == 0
                    || *columns == 0
                    || values_row_major.len() != rows * columns
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
                if *dimension == 0
                    || values_row_major.len() != dimension * dimension
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
}

impl QuantityValue {
    pub fn validate(&self) -> Result<(), String> {
        self.quantity_id.validate("quantity identifier")?;
        require_nonempty("quantity kind", &self.quantity_kind)?;
        require_nonempty("quantity unit", &self.unit)?;
        self.uncertainty.validate()?;
        match &self.observation {
            QuantityObservation::Observed { payload } => {
                payload.validate()?;
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
}

impl Specimen {
    pub fn validate(&self) -> Result<(), String> {
        self.specimen_id.validate("specimen identifier")?;
        self.state_id.validate("state identifier")?;
        require_nonempty("synthesis or deposition", &self.synthesis_or_deposition)?;
        require_nonempty("specimen geometry", &self.geometry)?;
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
        require_nonempty("model name", &self.model_name)?;
        require_nonempty("model version", &self.model_version)?;
        require_nonempty("validation status", &self.validation_status)?;
        if self.input_quantity_ids.is_empty() {
            return Err("model run requires at least one input quantity".to_owned());
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
        self.propagated_uncertainty.validate()
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
        let specimen_ids = unique_ids(
            "specimen identifier",
            self.specimens.iter().map(|record| &record.specimen_id),
        )?;
        let measurement_ids = unique_ids(
            "measurement identifier",
            self.measurements
                .iter()
                .map(|record| &record.measurement_id),
        )?;
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
        let model_run_ids = unique_ids(
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
                QuantityOrigin::Measurement { measurement_id }
                    if !measurement_ids.contains(measurement_id) =>
                {
                    return Err(format!(
                        "quantity {} references unknown measurement {}",
                        quantity.quantity_id.0, measurement_id.0
                    ));
                }
                QuantityOrigin::ModelRun { model_run_id }
                    if !model_run_ids.contains(model_run_id) =>
                {
                    return Err(format!(
                        "quantity {} references unknown model run {}",
                        quantity.quantity_id.0, model_run_id.0
                    ));
                }
                _ => {}
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
            }
        }
        for derived in &self.derived_values {
            let model_run = model_runs_by_id.get(&derived.model_run_id).ok_or_else(|| {
                format!(
                    "derived value {} references unknown model run {}",
                    derived.derived_value_id.0, derived.model_run_id.0
                )
            })?;
            let declared_inputs: BTreeSet<_> = model_run.input_quantity_ids.iter().collect();
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
            parameters: BTreeMap::from([("oscillators".to_owned(), "3".to_owned())]),
            convergence_settings: BTreeMap::from([(
                "relative_tolerance".to_owned(),
                "1e-10".to_owned(),
            )]),
            validation_status: "fixture validated".to_owned(),
        }
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
            "derived value derived:reflectivity input quantity:undeclared is not declared by model run model:drude-lorentz:v1"
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
