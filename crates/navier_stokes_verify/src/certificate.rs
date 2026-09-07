// SPDX-License-Identifier: MIT

//! Versioned, self-contained certificate requests and deterministic replay.
//!
//! Coefficient strings name exact rational numbers. An unlisted coefficient
//! means zero only under the complete-polynomial declaration. The certified
//! datum is the explicitly recorded zero-mean Leray projection of the supplied
//! polynomial. No inferred source accuracy or unknown tail is certified.

use crate::{
    beltrami::radius_lower,
    interval::{ArithmeticError, MAX_PRECISION_BITS, checked_add, int, parse_rational, to_string},
    majorant::{enclose_slab, small_data_threshold},
    residual::{ExactComplex, FourierPolynomial, TemporalSlab, VerifyMathError, bound_slab},
    tail_bounds::TailDeclaration,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAX_JSON_BYTES: usize = 4 * 1024 * 1024;
const MAX_SLABS: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    Certified,
    Inconclusive,
    InvalidInput,
    ImplementationFailure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLevel {
    FiniteTimeCertificate,
    SpecifiedDataGlobalCertificate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Equation {
    UnforcedConstantViscosityNavierStokes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormConvention {
    HomogeneousH3NormalizedTorus2Pi,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TailInput {
    CompletePolynomial,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexInput {
    pub re: String,
    pub im: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModeInput {
    pub k: [i32; 3],
    pub u: [ComplexInput; 3],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialInput {
    pub modes: Vec<ModeInput>,
}

impl PolynomialInput {
    pub fn decode(&self) -> Result<FourierPolynomial, VerifyMathError> {
        if self.modes.len() > crate::residual::MAX_INPUT_MODES {
            return Err(VerifyMathError::InvalidInput(
                "input exceeds 64 Fourier modes",
            ));
        }
        let mut modes = Vec::with_capacity(self.modes.len());
        for mode in &self.modes {
            let mut vector = std::array::from_fn(|_| ExactComplex {
                re: int(0),
                im: int(0),
            });
            for (target, source) in vector.iter_mut().zip(&mode.u) {
                target.re = parse_rational(&source.re)?;
                target.im = parse_rational(&source.im)?;
            }
            modes.push((mode.k, vector));
        }
        FourierPolynomial::from_modes(modes)
    }

    pub fn from_polynomial(value: &FourierPolynomial) -> Self {
        Self {
            modes: value
                .modes()
                .iter()
                .map(|(k, u)| ModeInput {
                    k: *k,
                    u: std::array::from_fn(|axis| ComplexInput {
                        re: to_string(&u[axis].re),
                        im: to_string(&u[axis].im),
                    }),
                })
                .collect(),
        }
    }
}

/// Power coefficients in theta=(t-start)/(end-start), with degree at most 3.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlabInput {
    pub start: String,
    pub end: String,
    pub coefficients: Vec<PolynomialInput>,
}

impl SlabInput {
    fn decode(&self) -> Result<TemporalSlab, VerifyMathError> {
        if self.coefficients.is_empty() || self.coefficients.len() > 4 {
            return Err(VerifyMathError::InvalidInput(
                "temporal degree must be in 0..=3",
            ));
        }
        let slab = TemporalSlab {
            start: parse_rational(&self.start)?,
            end: parse_rational(&self.end)?,
            coefficients: self
                .coefficients
                .iter()
                .map(PolynomialInput::decode)
                .collect::<Result<_, _>>()?,
        };
        slab.validate()?;
        Ok(slab)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reference {
    BeltramiNeighborhood { center: PolynomialInput },
    PolynomialTrajectory { slabs: Vec<SlabInput> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub input_source: String,
    /// Context only; the exact serialized input, rather than this claim, is verified.
    pub solver_commit: Option<String>,
    /// Optional revision context supplied by the runner; source hash identifies code.
    pub verifier_commit: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub equation: Equation,
    pub norm: NormConvention,
    pub viscosity: String,
    pub precision_bits: u32,
    pub tails: TailInput,
    pub provenance: Provenance,
    pub initial: PolynomialInput,
    pub reference: Reference,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentRecord {
    pub start: String,
    pub end: String,
    pub r_start: String,
    pub r_end: String,
    pub slope: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlabRecord {
    pub start: String,
    pub end: String,
    pub d3_upper: String,
    pub d4_upper: String,
    pub residual3_upper: String,
    pub quadratic_output_cutoff: i32,
    pub residual_outside_reconstruction_support_upper: String,
    pub majorant: Vec<SegmentRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Bounds {
    BeltramiNeighborhood {
        m3_upper: String,
        distance3_upper: String,
        open_radius3_lower: String,
        reference_residual3: String,
    },
    PolynomialTrajectory {
        initial_error3_upper: String,
        slabs: Vec<SlabRecord>,
        endpoint: String,
        error3_at_endpoint_upper: String,
        solution3_at_endpoint_upper: String,
        small_data_threshold3: String,
        global_continuation: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    pub schema_version: u32,
    pub status: Status,
    pub evidence_level: Option<EvidenceLevel>,
    pub theorem_scope: Option<String>,
    pub reason: String,
    pub request: Request,
    pub request_sha256: String,
    pub verifier_source_sha256: String,
    pub trusted_analytic_contract: String,
    pub formal_proof_checked: bool,
    pub error_accounting: ErrorAccounting,
    pub projected_initial: Option<PolynomialInput>,
    pub supplied_initial_equals_projected: Option<bool>,
    pub bounds: Option<Bounds>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ErrorAccounting {
    pub k3: u32,
    pub g3: u32,
    pub arithmetic: String,
    pub source_quantization: String,
    pub input_and_reference_tails: String,
    pub aliasing: String,
    pub temporal_reconstruction: String,
    pub discarded_modes: String,
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(2 * bytes.len());
    for byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    result
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Hash length-prefixed compile-time sources and dependency lockfile.
pub fn source_sha256() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_str!("lib.rs"),
        include_str!("interval.rs"),
        include_str!("residual.rs"),
        include_str!("tail_bounds.rs"),
        include_str!("majorant.rs"),
        include_str!("beltrami.rs"),
        include_str!("certificate.rs"),
        include_str!("report.rs"),
        include_str!("../Cargo.toml"),
        include_str!("../../../Cargo.toml"),
        include_str!("../../../Cargo.lock"),
        include_str!("../../../rust-toolchain.toml"),
        include_str!("../../../.cargo/config.toml"),
    ] {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source.as_bytes());
    }
    hex(&hash.finalize())
}

#[derive(Debug)]
struct Failure {
    status: Status,
    reason: String,
}

impl From<ArithmeticError> for Failure {
    fn from(error: ArithmeticError) -> Self {
        let status = if error.is_implementation_failure() {
            Status::ImplementationFailure
        } else if error.is_invalid_input() {
            Status::InvalidInput
        } else {
            Status::Inconclusive
        };
        Self {
            status,
            reason: error.to_string(),
        }
    }
}

impl From<VerifyMathError> for Failure {
    fn from(error: VerifyMathError) -> Self {
        let status = if error.is_implementation_failure() {
            Status::ImplementationFailure
        } else if error.is_invalid_input() {
            Status::InvalidInput
        } else {
            Status::Inconclusive
        };
        Self {
            status,
            reason: error.to_string(),
        }
    }
}

fn reject(status: Status, message: &str) -> Failure {
    Failure {
        status,
        reason: message.into(),
    }
}

fn compute(request: &Request, certificate: &mut Certificate) -> Result<(), Failure> {
    if request.schema_version != 1 || !(1..=MAX_PRECISION_BITS).contains(&request.precision_bits) {
        return Err(reject(
            Status::InvalidInput,
            "unsupported schema version or precision",
        ));
    }
    let provenance = &request.provenance;
    if provenance.input_source.trim().is_empty()
        || provenance.input_source.len() > 1024
        || [&provenance.solver_commit, &provenance.verifier_commit]
            .into_iter()
            .any(|commit| {
                commit.as_ref().is_some_and(|sha| {
                    sha.len() != 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit())
                })
            })
    {
        return Err(reject(Status::InvalidInput, "invalid input provenance"));
    }
    let tails = match request.tails {
        TailInput::CompletePolynomial => TailDeclaration::CompletePolynomial,
        TailInput::Unknown => TailDeclaration::Unknown,
    };
    if tails.h3_upper().is_none() {
        return Err(reject(
            Status::Inconclusive,
            "unknown Fourier tails require a proved bound; this schema supports complete polynomials only",
        ));
    }
    let viscosity = parse_rational(&request.viscosity)?;
    if viscosity <= int(0) {
        return Err(reject(Status::InvalidInput, "viscosity must be positive"));
    }
    let supplied = request.initial.decode()?;
    let initial = supplied.project_zero_mean()?;
    initial.validate_zero_mean_divergence_free()?;
    certificate.projected_initial = Some(PolynomialInput::from_polynomial(&initial));
    certificate.supplied_initial_equals_projected = Some(initial == supplied);
    let precision = request.precision_bits;
    match &request.reference {
        Reference::BeltramiNeighborhood { center } => {
            let center = center.decode()?;
            if !center.is_unit_curl_beltrami()? {
                return Err(reject(
                    Status::InvalidInput,
                    "reference must satisfy curl(v0)=v0 exactly",
                ));
            }
            if !center.nonlinear()?.is_zero() {
                return Err(reject(
                    Status::ImplementationFailure,
                    "Beltrami nonlinearity identity failed",
                ));
            }
            let m = center.sobolev_norm_upper(3, precision)?;
            let delta = initial
                .difference(&center)?
                .sobolev_norm_upper(3, precision)?;
            let radius = radius_lower(&viscosity, &m, precision)?;
            certificate.bounds = Some(Bounds::BeltramiNeighborhood {
                m3_upper: to_string(&m),
                distance3_upper: to_string(&delta),
                open_radius3_lower: to_string(&radius),
                reference_residual3: "0".into(),
            });
            if delta >= radius {
                return Err(reject(
                    Status::Inconclusive,
                    "strict Beltrami radius test was not satisfied",
                ));
            }
            certificate.evidence_level = Some(EvidenceLevel::SpecifiedDataGlobalCertificate);
            certificate.theorem_scope = Some("Global smooth existence for the recorded projected initial datum and every smooth real zero-mean divergence-free datum in the strict H3 ball centered at the recorded Beltrami reference with the reported lower radius; original equation, nu as supplied, all Fourier modes, t >= 0.".into());
            certificate.reason =
                "The complete initial distance is strictly below the enclosed sufficient radius."
                    .into();
        }
        Reference::PolynomialTrajectory { slabs } => {
            if slabs.is_empty() || slabs.len() > MAX_SLABS {
                return Err(reject(
                    Status::InvalidInput,
                    "trajectory requires 1..=32 slabs",
                ));
            }
            let decoded: Vec<_> = slabs
                .iter()
                .map(SlabInput::decode)
                .collect::<Result<_, _>>()?;
            if decoded[0].start != int(0) {
                return Err(reject(
                    Status::InvalidInput,
                    "trajectory must start at zero",
                ));
            }
            for pair in decoded.windows(2) {
                if !pair[0].joins_exactly(&pair[1])? {
                    return Err(reject(
                        Status::InvalidInput,
                        "temporal reconstruction has a gap, overlap, or Fourier jump",
                    ));
                }
            }
            let delta = initial
                .difference(&decoded[0].coefficients[0])?
                .sobolev_norm_upper(3, precision)?;
            let mut r = delta.clone();
            let mut records = Vec::new();
            for slab in &decoded {
                let bound = bound_slab(slab, &viscosity, precision)?;
                let Some(segments) = enclose_slab(
                    &slab.start,
                    &slab.end,
                    &r,
                    &viscosity,
                    &bound.d3_upper,
                    &bound.d4_upper,
                    &bound.residual3_upper,
                    precision,
                )?
                else {
                    return Err(reject(
                        Status::Inconclusive,
                        "bounded subdivision search found no slab supersolution",
                    ));
                };
                r = segments
                    .last()
                    .ok_or_else(|| {
                        reject(Status::ImplementationFailure, "empty accepted majorant")
                    })?
                    .r_end
                    .clone();
                records.push(SlabRecord {
                    start: to_string(&slab.start),
                    end: to_string(&slab.end),
                    d3_upper: to_string(&bound.d3_upper),
                    d4_upper: to_string(&bound.d4_upper),
                    residual3_upper: to_string(&bound.residual3_upper),
                    quadratic_output_cutoff: bound.quadratic_output_cutoff,
                    residual_outside_reconstruction_support_upper: to_string(
                        &bound.discarded_residual3_upper,
                    ),
                    majorant: segments
                        .into_iter()
                        .map(|s| SegmentRecord {
                            start: to_string(&s.start),
                            end: to_string(&s.end),
                            r_start: to_string(&s.r_start),
                            r_end: to_string(&s.r_end),
                            slope: to_string(&s.slope),
                        })
                        .collect(),
                });
            }
            let last = &decoded[decoded.len() - 1];
            let endpoint_norm = last.endpoints()?.1.sobolev_norm_upper(3, precision)?;
            let solution_norm = checked_add(&endpoint_norm, &r)?;
            let threshold = small_data_threshold(&viscosity)?;
            let global = solution_norm <= threshold;
            certificate.bounds = Some(Bounds::PolynomialTrajectory {
                initial_error3_upper: to_string(&delta),
                slabs: records,
                endpoint: to_string(&last.end),
                error3_at_endpoint_upper: to_string(&r),
                solution3_at_endpoint_upper: to_string(&solution_norm),
                small_data_threshold3: to_string(&threshold),
                global_continuation: global,
            });
            certificate.evidence_level = Some(if global {
                EvidenceLevel::SpecifiedDataGlobalCertificate
            } else {
                EvidenceLevel::FiniteTimeCertificate
            });
            certificate.theorem_scope = Some(if global {
                "Global smooth existence for the recorded projected initial datum, using the finite-time H3 error bound and endpoint small-data continuation; original equation, all Fourier modes."
            } else {
                "Smooth continuum existence and the recorded slab-wise H3 error bound through the reported endpoint for the recorded projected initial datum; original equation, all Fourier modes."
            }.into());
            certificate.reason =
                "Every whole-slab residual bound and affine supersolution check succeeded.".into();
        }
    }
    certificate.status = Status::Certified;
    Ok(())
}

/// A status is a result of recomputation, never a trusted assertion in input.
///
/// Typed callers must budget their own Request allocations: this API serializes
/// and clones the supplied value. Untrusted JSON should enter through
/// [`parse_request`], which applies its byte budget before deserialization.
/// Allocator or process aborts yield no certificate and are not caught panics.
pub fn verify(request: &Request) -> Certificate {
    let request_bytes = serde_json::to_vec(request);
    let mut certificate = Certificate {
        schema_version: 1, status: Status::ImplementationFailure, evidence_level: None,
        theorem_scope: None, reason: String::new(), request: request.clone(),
        request_sha256: request_bytes.as_ref().map(|bytes| sha256(bytes)).unwrap_or_default(),
        verifier_source_sha256: source_sha256(),
        trusted_analytic_contract: "H3 normalized zero-mean torus; K3=64, G3=96; exact rational implementation plus analytic local existence, continuation and comparison described in beltrami.rs; dependencies, compiler, runtime and certificate I/O remain trusted.".into(),
        formal_proof_checked: false,
        error_accounting: ErrorAccounting {
            k3: 64, g3: 96,
            arithmetic: "Exact rational operations; outward sqrt, exp and slope quantization; no ordinary floating-point operations in bounds.".into(),
            source_quantization: "Recorded rational coefficients define the exact input. Error in unrecorded source data is outside scope; binary64 import preserves only its represented value.".into(),
            input_and_reference_tails: "Zero only if the complete-polynomial declaration is accepted. Unknown tails prevent certification. This does not set the continuum solution tail to zero.".into(),
            aliasing: "Zero for exact nonmodular ordered convolution; no FFT aliasing path is used.".into(),
            temporal_reconstruction: "Analytic exponential for Beltrami; exact degree-at-most-three polynomial per slab otherwise. Derivatives and full residual are bounded throughout each slab; exact endpoint continuity is required.".into(),
            discarded_modes: "All quadratic modes through twice the reconstruction cutoff enter the residual bound, including modes outside reconstruction support; none are discarded.".into(),
        },
        projected_initial: None,
        supplied_initial_equals_projected: None, bounds: None,
    };
    if request_bytes
        .as_ref()
        .is_ok_and(|bytes| bytes.len() > MAX_JSON_BYTES)
    {
        certificate.status = Status::InvalidInput;
        certificate.reason = "request exceeds 4 MiB".into();
        return certificate;
    }
    if request_bytes.is_err() {
        certificate.reason = "request serialization failed".into();
        return certificate;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compute(request, &mut certificate)
    })) {
        Ok(Ok(())) => (),
        Ok(Err(error)) => {
            certificate.status = error.status;
            certificate.reason = error.reason;
        }
        Err(_) => {
            certificate.status = Status::ImplementationFailure;
            certificate.reason = "verifier panicked; no certification claim is valid".into();
        }
    }
    if certificate.status != Status::Certified {
        certificate.evidence_level = None;
        certificate.theorem_scope = None;
    }
    certificate
}

pub fn parse_request(bytes: &[u8]) -> Result<Request, String> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err("request exceeds 4 MiB".into());
    }
    serde_json::from_slice(bytes).map_err(|error| error.to_string())
}

/// Recompute all checks and compare the complete transcript, including source identity.
pub fn reproduce(certificate: &Certificate) -> bool {
    certificate.status == Status::Certified && verify(&certificate.request) == *certificate
}
