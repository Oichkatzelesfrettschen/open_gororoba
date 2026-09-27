// SPDX-License-Identifier: MIT

//! Human-readable certificate summaries preserve scope and the trust boundary.

use crate::certificate::{Bounds, Certificate};

pub fn human_readable(certificate: &Certificate) -> String {
    let status = match certificate.status {
        crate::certificate::Status::Certified => "CERTIFIED",
        crate::certificate::Status::Inconclusive => "INCONCLUSIVE",
        crate::certificate::Status::InvalidInput => "INVALID_INPUT",
        crate::certificate::Status::ImplementationFailure => "IMPLEMENTATION_FAILURE",
    };
    let mut result = format!("Status: {status}\nReason: {}\n", certificate.reason);
    if let Some(level) = certificate.evidence_level {
        result.push_str(&format!("Evidence level: {level:?}\n"));
    }
    if let Some(scope) = &certificate.theorem_scope {
        result.push_str(&format!("Scope: {scope}\n"));
    }
    if let Some(equal) = certificate.supplied_initial_equals_projected {
        result.push_str(&format!(
            "Supplied datum equals certified projection: {equal}\n"
        ));
    }
    match &certificate.bounds {
        Some(Bounds::BeltramiNeighborhood {
            m3_upper,
            distance3_upper,
            open_radius3_lower,
            ..
        }) => {
            result.push_str(&format!("Reference H3 upper: {m3_upper}\nInitial H3 distance upper: {distance3_upper}\nStrict H3 radius lower: {open_radius3_lower}\n"));
        }
        Some(Bounds::PolynomialTrajectory {
            endpoint,
            error3_at_endpoint_upper,
            global_continuation,
            ..
        }) => {
            result.push_str(&format!("Certified endpoint: {endpoint}\nEndpoint H3 error upper: {error3_at_endpoint_upper}\nSmall-data continuation: {global_continuation}\n"));
        }
        None => (),
    }
    result.push_str(&format!(
        "Formal proof checked: {}\nTrust: {}\nVerifier source SHA256: {}\nInput SHA256: {}\n",
        certificate.formal_proof_checked,
        certificate.trusted_analytic_contract,
        certificate.verifier_source_sha256,
        certificate.request_sha256
    ));
    result
}
