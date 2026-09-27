// SPDX-License-Identifier: MIT

use navier_stokes_verify::{
    certificate::{
        Bounds, Certificate, ComplexInput, EvidenceLevel, ModeInput, PolynomialInput, Reference,
        Request, SlabInput, Status, TailInput, parse_request, reproduce, verify,
    },
    interval::{checked_mul, int, parse_rational, rational},
};

fn fixture() -> Request {
    parse_request(include_bytes!("../fixtures/beltrami_neighborhood.json")).unwrap()
}

fn zero() -> PolynomialInput {
    PolynomialInput { modes: vec![] }
}

#[test]
fn nontrivial_beltrami_ball_certifies_above_small_data_regime() {
    let request = fixture();
    let certificate = verify(&request);
    assert_eq!(
        certificate.status,
        Status::Certified,
        "{}",
        certificate.reason
    );
    assert_eq!(
        certificate.evidence_level,
        Some(EvidenceLevel::SpecifiedDataGlobalCertificate)
    );
    assert!(!certificate.formal_proof_checked);
    let Bounds::BeltramiNeighborhood {
        m3_upper,
        distance3_upper,
        open_radius3_lower,
        ..
    } = certificate.bounds.as_ref().unwrap()
    else {
        panic!("wrong certificate kind");
    };
    let m = parse_rational(m3_upper).unwrap();
    let distance = parse_rational(distance3_upper).unwrap();
    let radius = parse_rational(open_radius3_lower).unwrap();
    assert!(m > rational(1, 96).unwrap());
    assert!(distance > int(0) && distance < radius);
    // Independent closed-form norms for the six ABC modes and perturbing pair.
    let squared_reference = rational(3, 64).unwrap();
    assert!(checked_mul(&m, &m).unwrap() >= squared_reference);
    let amplitude = parse_rational("1/1267650600228229401496703205376").unwrap();
    let squared_distance =
        checked_mul(&int(250), &checked_mul(&amplitude, &amplitude).unwrap()).unwrap();
    assert!(checked_mul(&distance, &distance).unwrap() >= squared_distance);
    assert_eq!(
        request
            .initial
            .decode()
            .unwrap()
            .sobolev_norm_squared(3)
            .unwrap(),
        squared_reference + squared_distance
    );
    assert!(
        !request
            .initial
            .decode()
            .unwrap()
            .nonlinear()
            .unwrap()
            .is_zero()
    );
    assert!(reproduce(&certificate));
    let encoded = serde_json::to_vec(&certificate).unwrap();
    let decoded: Certificate = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(certificate, decoded);
}

#[test]
fn every_transcript_part_is_recomputed() {
    let certificate = verify(&fixture());
    let mut tampered = certificate.clone();
    tampered.formal_proof_checked = true;
    assert!(!reproduce(&tampered));
    tampered = certificate.clone();
    tampered.request.viscosity = "2".into();
    assert!(!reproduce(&tampered));
    tampered = certificate.clone();
    tampered.verifier_source_sha256 = "0".repeat(64);
    assert!(!reproduce(&tampered));
    tampered = certificate;
    if let Some(Bounds::BeltramiNeighborhood {
        open_radius3_lower, ..
    }) = &mut tampered.bounds
    {
        *open_radius3_lower = "1".into();
    }
    assert!(!reproduce(&tampered));
}

#[test]
fn tails_large_perturbations_and_invalid_conventions_never_certify() {
    let mut request = fixture();
    request.tails = TailInput::Unknown;
    assert_eq!(verify(&request).status, Status::Inconclusive);
    request = fixture();
    request.initial.modes[6].u[2].re = "1".into();
    request.initial.modes[7].u[2].re = "1".into();
    let result = verify(&request);
    assert_eq!(result.status, Status::Inconclusive);
    assert!(result.evidence_level.is_none() && result.theorem_scope.is_none());
    request = fixture();
    request.viscosity = "0".into();
    assert_eq!(verify(&request).status, Status::InvalidInput);
    request = fixture();
    request.precision_bits = 0;
    assert_eq!(verify(&request).status, Status::InvalidInput);
    let raw = String::from_utf8(include_bytes!("../fixtures/beltrami_neighborhood.json").to_vec())
        .unwrap();
    assert!(
        parse_request(
            raw.replace("homogeneous_h3_normalized_torus2_pi", "inhomogeneous_h3")
                .as_bytes()
        )
        .is_err()
    );
    assert!(
        parse_request(
            raw.replace(
                "\"schema_version\": 1",
                "\"schema_version\": 1, \"interaction_weight\": 2"
            )
            .as_bytes()
        )
        .is_err()
    );
}

#[test]
fn certificate_identifies_changed_projected_datum() {
    let mut request = fixture();
    request.initial.modes.push(ModeInput {
        k: [0, 0, 0],
        u: std::array::from_fn(|_| ComplexInput {
            re: "1".into(),
            im: "0".into(),
        }),
    });
    request.initial.modes[0].u[0].re = "1".into();
    request.initial.modes[1].u[0].re = "1".into();
    let result = verify(&request);
    assert_eq!(result.status, Status::Certified, "{}", result.reason);
    assert_eq!(result.supplied_initial_equals_projected, Some(false));
    assert_eq!(
        result.projected_initial.unwrap().decode().unwrap(),
        fixture().initial.decode().unwrap()
    );
}

#[test]
fn zero_reference_strict_boundary_is_inconclusive() {
    let mut request = fixture();
    request.reference = Reference::BeltramiNeighborhood { center: zero() };
    request.initial = zero();
    assert_eq!(verify(&request).status, Status::Certified);
    // A real sine/cosine pair has norm exactly 1/96, at the excluded boundary.
    for sign in [-1, 1] {
        request.initial.modes.push(ModeInput {
            k: [sign, 0, 0],
            u: [
                ComplexInput {
                    re: "0".into(),
                    im: "0".into(),
                },
                ComplexInput {
                    re: "1/192".into(),
                    im: "0".into(),
                },
                ComplexInput {
                    re: "0".into(),
                    im: format!("{}/192", sign),
                },
            ],
        });
    }
    assert_eq!(verify(&request).status, Status::Inconclusive);
}

fn shear() -> PolynomialInput {
    PolynomialInput {
        modes: [-1, 1]
            .into_iter()
            .map(|sign| ModeInput {
                k: [sign, 0, 0],
                u: [
                    ComplexInput {
                        re: "0".into(),
                        im: "0".into(),
                    },
                    ComplexInput {
                        re: "1/1000".into(),
                        im: "0".into(),
                    },
                    ComplexInput {
                        re: "0".into(),
                        im: "0".into(),
                    },
                ],
            })
            .collect(),
    }
}

#[test]
fn complete_slab_residual_majorant_and_endpoint_continuation() {
    let mut request = fixture();
    request.initial = shear();
    request.reference = Reference::PolynomialTrajectory {
        slabs: vec![SlabInput {
            start: "0".into(),
            end: "1/1000".into(),
            coefficients: vec![shear()],
        }],
    };
    let certificate = verify(&request);
    assert_eq!(
        certificate.status,
        Status::Certified,
        "{}",
        certificate.reason
    );
    assert!(reproduce(&certificate));
    let Bounds::PolynomialTrajectory {
        initial_error3_upper,
        slabs,
        global_continuation,
        ..
    } = certificate.bounds.unwrap()
    else {
        panic!("wrong bounds")
    };
    assert_eq!(initial_error3_upper, "0");
    assert!(parse_rational(&slabs[0].residual3_upper).unwrap() > int(0));
    assert!(global_continuation);
}

#[test]
fn nonlinear_complete_residual_and_temporal_jumps_are_detected() {
    let mut request = fixture();
    let field = request.initial.clone();
    request.reference = Reference::PolynomialTrajectory {
        slabs: vec![SlabInput {
            start: "0".into(),
            end: "1/100000".into(),
            coefficients: vec![field],
        }],
    };
    let result = verify(&request);
    assert_eq!(result.status, Status::Certified, "{}", result.reason);
    assert_eq!(
        result.evidence_level,
        Some(EvidenceLevel::FiniteTimeCertificate)
    );
    let Some(Bounds::PolynomialTrajectory { slabs, .. }) = result.bounds else {
        panic!("wrong bounds")
    };
    assert!(
        parse_rational(&slabs[0].residual_outside_reconstruction_support_upper).unwrap() > int(0)
    );
    if let Reference::PolynomialTrajectory { slabs } = &mut request.reference {
        slabs.push(SlabInput {
            start: "1/100000".into(),
            end: "1/10000".into(),
            coefficients: vec![zero()],
        });
    }
    assert_eq!(verify(&request).status, Status::InvalidInput);
}

#[test]
fn derivative_discontinuity_is_allowed_but_field_jumps_are_not() {
    let mut request = fixture();
    request.initial = shear();
    let linear_end =
        PolynomialInput::from_polynomial(&shear().decode().unwrap().scaled(&int(2)).unwrap());
    request.reference = Reference::PolynomialTrajectory {
        slabs: vec![
            SlabInput {
                start: "0".into(),
                end: "1/10000".into(),
                coefficients: vec![shear()],
            },
            SlabInput {
                start: "1/10000".into(),
                end: "1/5000".into(),
                coefficients: vec![shear(), shear()],
            },
            SlabInput {
                start: "1/5000".into(),
                end: "3/10000".into(),
                coefficients: vec![linear_end],
            },
        ],
    };
    let result = verify(&request);
    assert_eq!(result.status, Status::Certified, "{}", result.reason);
}
