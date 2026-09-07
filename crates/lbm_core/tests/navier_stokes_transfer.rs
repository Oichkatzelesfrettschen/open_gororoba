//! Independent finite Fourier falsifiers for the signed Navier-Stokes diagnostic.

use lbm_core::turbulence::{
    calculate_triad_energy_transfer, extract_dominant_triads_3d, leray_project,
};
use ndarray::Array3;
use rustfft::num_complex::Complex64;
use std::collections::BTreeMap;

type Wave = [i32; 3];
type Velocity = [Complex64; 3];
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);

fn datum() -> [(Wave, Velocity); 3] {
    [
        ([1, 1, 0], [ZERO, ZERO, I]),
        ([1, 0, 0], [ZERO, ONE, ZERO]),
        ([0, 1, 0], [ZERO, ZERO, ONE]),
    ]
}

fn transfer(modes: [(Wave, Velocity); 3]) -> f64 {
    calculate_triad_energy_transfer(
        modes[0].0, modes[1].0, modes[2].0, modes[0].1, modes[1].1, modes[2].1,
    )
}

#[test]
fn exact_phase_reversal_and_three_mode_dependence() {
    let mut modes = datum();
    assert_eq!(transfer(modes), -1.0);
    modes[0].1 = modes[0].1.map(|z| -z);
    assert_eq!(transfer(modes), 1.0);
    modes[0].1 = modes[0].1.map(|z| z * I);
    assert_eq!(transfer(modes), 0.0);
    for j in 0..3 {
        let mut absent = datum();
        absent[j].1 = [ZERO; 3];
        assert_eq!(transfer(absent), 0.0, "missing mode {j}");
    }
}

#[test]
fn real_cubic_scaling_preserves_sign() {
    for lambda in [-3.0_f64, -0.5, 0.0, 0.5, 2.0] {
        let scaled = datum().map(|(k, u)| (k, u.map(|z| lambda * z)));
        assert_eq!(transfer(scaled), -lambda.powi(3));
    }
}

#[test]
fn signed_permutation_covariance_and_input_exchange() {
    for axes in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for mask in 0..8 {
            let sign: [i32; 3] = std::array::from_fn(|j| if mask & (1 << j) == 0 { 1 } else { -1 });
            let mut transformed = datum().map(|(k, u)| {
                (
                    std::array::from_fn(|j| sign[j] * k[axes[j]]),
                    std::array::from_fn(|j| f64::from(sign[j]) * u[axes[j]]),
                )
            });
            assert_eq!(transfer(transformed), -1.0);
            transformed.swap(1, 2);
            assert_eq!(transfer(transformed), -1.0);
        }
    }
}

/// Ordered-pair convolution with an explicit projector matrix, independent
/// of the production triad and projection routines.
fn oracle(field: &BTreeMap<Wave, Velocity>) -> BTreeMap<Wave, Velocity> {
    let mut result = BTreeMap::new();
    for (&p, up) in field {
        for (&q, uq) in field {
            let k: Wave = std::array::from_fn(|j| p[j] + q[j]);
            let entry = result.entry(k).or_insert([ZERO; 3]);
            let norm: f64 = k.iter().map(|&x| f64::from(x).powi(2)).sum();
            if norm == 0.0 {
                continue;
            }
            for (a, value) in entry.iter_mut().enumerate() {
                for b in 0..3 {
                    let projector =
                        f64::from(u8::from(a == b)) - f64::from(k[a]) * f64::from(k[b]) / norm;
                    for c in 0..3 {
                        *value += -I * projector * f64::from(q[c]) * up[c] * uq[b];
                    }
                }
            }
        }
    }
    result
}

fn rich_field() -> BTreeMap<Wave, Velocity> {
    let mut field = BTreeMap::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let k = [x, y, z];
                if k <= [0, 0, 0] {
                    continue;
                }
                let u = leray_project(
                    k,
                    std::array::from_fn(|j| {
                        Complex64::new(
                            f64::from(x + 2 * y + 3 * z) + j as f64,
                            f64::from(3 * x - y + z) - 0.5 * j as f64,
                        )
                    }),
                );
                field.insert(k, u);
                field.insert(k.map(|n| -n), u.map(|a| a.conj()));
            }
        }
    }
    field
}

#[test]
fn direct_convolution_agreement_energy_conservation_and_incompressibility() {
    let field = rich_field();
    let nonlinear = oracle(&field);
    let mut total = 0.0;
    let mut absolute_total = 0.0;
    for (&k, nk) in &nonlinear {
        let divergence: Complex64 = (0..3).map(|j| f64::from(k[j]) * nk[j]).sum();
        assert!(
            divergence.norm() < 2e-11,
            "projection at {k:?}: {divergence}"
        );
        let uk = field.get(&k).copied().unwrap_or([ZERO; 3]);
        let expected: f64 = (0..3).map(|j| (uk[j].conj() * nk[j]).re).sum();
        let mut actual = 0.0;
        for (&p, &up) in &field {
            let q = std::array::from_fn(|j| k[j] - p[j]);
            if p <= q
                && let Some(&uq) = field.get(&q)
            {
                actual += calculate_triad_energy_transfer(k, p, q, uk, up, uq);
            }
        }
        assert!(
            (actual - expected).abs() < 2e-10,
            "mode {k:?}: {actual} != {expected}"
        );
        total += actual;
        absolute_total += actual.abs();
    }
    assert!(
        absolute_total > 1.0,
        "conservation control must be nonlinear"
    );
    assert!(total.abs() < 1e-12 * absolute_total, "sum T = {total}");
    assert_eq!(nonlinear[&[0, 0, 0]], [ZERO; 3]);
}

#[test]
fn projection_is_idempotent_and_zero_mode_is_explicit() {
    let u = [Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0), I];
    let k = [2, -3, 4];
    let projected = leray_project(k, u);
    let again = leray_project(k, projected);
    for j in 0..3 {
        assert!((again[j] - projected[j]).norm() < 1e-14);
    }
    let divergence: Complex64 = (0..3).map(|j| f64::from(k[j]) * projected[j]).sum();
    assert!(divergence.norm() < 1e-14);
    assert_eq!(leray_project([0, 0, 0], u), [ZERO; 3]);
}

#[test]
#[should_panic(expected = "triad must satisfy")]
fn invalid_closure_is_rejected() {
    calculate_triad_energy_transfer(
        [1, 2, 3],
        [1, 0, 0],
        [0, 1, 0],
        [ONE; 3],
        [ONE; 3],
        [ONE; 3],
    );
}

#[test]
fn fft_extraction_uses_all_components_and_resolution_independent_normalization() {
    for n in [16, 24] {
        let fields: [Array3<f64>; 3] = std::array::from_fn(|j| {
            Array3::from_shape_fn((n, n, n), |(x, y, z)| {
                datum()
                    .iter()
                    .map(|(k, u)| {
                        let angle = std::f64::consts::TAU
                            * (f64::from(k[0]) * x as f64
                                + f64::from(k[1]) * y as f64
                                + f64::from(k[2]) * z as f64)
                            / n as f64;
                        2.0 * (u[j] * Complex64::from_polar(1.0, angle)).re
                    })
                    .sum()
            })
        });
        let triads = extract_dominant_triads_3d(&fields[0], &fields[1], &fields[2], 1e-8);
        let receiving = triads
            .iter()
            .find(|t| t.k == [-1, -1, 0] && t.p == [0, 1, 0] && t.q == [1, 0, 0])
            .expect("nonzero diagnostic triad");
        assert!(
            (receiving.energy_transfer + 1.0).abs() < 1e-12,
            "{}",
            receiving.energy_transfer
        );
    }
}

#[test]
fn extracted_receivers_are_covariant_and_conserve_each_closed_triple() {
    let sample = |modes: [(Wave, Velocity); 3]| {
        let fields: [Array3<f64>; 3] = std::array::from_fn(|j| {
            Array3::from_shape_fn((16, 16, 16), |(x, y, z)| {
                modes
                    .iter()
                    .map(|(k, u)| {
                        let angle = std::f64::consts::TAU
                            * (f64::from(k[0]) * x as f64
                                + f64::from(k[1]) * y as f64
                                + f64::from(k[2]) * z as f64)
                            / 16.0;
                        2.0 * (u[j] * Complex64::from_polar(1.0, angle)).re
                    })
                    .sum()
            })
        });
        extract_dominant_triads_3d(&fields[0], &fields[1], &fields[2], 1e-8)
    };
    let baseline = sample(datum());
    assert_eq!(
        baseline.len(),
        6,
        "both Hermitian triples need three receivers"
    );
    for triple in baseline.chunks_exact(3) {
        assert!(triple.iter().map(|t| t.energy_transfer).sum::<f64>().abs() < 1e-12);
    }
    for axes in [[0, 1, 2], [1, 0, 2], [2, 0, 1]] {
        for sign in [[1, 1, 1], [-1, 1, -1]] {
            let rotate = |k: Wave| std::array::from_fn(|j| sign[j] * k[axes[j]]);
            let transformed = sample(datum().map(|(k, u)| {
                (
                    rotate(k),
                    std::array::from_fn(|j| f64::from(sign[j]) * u[axes[j]]),
                )
            }));
            assert_eq!(baseline.len(), transformed.len());
            for t in &baseline {
                let other = transformed
                    .iter()
                    .find(|r| {
                        r.k == rotate(t.k)
                            && ((r.p == rotate(t.p) && r.q == rotate(t.q))
                                || (r.p == rotate(t.q) && r.q == rotate(t.p)))
                    })
                    .expect("every receiving mode must transform with its wavevector");
                assert!((t.energy_transfer - other.energy_transfer).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn two_dimensional_transfer_uses_both_velocity_components() {
    let t = calculate_triad_energy_transfer(
        [1, 2],
        [1, 0],
        [0, 2],
        [2.0 * I, -I],
        [ZERO, ONE],
        [ONE, ZERO],
    );
    assert!((t + 3.0).abs() < 1e-14);
}

#[test]
fn invalid_grid_data_and_arithmetic_overflow_cannot_become_empty_diagnostics() {
    use lbm_core::turbulence::{extract_dominant_triads, power_spectrum_details};
    for invalid in [f64::NAN, f64::INFINITY, f64::MAX] {
        let field2 = ndarray::Array2::from_elem((8, 8), invalid);
        assert!(
            std::panic::catch_unwind(|| extract_dominant_triads(&field2, &field2, 0.0)).is_err()
        );
        let field3 = Array3::from_elem((8, 8, 8), invalid);
        assert!(
            std::panic::catch_unwind(|| extract_dominant_triads_3d(&field3, &field3, &field3, 0.0))
                .is_err()
        );
    }
    let empty = Array3::zeros((0, 8, 8));
    assert!(
        std::panic::catch_unwind(|| extract_dominant_triads_3d(&empty, &empty, &empty, 0.0))
            .is_err()
    );
    let overflowing_power = ndarray::Array2::from_elem((1, 1), 1e200);
    assert!(std::panic::catch_unwind(|| power_spectrum_details(&overflowing_power)).is_err());
}

#[test]
fn complete_shell_accounting_satisfies_parseval_on_rectangular_grids() {
    for (nx, ny) in [(1, 1), (7, 9), (8, 12)] {
        let field = ndarray::Array2::from_shape_fn((nx, ny), |(x, y)| {
            0.7 + ((x * 17 + y * 31) as f64).sin() + if (x + y) % 2 == 0 { 1.0 } else { -1.0 }
        });
        let spectrum = lbm_core::turbulence::power_spectrum_details(&field);
        assert_eq!(spectrum.sample_count, nx * ny);
        assert_eq!(
            1 + spectrum.shells.iter().map(|s| s.mode_count).sum::<usize>(),
            nx * ny
        );
        let physical = field.iter().map(|x| x * x).sum::<f64>() / (nx * ny) as f64;
        let spectral = spectrum.dc_power
            + spectrum
                .shells
                .iter()
                .map(|s| s.integrated_power)
                .sum::<f64>();
        assert!((physical - spectral).abs() < 1e-12);
        for shell in spectrum.shells {
            assert!(
                (shell.mean_power * shell.mode_count as f64 - shell.integrated_power).abs() < 1e-14
            );
            assert_eq!(2.0 * shell.kinetic_energy, shell.integrated_power);
        }
    }
}
