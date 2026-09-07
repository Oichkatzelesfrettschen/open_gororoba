//! Calibrate conditional radius-angle dependence on fixed synthetic ensembles.
//!
//! Radial permutations preserve the angular selection and radial multiset.
//! The score measures dependence; flat clusters and gradients deliberately
//! challenge a hierarchy-specific interpretation of detected dependence.

use rand::{RngExt, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use stats_core::ultrametric::dendrogram::euclidean_distance_matrix_3d;

const POINTS: usize = 32;
const PERMUTATIONS: usize = 199;
const ALPHA: f64 = 0.05;
const FAMILIES: [&str; 5] = [
    "nested_strong",
    "nested_weak",
    "exchangeable_control",
    "flat_coupled_stress",
    "continuous_gradient_stress",
];

#[derive(Clone, Debug, PartialEq)]
struct Point {
    direction: (f64, f64, f64),
    radius: f64,
}

fn generate(family: usize, seed: u64) -> Vec<Point> {
    let mut random = ChaCha8Rng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(POINTS);
    for row in 0..POINTS {
        // Interleaving leaves gives the weak injection balanced retained support.
        let parent = ((row % 4) / 2) as f64;
        let child = (row % 2) as f64;
        let fraction = row as f64 / (POINTS - 1) as f64;
        let (longitude_center, radial_center) = match family {
            3 => (parent * 1.6 - 0.8, 1000.0 + parent * 600.0),
            4 => (-1.0 + 2.0 * fraction, 1000.0 + 600.0 * fraction),
            _ => (
                parent * 1.6 - 0.8 + child * 0.12 - 0.06,
                1000.0 + parent * 600.0 + child * 30.0,
            ),
        };
        let longitude = longitude_center + random.random_range(-0.01..0.01);
        let latitude: f64 = random.random_range(-0.01..0.01);
        let radius = radial_center + random.random_range(-5.0..5.0);
        points.push(Point {
            direction: (
                latitude.cos() * longitude.cos(),
                latitude.cos() * longitude.sin(),
                latitude.sin(),
            ),
            radius,
        });
    }
    if family == 1 || family == 2 {
        let first = if family == 1 { POINTS / 4 } else { 0 };
        let mut radii: Vec<f64> = points[first..].iter().map(|point| point.radius).collect();
        radii.shuffle(&mut random);
        for (point, radius) in points[first..].iter_mut().zip(radii) {
            point.radius = radius;
        }
    }
    points
}

fn canonical_points(points: &[Point]) -> Vec<Point> {
    let mut ordered = points.to_vec();
    ordered.sort_by(|left, right| {
        left.direction
            .0
            .total_cmp(&right.direction.0)
            .then(left.direction.1.total_cmp(&right.direction.1))
            .then(left.direction.2.total_cmp(&right.direction.2))
            .then(left.radius.total_cmp(&right.radius))
    });
    ordered
}

fn centered(mut matrix: Vec<f64>) -> Vec<f64> {
    let row_means: Vec<f64> = matrix
        .chunks_exact(POINTS)
        .map(|row| row.iter().sum::<f64>() / POINTS as f64)
        .collect();
    let grand_mean = row_means.iter().sum::<f64>() / POINTS as f64;
    for row in 0..POINTS {
        for column in 0..POINTS {
            matrix[row * POINTS + column] += grand_mean - row_means[row] - row_means[column];
        }
    }
    matrix
}

fn angular_matrix(points: &[Point]) -> Vec<f64> {
    let directions: Vec<_> = points.iter().map(|point| point.direction).collect();
    let condensed = euclidean_distance_matrix_3d(&directions);
    let mut matrix = vec![0.0; POINTS * POINTS];
    let mut index = 0;
    for row in 0..POINTS {
        for column in row + 1..POINTS {
            matrix[row * POINTS + column] = condensed[index];
            matrix[column * POINTS + row] = condensed[index];
            index += 1;
        }
    }
    centered(matrix)
}

fn score(angular: &[f64], radii: &[f64]) -> f64 {
    let radial = centered(
        radii
            .iter()
            .flat_map(|left| radii.iter().map(move |right| (left - right).abs()))
            .collect(),
    );
    let covariance: f64 = angular
        .iter()
        .zip(&radial)
        .map(|(left, right)| left * right)
        .sum();
    let angular_variance: f64 = angular.iter().map(|value| value * value).sum();
    let radial_variance: f64 = radial.iter().map(|value| value * value).sum();
    let statistic = covariance / (angular_variance * radial_variance).sqrt();
    assert!(
        statistic.is_finite(),
        "Degenerate or nonfinite dependence score"
    );
    assert!((-1e-12..=1.0 + 1e-12).contains(&statistic));
    statistic
}

fn radius_bits(radii: &[f64]) -> Vec<u64> {
    let mut bits: Vec<_> = radii.iter().map(|radius| radius.to_bits()).collect();
    bits.sort_unstable();
    bits
}

fn wilson(rejections: usize, replicates: usize) -> (f64, f64) {
    let quantile = 1.644_853_626_951_472_2;
    let count = replicates as f64;
    let fraction = rejections as f64 / count;
    let denominator = 1.0 + quantile * quantile / count;
    let center = (fraction + quantile * quantile / (2.0 * count)) / denominator;
    let half_width = quantile
        * (fraction * (1.0 - fraction) / count + quantile * quantile / (4.0 * count * count))
            .sqrt()
        / denominator;
    (
        (center - half_width).max(0.0),
        (center + half_width).min(1.0),
    )
}

fn check_score_oracles() {
    // Identical binary distances have normalized inner product one.
    let points: Vec<_> = (0..POINTS)
        .map(|row| Point {
            direction: if row < POINTS / 2 {
                (1.0, 0.0, 0.0)
            } else {
                (0.0, 1.0, 0.0)
            },
            radius: if row < POINTS / 2 { 1.0 } else { 2.0 },
        })
        .collect();
    let angular = angular_matrix(&points);
    let radii: Vec<_> = points.iter().map(|point| point.radius).collect();
    assert!((score(&angular, &radii) - 1.0).abs() < 1e-12);
    let rescaled: Vec<_> = radii.iter().map(|radius| radius * 7.0 + 31.0).collect();
    assert!((score(&angular, &rescaled) - 1.0).abs() < 1e-12);
    // A balanced Cartesian product of two binary variables has zero covariance.
    let independent: Vec<_> = (0..POINTS).map(|row| (row % 2) as f64).collect();
    assert!(score(&angular, &independent).abs() < 1e-12);
}

fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let confirmatory = match arguments.as_slice() {
        [] => false,
        [option] if option == "--confirmatory-precision" => true,
        _ => {
            eprintln!("Usage: frb_null_sensitivity [--confirmatory-precision]");
            std::process::exit(2);
        }
    };
    let replicates = if confirmatory { 1000 } else { 100 };
    let data_seed_base = if confirmatory { 1_000_000 } else { 100_000 };
    let permutation_seed_base = if confirmatory { 2_000_000 } else { 200_000 };
    let family_seed_stride = if confirmatory { 10_000 } else { 1000 };
    check_score_oracles();
    println!("schema_version = 1");
    if confirmatory {
        println!("protocol_id = \"frb-conditional-radius-angle-confirmatory-precision\"");
        println!("execution_kind = \"independent_confirmatory_precision\"");
    } else {
        println!("protocol_id = \"frb-conditional-radius-angle-dependence\"");
        println!("execution_kind = \"heldout_synthetic_instrument_calibration\"");
    }
    println!(
        "points = {POINTS}\npermutations = {PERMUTATIONS}\nreplicates_per_family = {replicates}"
    );
    println!("alpha = {ALPHA}");
    println!("physical_evidence = false\nhierarchy_specificity_claim = false");
    let mut rejections = [0_usize; FAMILIES.len()];
    let mut identity_checks = 0;
    let mut multiset_checks = 0;
    for (family, family_name) in FAMILIES.iter().enumerate() {
        for replicate in 0..replicates {
            let data_seed = data_seed_base + family as u64 * family_seed_stride + replicate as u64;
            let permutation_seed =
                permutation_seed_base + family as u64 * family_seed_stride + replicate as u64;
            let generated = generate(family, data_seed);
            assert_eq!(generated, generate(family, data_seed));
            for point in &generated {
                let (horizontal, vertical, polar) = point.direction;
                assert!(
                    (horizontal * horizontal + vertical * vertical + polar * polar - 1.0).abs()
                        < 1e-12
                );
            }
            let points = canonical_points(&generated);
            let angular = angular_matrix(&points);
            let original: Vec<f64> = points.iter().map(|point| point.radius).collect();
            let observed = score(&angular, &original);
            let mut relabeled = generated.clone();
            relabeled.reverse();
            let relabeled = canonical_points(&relabeled);
            let relabeled_radii: Vec<_> = relabeled.iter().map(|point| point.radius).collect();
            assert_eq!(
                score(&angular_matrix(&relabeled), &relabeled_radii).to_bits(),
                observed.to_bits()
            );
            identity_checks += 1;
            let original_bits = radius_bits(&original);
            let mut random = ChaCha8Rng::seed_from_u64(permutation_seed);
            let mut replay = ChaCha8Rng::seed_from_u64(permutation_seed);
            let mut replay_first = original.clone();
            replay_first.shuffle(&mut replay);
            let mut exceedances = 0;
            let mut null_sum = 0.0;
            let mut null_min = f64::INFINITY;
            let mut null_max = f64::NEG_INFINITY;
            for permutation in 0..PERMUTATIONS {
                let mut shuffled = original.clone();
                shuffled.shuffle(&mut random);
                if permutation == 0 {
                    assert_eq!(shuffled, replay_first);
                }
                assert_eq!(radius_bits(&shuffled), original_bits);
                multiset_checks += 1;
                let null_score = score(&angular, &shuffled);
                exceedances += usize::from(null_score >= observed);
                null_sum += null_score;
                null_min = null_min.min(null_score);
                null_max = null_max.max(null_score);
            }
            let p_value = (exceedances + 1) as f64 / (PERMUTATIONS + 1) as f64;
            let rejected = p_value <= ALPHA;
            rejections[family] += usize::from(rejected);
            println!("\n[[replicate]]\nfamily = \"{family_name}\"\nindex = {replicate}");
            println!("data_seed = {data_seed}\npermutation_seed = {permutation_seed}");
            println!(
                "observed_score = {observed:.17}\nnull_mean = {:.17}",
                null_sum / PERMUTATIONS as f64
            );
            println!("null_min = {null_min:.17}\nnull_max = {null_max:.17}");
            println!("exceedances = {exceedances}\np_value = {p_value:.17}\nrejected = {rejected}");
        }
    }
    for (family, family_name) in FAMILIES.iter().enumerate() {
        let (lower, upper) = wilson(rejections[family], replicates);
        println!(
            "\n[[family_summary]]\nfamily = \"{family_name}\"\nrejections = {}",
            rejections[family]
        );
        println!(
            "replicates = {replicates}\nrejection_rate = {:.17}",
            rejections[family] as f64 / replicates as f64
        );
        println!(
            "wilson_one_sided_95_lower = {lower:.17}\nwilson_one_sided_95_upper = {upper:.17}"
        );
    }
    let strong_power_pass = wilson(rejections[0], replicates).0 >= 0.80;
    let false_positive_pass = wilson(rejections[2], replicates).1 <= 0.10;
    println!(
        "\n[acceptance]\nstrong_power_pass = {strong_power_pass}\nfalse_positive_pass = {false_positive_pass}"
    );
    println!(
        "instrument_accepted = {}",
        strong_power_pass && false_positive_pass
    );
    println!(
        "\n[self_checks]\nrow_identity_checks = {identity_checks}\nradial_multiset_checks = {multiset_checks}"
    );
    println!(
        "data_seed_replay_checks = {identity_checks}\npermutation_seed_replay_checks = {identity_checks}"
    );
    println!(
        "finite_score_checks = {}",
        identity_checks * (PERMUTATIONS + 2)
    );
    println!("all_passed = true");
    println!("analytic_score_oracles = 3");
}
