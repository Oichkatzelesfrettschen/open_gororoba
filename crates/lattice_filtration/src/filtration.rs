//! Collision-storm filtration simulator for thesis falsification.

use crate::basis_index::BasisIndexCodec;
use crate::patricia_trie::PatriciaIndex;
use crate::survival_spectrum::{classify_latency_law, inverse_square_r2, LatencyLaw};
use std::collections::HashMap;

/// Per-step collision observation.
#[derive(Debug, Clone, Copy)]
pub struct CollisionObservation {
    pub step: usize,
    pub radius: f64,
    pub latency: f64,
    pub collisions: usize,
}

/// Aggregate stats for a run.
#[derive(Debug, Clone)]
pub struct CollisionStormStats {
    pub n_steps: usize,
    pub total_collisions: usize,
    pub peak_bucket_occupancy: usize,
    pub mean_latency: f64,
    pub inverse_square_r2: f64,
    pub latency_law: LatencyLaw,
}

/// Simulate collision storm from a Fibonacci-driven key stream.
///
/// The stream is intentionally deterministic for reproducibility.
pub fn simulate_fibonacci_collision_storm(
    n_steps: usize,
    n_buckets: usize,
) -> (CollisionStormStats, Vec<CollisionObservation>) {
    assert!(n_steps > 0, "n_steps must be > 0");
    assert!(n_buckets > 0, "n_buckets must be > 0");

    let codec = BasisIndexCodec::new(16);
    let mut trie = PatriciaIndex::new();
    let mut buckets: HashMap<usize, usize> = HashMap::new();
    let mut observations = Vec::with_capacity(n_steps);

    let mut a = 1_u64;
    let mut b = 1_u64;
    let zero_div_anchor = codec.encode(0, -1);

    let mut total_collisions = 0usize;
    let mut peak_bucket = 0usize;
    let mut latency_sum = 0.0;

    for step in 0..n_steps {
        let fib = if step < 2 {
            1
        } else {
            let c = a.wrapping_add(b);
            a = b;
            b = c;
            c
        };

        let basis = (fib as usize) % 16;
        let sign = if (fib & 1) == 0 { 1 } else { -1 };
        let key = codec.encode(basis, sign);
        trie.insert(key);

        let bucket_id = (fib as usize) % n_buckets;
        let occupancy = buckets
            .entry(bucket_id)
            .and_modify(|x| *x += 1)
            .or_insert(1);
        let occ = *occupancy;

        if occ > 1 {
            total_collisions += occ - 1;
        }
        peak_bucket = peak_bucket.max(occ);

        let prefix = PatriciaIndex::shared_prefix_bits(zero_div_anchor, key) as f64;
        let radius = (64.0 - prefix).max(1.0);
        let latency = (occ as f64) / (radius * radius);
        latency_sum += latency;

        observations.push(CollisionObservation {
            step,
            radius,
            latency,
            collisions: occ.saturating_sub(1),
        });
    }

    let samples = observations
        .iter()
        .map(|o| (o.radius, o.latency))
        .collect::<Vec<_>>();
    let r2 = inverse_square_r2(&samples);
    let law = classify_latency_law(&samples);

    let stats = CollisionStormStats {
        n_steps,
        total_collisions,
        peak_bucket_occupancy: peak_bucket,
        mean_latency: latency_sum / n_steps as f64,
        inverse_square_r2: r2,
        latency_law: law,
    };
    (stats, observations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survival_spectrum::LatencyLaw;

    #[test]
    fn test_collision_storm_stats_nonempty() {
        let (stats, observations) = simulate_fibonacci_collision_storm(200, 13);
        assert_eq!(stats.n_steps, 200);
        assert_eq!(observations.len(), 200);
        assert!(stats.mean_latency.is_finite());
    }

    #[test]
    fn test_collision_storm_detects_inverse_square_tendency() {
        let (stats, _) = simulate_fibonacci_collision_storm(300, 17);
        assert!(
            matches!(
                stats.latency_law,
                LatencyLaw::InverseSquare | LatencyLaw::Undetermined
            ),
            "unexpected law: {:?}",
            stats.latency_law
        );
    }
}
