use super::daf::DafReader;
use super::chebyshev::{evaluate_chebyshev_position, evaluate_chebyshev_velocity};
use wide::f64x4;
use std::collections::HashMap;

/// Ephemeris evaluation result containing 3D position and velocity.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    pub position: [f64; 3], // km
    pub velocity: [f64; 3], // km / s
}

/// Represents a loaded SPK segment.
pub struct SpkSegment {
    pub target: i32,
    pub center: i32,
    pub frame: i32,
    pub spk_type: i32,
    pub start_jed: f64,
    pub end_jed: f64,
    
    // Extracted from the f64 slice
    pub data: Vec<f64>,
}

pub struct SpkReader {
    #[expect(dead_code)]
    daf: DafReader,
    segments: Vec<SpkSegment>,
    // Adjacency list for coordinate graph resolution: node -> list of (neighbor, segment_idx, is_forward)
    graph: HashMap<i32, Vec<(i32, usize, bool)>>,
}

impl SpkReader {
    pub fn new(daf: DafReader) -> std::io::Result<Self> {
        let mut segments = Vec::new();
        let mut graph: HashMap<i32, Vec<(i32, usize, bool)>> = HashMap::new();

        for (idx, array) in daf.arrays().enumerate() {
            if array.doubles.len() < 2 || array.integers.len() < 6 {
                continue; // Invalid SPK summary
            }

            let start_jed = array.doubles[0];
            let end_jed = array.doubles[1];

            let target = array.integers[0];
            let center = array.integers[1];
            let frame = array.integers[2];
            let spk_type = array.integers[3];
            let start_word = array.integers[4] as usize;
            let end_word = array.integers[5] as usize;

            let data = daf.read_f64_slice(start_word, end_word)?.to_vec();

            segments.push(SpkSegment {
                target,
                center,
                frame,
                spk_type,
                start_jed,
                end_jed,
                data,
            });

            graph.entry(center).or_default().push((target, idx, true));
            graph.entry(target).or_default().push((center, idx, false));
        }

        Ok(Self { daf, segments, graph })
    }

    pub fn segments(&self) -> &[SpkSegment] {
        &self.segments
    }

    /// Evaluates the state vector (position, velocity) for a specific segment at a given JED.
    fn evaluate_segment(&self, seg_idx: usize, jed: f64) -> Option<StateVector> {
        let seg = &self.segments[seg_idx];
        if jed < seg.start_jed || jed > seg.end_jed {
            return None;
        }

        if seg.spk_type == 2 {
            // Type 2: Chebyshev polynomials for position only.
            // Record structure:
            // [ MID, RADIUS, X_coeffs..., Y_coeffs..., Z_coeffs... ] x N records
            // Followed by metadata: [INIT, INTVL, RSIZE, N_RECORDS]
            let n = seg.data.len();
            if n < 4 { return None; }
            let rsize = seg.data[n - 2] as usize;
            let n_records = seg.data[n - 1] as usize;
            let intvl = seg.data[n - 3];
            let init = seg.data[n - 4];

            // Find the correct record for this time
            // Note: intvl is the length of the time interval for one record (e.g. 32 days).
            // init is the start time of the first record.
            let mut record_idx = ((jed - init) / intvl).floor() as usize;
            if record_idx >= n_records {
                if record_idx == n_records && jed == seg.end_jed {
                    record_idx = n_records - 1;
                } else {
                    return None;
                }
            }

            let rec_start = record_idx * rsize;
            let mid = seg.data[rec_start];
            let radius = seg.data[rec_start + 1];

            // Normalize time to [-1, 1]
            let x = (jed - mid) / radius;

            // Extract coefficients. degree = (rsize - 2) / 3 - 1. But coeffs count per axis is (rsize - 2)/3.
            let coeff_count = (rsize - 2) / 3;
            let mut coeffs = Vec::with_capacity(coeff_count);
            for i in 0..coeff_count {
                coeffs.push(f64x4::new([
                    seg.data[rec_start + 2 + i],
                    seg.data[rec_start + 2 + coeff_count + i],
                    seg.data[rec_start + 2 + 2 * coeff_count + i],
                    0.0,
                ]));
            }

            let pos_simd = evaluate_chebyshev_position(x, &coeffs);
            let vel_simd = evaluate_chebyshev_velocity(x, &coeffs);

            let pos = pos_simd.to_array();
            let vel_norm = vel_simd.to_array();

            // Velocity must be divided by radius, and then converted from km/day to km/sec (usually radius is in days)
            // SPK times are in seconds past J2000 for type 2 in modern ephemerides!
            // Wait, DE440 is actually in seconds, not days.
            let _sec_per_day = 86400.0;
            // Radius in type 2 is half the interval in SECONDS.
            
            return Some(StateVector {
                position: [pos[0], pos[1], pos[2]],
                velocity: [
                    vel_norm[0] / radius,
                    vel_norm[1] / radius,
                    vel_norm[2] / radius,
                ],
            });
        }
        
        None
    }

    /// Evaluates the path from `center` to `target` at `jed`.
    /// Uses BFS to find the shortest path in the coordinate graph.
    pub fn compute_state(&self, center: i32, target: i32, jed: f64) -> Option<StateVector> {
        if center == target {
            return Some(StateVector { position: [0.0; 3], velocity: [0.0; 3] });
        }

        let mut queue = std::collections::VecDeque::new();
        let mut visited = HashMap::new();
        
        queue.push_back(center);
        visited.insert(center, None);

        while let Some(curr) = queue.pop_front() {
            if curr == target {
                break;
            }

            if let Some(neighbors) = self.graph.get(&curr) {
                for &(next, seg_idx, is_fwd) in neighbors {
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(next) {
                        // Check if segment is valid for this time
                        let seg = &self.segments[seg_idx];
                        if jed >= seg.start_jed && jed <= seg.end_jed {
                            e.insert(Some((curr, seg_idx, is_fwd)));
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        if !visited.contains_key(&target) {
            return None; // No path found for this time
        }

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        let mut curr = target;

        // Traverse backwards from target to center
        while curr != center {
            let (prev, seg_idx, is_fwd) = visited[&curr].unwrap();
            
            if let Some(state) = self.evaluate_segment(seg_idx, jed) {
                let sign = if is_fwd { 1.0 } else { -1.0 };
                pos[0] += sign * state.position[0];
                pos[1] += sign * state.position[1];
                pos[2] += sign * state.position[2];
                vel[0] += sign * state.velocity[0];
                vel[1] += sign * state.velocity[1];
                vel[2] += sign * state.velocity[2];
            } else {
                return None;
            }
            
            curr = prev;
        }

        Some(StateVector { position: pos, velocity: vel })
    }
}
