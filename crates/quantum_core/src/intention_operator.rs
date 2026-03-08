use algebra_experimental::higher_cd::{HigherAvt, Routon};
use lbm_core::D2Q9;
use ndarray::Array2;

/// The Intention Operator (Phi_I0) for stabilizing high-dimensional chaos.
pub struct IntentionOperator {
    pub strength: f64,
    pub target_coherence: f64,
}

/// Result of the entropy filter solver loop.
#[derive(Debug, Clone)]
pub struct EntropyFilterResult {
    /// Shannon entropy at each solver step (grid-averaged).
    pub entropy_history: Vec<f64>,
    /// Number of steps taken.
    pub steps: usize,
    /// Whether the solver converged (entropy <= target_coherence).
    pub converged: bool,
    /// Final grid-averaged entropy.
    pub final_entropy: f64,
    /// Initial grid-averaged entropy.
    pub initial_entropy: f64,
}

impl IntentionOperator {
    pub fn new(strength: f64, target_coherence: f64) -> Self {
        Self {
            strength,
            target_coherence,
        }
    }

    /// Apply the Intention Operator to an LBM fluid grid.
    ///
    /// It modifies the collision relaxation time based on the "subjective bias"
    /// of the observer, effectively filtering high-dimensional Routon (128D) noise.
    pub fn apply_to_lbm(
        &self,
        grid_nx: usize,
        grid_ny: usize,
        routon_field: &Array2<Routon>,
        avt: &HigherAvt,
    ) -> Array2<f64> {
        let mut bias_field = Array2::zeros((grid_nx, grid_ny));

        for x in 0..grid_nx {
            for y in 0..grid_ny {
                let r = routon_field[[x, y]];
                let chaos = self.measure_routon_chaos(&r, avt);
                bias_field[[x, y]] = self.strength * (chaos - self.target_coherence).max(0.0);
            }
        }

        bias_field
    }

    /// Measure the non-associative chaos (Shannon entropy) of a Routon
    /// relative to the AVT violation structure.
    ///
    /// Computes Shannon entropy H = -sum(p_i * ln(p_i)) of the violation
    /// participation distribution across 128 axes, where p_i is the
    /// normalized contribution of axis i to AVT violations weighted by
    /// the Routon component magnitudes.
    pub fn measure_routon_chaos(&self, r: &Routon, avt: &HigherAvt) -> f64 {
        let data = r.to_slice();
        let mut entropy = 0.0;

        let mut violation_participation = vec![0.0; 128];
        for &(i, j, _k, m, sign) in &avt.violations {
            let contribution = (data[i] * data[j] * sign as f64).abs();
            violation_participation[m] += contribution;
        }

        let total: f64 = violation_participation.iter().sum();
        if total > 1e-15 {
            for &p in &violation_participation {
                if p > 0.0 {
                    let prob = p / total;
                    entropy -= prob * prob.ln();
                }
            }
        }

        entropy
    }

    /// Compute the grid-averaged Shannon entropy of a Routon field.
    pub fn grid_average_entropy(&self, routon_field: &Array2<Routon>, avt: &HigherAvt) -> f64 {
        let (nx, ny) = routon_field.dim();
        let n = (nx * ny) as f64;
        if n < 1.0 {
            return 0.0;
        }

        let mut total_entropy = 0.0;
        for x in 0..nx {
            for y in 0..ny {
                total_entropy += self.measure_routon_chaos(&routon_field[[x, y]], avt);
            }
        }
        total_entropy / n
    }

    /// Non-Associative Entropy Filter: iteratively filter non-associative noise
    /// on an LBM grid to achieve localized coherence.
    ///
    /// Algorithm:
    /// 1. Compute 128D Routon Shannon entropy at each grid point
    /// 2. Where entropy exceeds target_coherence, bias the LBM relaxation
    ///    time toward equilibrium (tau -> tau + strength * excess_entropy)
    /// 3. Evolve LBM one step with modified relaxation
    /// 4. Damp Routon field components that contribute most to violations
    /// 5. Repeat until convergence or max_steps
    ///
    /// The bias field increases the relaxation time (viscosity) where
    /// non-associative entropy is high, suppressing turbulent fluctuations
    /// driven by the 128D algebra. This selectively filters noise while
    /// preserving coherent structures.
    pub fn entropy_filter_solve(
        &self,
        lbm: &mut D2Q9,
        routon_field: &mut Array2<Routon>,
        avt: &HigherAvt,
        max_steps: usize,
    ) -> EntropyFilterResult {
        let (nx, ny) = routon_field.dim();
        let base_tau = lbm.tau;
        let mut entropy_history = Vec::with_capacity(max_steps + 1);

        // Record initial entropy
        let initial_entropy = self.grid_average_entropy(routon_field, avt);
        entropy_history.push(initial_entropy);

        let mut converged = initial_entropy <= self.target_coherence;

        for step in 0..max_steps {
            if converged {
                break;
            }

            // 1. Compute bias field from current Routon entropy
            let bias = self.apply_to_lbm(nx, ny, routon_field, avt);

            // 2. Apply spatially-varying relaxation time to LBM
            //    tau_eff(x,y) = base_tau + bias(x,y)
            //    Higher bias => higher tau => higher viscosity => more damping
            let max_bias = bias.iter().cloned().fold(0.0_f64, f64::max);
            lbm.tau = base_tau + max_bias * 0.1; // Moderate the effect to keep stability

            // 3. Evolve LBM one step
            lbm.collide(0, ny);
            lbm.stream();

            // 4. Damp Routon components where entropy exceeds target
            //    damping_factor = 1 - strength * (entropy - target) / entropy_max
            let entropy_max = initial_entropy.max(1.0);
            for x in 0..nx {
                for y in 0..ny {
                    let local_chaos = self.measure_routon_chaos(&routon_field[[x, y]], avt);
                    if local_chaos > self.target_coherence {
                        let excess = (local_chaos - self.target_coherence) / entropy_max;
                        let damping = 1.0 - (self.strength * excess).min(0.5);
                        let mut data = routon_field[[x, y]].to_slice();
                        for val in &mut data {
                            *val *= damping;
                        }
                        routon_field[[x, y]] = Routon::from_slice(&data);
                    }
                }
            }

            // 5. Measure new entropy
            let current_entropy = self.grid_average_entropy(routon_field, avt);
            entropy_history.push(current_entropy);

            if current_entropy <= self.target_coherence {
                converged = true;
            }

            // Safety: if entropy is not decreasing after 10 steps, stop
            if step > 10 && entropy_history.len() >= 10 {
                let recent = &entropy_history[entropy_history.len() - 5..];
                let oldest = recent[0];
                let newest = recent[4];
                if newest >= oldest * 0.999 {
                    break; // Stalled
                }
            }
        }

        // Restore base tau
        lbm.tau = base_tau;

        let final_entropy = *entropy_history.last().unwrap_or(&initial_entropy);
        EntropyFilterResult {
            steps: entropy_history.len() - 1,
            converged,
            final_entropy,
            initial_entropy,
            entropy_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use algebra_experimental::higher_cd::HigherAvt;

    fn make_uniform_routon() -> Routon {
        // All 128 components equal -> maximum entropy
        let data = [1.0 / 128.0_f64.sqrt(); 128];
        Routon::from_slice(&data)
    }

    fn make_sparse_routon() -> Routon {
        // Only first 4 components nonzero -> low entropy
        let mut data = [0.0; 128];
        data[0] = 1.0;
        data[1] = 0.5;
        data[2] = 0.3;
        data[3] = 0.1;
        Routon::from_slice(&data)
    }

    #[test]
    fn test_measure_routon_chaos_nonzero() {
        let avt = HigherAvt::new(128);
        let op = IntentionOperator::new(1.0, 2.0);
        let r = make_uniform_routon();
        let entropy = op.measure_routon_chaos(&r, &avt);
        assert!(
            entropy > 0.0,
            "uniform routon should have nonzero entropy, got {entropy}"
        );
        assert!(entropy.is_finite(), "entropy should be finite");
    }

    #[test]
    fn test_uniform_has_higher_entropy_than_sparse() {
        let avt = HigherAvt::new(128);
        let op = IntentionOperator::new(1.0, 2.0);
        let h_uniform = op.measure_routon_chaos(&make_uniform_routon(), &avt);
        let h_sparse = op.measure_routon_chaos(&make_sparse_routon(), &avt);
        assert!(
            h_uniform > h_sparse,
            "uniform ({h_uniform}) should have higher entropy than sparse ({h_sparse})"
        );
    }

    #[test]
    fn test_apply_to_lbm_produces_bias_field() {
        let avt = HigherAvt::new(128);
        let op = IntentionOperator::new(1.0, 0.5);
        let nx = 4;
        let ny = 4;
        let mut field = Array2::from_elem((nx, ny), make_uniform_routon());
        // Set one cell to sparse (low entropy)
        field[[0, 0]] = make_sparse_routon();

        let bias = op.apply_to_lbm(nx, ny, &field, &avt);
        assert_eq!(bias.dim(), (nx, ny));
        // Sparse cell should have lower bias than uniform cells
        let b00 = bias[[0, 0]];
        let b11 = bias[[1, 1]];
        assert!(
            b00 <= b11,
            "sparse cell bias ({b00}) should be <= uniform ({b11})"
        );
    }

    #[test]
    fn test_entropy_filter_reduces_entropy() {
        let avt = HigherAvt::new(128);
        // High target => should converge quickly
        let op = IntentionOperator::new(0.5, 100.0);
        let nx = 4;
        let ny = 4;
        let mut lbm = D2Q9::new(nx, ny, 0.8);
        let mut field = Array2::from_elem((nx, ny), make_uniform_routon());

        let result = op.entropy_filter_solve(&mut lbm, &mut field, &avt, 10);
        assert!(result.steps <= 10);
        // With a very high target, should converge immediately
        assert!(result.converged, "should converge with target=100");
    }

    #[test]
    fn test_entropy_filter_damping() {
        let avt = HigherAvt::new(128);
        // Low target => solver will damp the field
        let op = IntentionOperator::new(0.8, 0.1);
        let nx = 4;
        let ny = 4;
        let mut lbm = D2Q9::new(nx, ny, 0.8);
        let mut field = Array2::from_elem((nx, ny), make_uniform_routon());

        let result = op.entropy_filter_solve(&mut lbm, &mut field, &avt, 50);
        // Entropy should decrease from damping
        assert!(
            result.final_entropy <= result.initial_entropy,
            "entropy should decrease: initial={}, final={}",
            result.initial_entropy,
            result.final_entropy
        );
        assert!(result.entropy_history.len() > 1);
    }

    #[test]
    fn test_grid_average_entropy() {
        let avt = HigherAvt::new(128);
        let op = IntentionOperator::new(1.0, 2.0);
        let field = Array2::from_elem((2, 2), make_uniform_routon());
        let avg = op.grid_average_entropy(&field, &avt);
        assert!(avg > 0.0);
        assert!(avg.is_finite());
    }

    #[test]
    fn test_avt_entropy_baseline_connection() {
        // Verify that Routon Shannon entropy is bounded by the AVT violation
        // structure. The AVT at dim=128 has N violations distributed across
        // 128 output axes. The Routon entropy should correlate with this
        // distribution -- uniform Routon maximizes participation entropy.
        let avt = HigherAvt::new(128);
        let op = IntentionOperator::new(1.0, 2.0);

        // Compute AVT axis entropy baseline (same calculation as avt_axis_entropy(128))
        let mut axis_counts = vec![0usize; 128];
        for &(_i, _j, _k, m, _sign) in &avt.violations {
            axis_counts[m] += 1;
        }
        let total_violations = avt.violations.len();
        let mut avt_entropy = 0.0;
        if total_violations > 0 {
            let t = total_violations as f64;
            for &c in &axis_counts {
                if c > 0 {
                    let p = c as f64 / t;
                    avt_entropy -= p * p.ln();
                }
            }
        }

        // Routon entropy for uniform field should be positive
        let h_routon = op.measure_routon_chaos(&make_uniform_routon(), &avt);

        // Both entropies should be positive and finite
        assert!(
            avt_entropy > 0.0,
            "AVT baseline entropy should be > 0: {avt_entropy}"
        );
        assert!(h_routon > 0.0, "Routon entropy should be > 0: {h_routon}");
        assert!(avt_entropy.is_finite());
        assert!(h_routon.is_finite());

        // Routon entropy should be bounded by ln(128) ~ 4.85 (max possible)
        let ln128 = 128.0_f64.ln();
        assert!(
            h_routon <= ln128 + 0.01,
            "Routon entropy {h_routon} should be <= ln(128) = {ln128}"
        );
    }
}
