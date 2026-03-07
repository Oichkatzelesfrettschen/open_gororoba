use crate::metric::{SpacetimeMetric, MetricComponents, DIM, R};

/// A scale-invariant fractal spacetime metric wrapper.
///
/// It modifies an underlying metric by introducing scale-dependent 
/// scaling laws characterized by a fractal dimension D_f.
pub struct FractalMetric<M: SpacetimeMetric> {
    pub base_metric: M,
    /// Fractal dimension (e.g. 2.7 for unified flyby/pioneer anomaly).
    pub d_f: f64,
    /// Reference scale r_0 (km).
    pub r_0: f64,
}

impl<M: SpacetimeMetric> FractalMetric<M> {
    pub fn new(base_metric: M, d_f: f64, r_0: f64) -> Self {
        Self { base_metric, d_f, r_0 }
    }
}

impl<M: SpacetimeMetric> SpacetimeMetric for FractalMetric<M> {
    fn metric_components(&self, x: &[f64; DIM]) -> MetricComponents {
        let mut g = self.base_metric.metric_components(x);
        let r_val = x[R];
        
        if r_val > 1e-10 {
            // Apply scale-invariant fractal scaling to the radial and angular components.
            // The scaling factor (r/r_0)^(D_f - 4) shifts the effective volume 
            // from 4D to D_f.
            let scale = (r_val / self.r_0).powf(self.d_f - 4.0);
            
            // Modify space-like components
            for i in 1..DIM {
                g[i][i] *= scale;
            }
        }
        
        g
    }

    fn event_horizon_radius(&self) -> Option<f64> {
        self.base_metric.event_horizon_radius()
    }
}
