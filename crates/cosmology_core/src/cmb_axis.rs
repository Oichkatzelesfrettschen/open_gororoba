use nalgebra::Vector3;

/// Analyzes 256D Voudon Global Bias vs CMB Axis of Evil alignment.
pub struct VoudonCmbAnalyzer {
    pub bias_256d: [f64; 256],
}

impl VoudonCmbAnalyzer {
    pub fn new(phi: f64) -> Self {
        let mut bias = [0.0; 256];
        for (i, b) in bias.iter_mut().enumerate() {
            *b = (i as f64 * phi).sin();
        }
        Self { bias_256d: bias }
    }

    /// Projects the 256D Voudon bias onto the 3D CMB multipole space.
    pub fn project_axis(&self) -> Vector3<f64> {
        let mut axis = Vector3::zeros();
        for (i, &b) in self.bias_256d.iter().enumerate() {
            let t = i as f64;
            let px = (t * std::f64::consts::FRAC_1_SQRT_2).cos();
            let py = (t * std::f64::consts::E).sin();
            let pz = (t * std::f64::consts::PI).cos();
            axis.x += b * px;
            axis.y += b * py;
            axis.z += b * pz;
        }
        axis.normalize()
    }
}
