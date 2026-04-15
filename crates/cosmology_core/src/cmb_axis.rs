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
    pub fn project_axis(&self) -> [f64; 3] {
        let mut axis = [0.0_f64; 3];
        for (i, &b) in self.bias_256d.iter().enumerate() {
            let t = i as f64;
            let px = (t * std::f64::consts::FRAC_1_SQRT_2).cos();
            let py = (t * std::f64::consts::E).sin();
            let pz = (t * std::f64::consts::PI).cos();
            axis[0] += b * px;
            axis[1] += b * py;
            axis[2] += b * pz;
        }
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 0.0 {
            [axis[0] / norm, axis[1] / norm, axis[2] / norm]
        } else {
            [1.0, 0.0, 0.0]
        }
    }
}
