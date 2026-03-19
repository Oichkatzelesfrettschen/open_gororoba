/// Standard Hellings-Downs correlation curve for stochastic gravitational wave background.
pub fn hellings_downs(separation_rad: f64) -> f64 {
    let x = (1.0 - separation_rad.cos()) / 2.0;
    if x <= 0.0 {
        0.5
    } else {
        1.5 * x * x.ln() - 0.25 * x + 0.5
    }
}

/// Angular separation between two 3D unit vectors in radians.
pub fn angular_separation(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot.clamp(-1.0, 1.0).acos()
}
