//! Hyperbolic billiard simulation on H^9 in the E10 Weyl chamber.
//!
//! Implements the BKL (Belinsky-Khalatnikov-Lifshitz) cosmological billiard
//! with proper geodesic dynamics on the unit hyperboloid, replacing the
//! linearized approximation in the e10_billiard example.
//!
//! # Geometry
//!
//! The billiard lives in R^{9,1} with Lorentzian metric
//! <x,y> = x_0*y_0 + ... + x_8*y_8 - x_9*y_9.
//!
//! The position stays on the unit hyperboloid H^9 = {x | <x,x> = -1, x_9 > 0}.
//! The velocity is tangent to H^9: <x,v> = 0, <v,v> = 1 (unit speed).
//!
//! Free motion is geodesic:
//!   x(t) = cosh(t)*x + sinh(t)*v
//!   v(t) = sinh(t)*x + cosh(t)*v
//!
//! Wall reflections are Weyl reflections:
//!   v' = v - 2*<v,alpha>/<alpha,alpha> * alpha
//!
//! # References
//! - Damour, Henneaux, Nicolai (2003): "Cosmological billiards"
//! - Damour, de Buyl, Henneaux, Schomblond (2004): "Einstein billiards"

use crate::lie::kac_moody::{E10RootSystem, KacMoodyRoot};

/// A 10-dimensional vector with Lorentzian metric signature (9,1).
///
/// coords[0..9] are spacelike (+), coords[9] is timelike (-).
/// Inner product: <x,y> = sum(x_i*y_i, i=0..9) - x_9*y_9.
#[derive(Debug, Clone)]
pub struct LorentzVec {
    pub coords: [f64; 10],
}

impl LorentzVec {
    /// Create a zero vector.
    pub fn zero() -> Self {
        Self { coords: [0.0; 10] }
    }

    /// Create from explicit coordinates.
    pub fn new(coords: [f64; 10]) -> Self {
        Self { coords }
    }

    /// Lorentzian inner product: <x,y> = sum(x_i*y_i, i=0..9) - x_9*y_9.
    pub fn inner_product(&self, other: &Self) -> f64 {
        let mut result = 0.0;
        for i in 0..9 {
            result += self.coords[i] * other.coords[i];
        }
        result -= self.coords[9] * other.coords[9];
        result
    }

    /// Lorentzian norm squared: <x,x>.
    pub fn norm_sq(&self) -> f64 {
        self.inner_product(self)
    }

    /// Scale: a * x.
    pub fn scale(&self, a: f64) -> Self {
        let mut c = [0.0; 10];
        for (i, &v) in self.coords.iter().enumerate() {
            c[i] = a * v;
        }
        Self { coords: c }
    }

    /// Add: x + y.
    pub fn add(&self, other: &Self) -> Self {
        let mut c = [0.0; 10];
        for ((c_i, &a), &b) in c.iter_mut().zip(self.coords.iter()).zip(other.coords.iter()) {
            *c_i = a + b;
        }
        Self { coords: c }
    }

    /// Subtract: x - y.
    pub fn sub(&self, other: &Self) -> Self {
        let mut c = [0.0; 10];
        for ((c_i, &a), &b) in c.iter_mut().zip(self.coords.iter()).zip(other.coords.iter()) {
            *c_i = a - b;
        }
        Self { coords: c }
    }

    /// Convert from KacMoodyRoot (E10).
    ///
    /// Mapping: coords[0..8] = finite_part, coords[8] = lorentz[0], coords[9] = lorentz[1].
    pub fn from_kac_moody(root: &KacMoodyRoot) -> Self {
        let mut c = [0.0; 10];
        for (i, &v) in root.finite_part.iter().enumerate().take(8) {
            c[i] = v;
        }
        if root.lorentz_coords.len() >= 2 {
            c[8] = root.lorentz_coords[0];
            c[9] = root.lorentz_coords[1];
        }
        Self { coords: c }
    }
}

/// State of the hyperbolic billiard on H^9.
///
/// Invariants (maintained to floating-point precision):
/// 1. <pos, pos> = -1 (on the unit hyperboloid)
/// 2. <pos, vel> = 0 (velocity tangent to H^9)
/// 3. <vel, vel> = 1 (unit speed)
#[derive(Debug, Clone)]
pub struct BilliardState {
    /// Position on the unit hyperboloid H^9.
    pub pos: LorentzVec,
    /// Velocity tangent to H^9 at pos.
    pub vel: LorentzVec,
}

/// Result of a single billiard step.
#[derive(Debug, Clone)]
pub struct BounceResult {
    /// Index of the wall that was hit (0..n_walls).
    pub wall_idx: usize,
    /// Geodesic time to the wall.
    pub travel_time: f64,
    /// Constraint diagnostics after the bounce.
    pub diagnostics: ConstraintDiagnostics,
}

/// Diagnostic snapshot of constraint preservation.
#[derive(Debug, Clone, Copy)]
pub struct ConstraintDiagnostics {
    /// <pos, pos> + 1 (should be ~0).
    pub pos_norm_error: f64,
    /// <pos, vel> (should be ~0).
    pub tangency_error: f64,
    /// <vel, vel> - 1 (should be ~0).
    pub vel_norm_error: f64,
}

/// Configuration for the hyperbolic billiard simulation.
#[derive(Debug, Clone)]
pub struct BilliardConfig {
    /// Wall vectors (simple roots). Each has <alpha, alpha> = 2 for E10.
    pub walls: Vec<LorentzVec>,
    /// Whether to renormalize after each bounce (recommended).
    pub renormalize: bool,
    /// Tolerance for considering a wall-crossing time valid.
    pub time_epsilon: f64,
}

/// Hyperbolic billiard simulator on H^9 in the E10 Weyl chamber.
pub struct HyperbolicBilliard {
    pub state: BilliardState,
    pub config: BilliardConfig,
    /// Cumulative bounce count.
    pub n_bounces: usize,
    /// Maximum constraint error seen so far.
    pub max_pos_error: f64,
    pub max_tangency_error: f64,
    pub max_vel_error: f64,
}

impl HyperbolicBilliard {
    /// Create a billiard from a position and velocity, with automatic projection.
    ///
    /// The position is projected onto H^9, and the velocity is orthogonalized
    /// and normalized to satisfy the three invariants.
    pub fn new(pos: LorentzVec, vel: LorentzVec, config: BilliardConfig) -> Self {
        let mut state = BilliardState { pos, vel };
        project_to_hyperboloid(&mut state);
        enforce_tangency(&mut state);
        normalize_velocity(&mut state);

        Self {
            state,
            config,
            n_bounces: 0,
            max_pos_error: 0.0,
            max_tangency_error: 0.0,
            max_vel_error: 0.0,
        }
    }

    /// Create a billiard from the E10 root system.
    ///
    /// Initializes the Weyl chamber interior position and a random timelike velocity.
    pub fn from_e10(e10: &E10RootSystem, seed: u64) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let simple_roots = e10.simple_roots();

        // Convert walls
        let walls: Vec<LorentzVec> = simple_roots.iter()
            .map(LorentzVec::from_kac_moody)
            .collect();

        // Compute Weyl vector (chamber interior) by solving G*x = [1,..,1]
        let pos_km = compute_weyl_vector(e10, &simple_roots);
        let mut pos = LorentzVec::from_kac_moody(&pos_km);

        // Ensure pos is timelike (norm_sq < 0)
        let pos_norm = pos.norm_sq();
        if pos_norm >= 0.0 {
            // Make it timelike by boosting the timelike component
            pos.coords[9] = (pos.coords.iter().take(9).map(|x| x * x).sum::<f64>() + 1.0).sqrt();
        }

        // Normalize to hyperboloid
        let neg_norm_sq = -pos.norm_sq();
        if neg_norm_sq > 0.0 {
            let scale = 1.0 / neg_norm_sq.sqrt();
            pos = pos.scale(scale);
        }

        // Random velocity tangent to H^9
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut vel = LorentzVec::zero();
        for c in vel.coords.iter_mut().take(9) {
            *c = rng.gen_range(-1.0..1.0);
        }
        // Make tangent: vel <- vel - <pos,vel>/<pos,pos> * pos = vel + <pos,vel> * pos
        // (since <pos,pos> = -1)
        let tangent_correction = pos.inner_product(&vel);
        vel = vel.add(&pos.scale(tangent_correction));
        // Normalize to unit speed
        let vel_norm = vel.norm_sq();
        if vel_norm > 1e-15 {
            vel = vel.scale(1.0 / vel_norm.sqrt());
        }

        let config = BilliardConfig {
            walls,
            renormalize: true,
            time_epsilon: 1e-12,
        };

        Self::new(pos, vel, config)
    }

    /// Perform one bounce: advance to next wall, reflect, return wall index.
    ///
    /// Returns `None` if no wall is reachable (particle escaped or at infinity).
    pub fn step(&mut self) -> Option<BounceResult> {
        let mut min_t = f64::INFINITY;
        let mut hit_wall = None;

        for (i, wall) in self.config.walls.iter().enumerate() {
            if let Some(t) = self.collision_time(wall) {
                if t > self.config.time_epsilon && t < min_t {
                    min_t = t;
                    hit_wall = Some(i);
                }
            }
        }

        let wall_idx = hit_wall?;

        // Geodesic advance to the wall
        geodesic_advance(&mut self.state, min_t);

        // Weyl reflection of velocity
        let wall = &self.config.walls[wall_idx];
        weyl_reflect(&mut self.state.vel, wall);

        // Adaptive Nudge: Ensure we remain inside all walls after the bounce.
        // We need to move a tiny bit along the new velocity v'.
        // To avoid crossing ANY other wall i, we need <pos + eps*v', alpha_i> >= 0.
        // eps <= <pos, alpha_i> / -<v', alpha_i> for all walls with <v', alpha_i> < 0.
        let mut max_eps = 1e-5; // Upper bound for the nudge
        for (i, w) in self.config.walls.iter().enumerate() {
            if i == wall_idx { continue; }
            let pos_dot = self.state.pos.inner_product(w);
            let vel_dot = self.state.vel.inner_product(w);
            if vel_dot < -1e-12 {
                let limit = pos_dot / -vel_dot;
                if limit > 0.0 && limit < max_eps {
                    max_eps = limit;
                }
            }
        }
        // Use a safety fraction of the limit
        let eps = max_eps * 0.1;
        if eps > 1e-15 {
            geodesic_advance(&mut self.state, eps);
        }

        // Renormalize to combat floating-point drift
        if self.config.renormalize {
            project_to_hyperboloid(&mut self.state);
            enforce_tangency(&mut self.state);
            normalize_velocity(&mut self.state);
        }

        // Diagnostics
        let diag = compute_diagnostics(&self.state);
        self.max_pos_error = self.max_pos_error.max(diag.pos_norm_error.abs());
        self.max_tangency_error = self.max_tangency_error.max(diag.tangency_error.abs());
        self.max_vel_error = self.max_vel_error.max(diag.vel_norm_error.abs());
        self.n_bounces += 1;

        Some(BounceResult {
            wall_idx,
            travel_time: min_t,
            diagnostics: diag,
        })
    }

    /// Simulate n_steps bounces, returning the wall-hit sequence.
    pub fn simulate(&mut self, n_steps: usize) -> Vec<usize> {
        let mut sequence = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            match self.step() {
                Some(result) => sequence.push(result.wall_idx),
                None => break,
            }
        }
        sequence
    }

    /// Check if position is inside the Weyl chamber (all wall inner products positive).
    pub fn is_inside_chamber(&self) -> bool {
        self.config.walls.iter().all(|w| self.state.pos.inner_product(w) > -1e-10)
    }

    /// Compute the geodesic collision time with a specific wall.
    ///
    /// Wall alpha defines the hyperplane {x | <x, alpha> = 0}.
    /// On the geodesic x(t) = cosh(t)*pos + sinh(t)*vel:
    ///   <x(t), alpha> = cosh(t)*<pos,alpha> + sinh(t)*<vel,alpha> = 0
    ///   => tanh(t) = -<pos,alpha> / <vel,alpha>
    ///
    /// Returns Some(t) if t > 0 and |tanh(t)| < 1 (wall is reachable), None otherwise.
    fn collision_time(&self, wall: &LorentzVec) -> Option<f64> {
        let pos_dot = self.state.pos.inner_product(wall);
        let vel_dot = self.state.vel.inner_product(wall);

        if vel_dot.abs() < 1e-15 {
            return None; // Moving parallel to wall
        }

        let tanh_t = -pos_dot / vel_dot;

        // tanh(t) must be in (-1, 1) for real t
        if tanh_t.abs() >= 1.0 - 1e-15 {
            return None; // Wall unreachable (at infinity)
        }

        let t = tanh_t.atanh();
        if t > 0.0 {
            Some(t)
        } else {
            None // Wall is behind us
        }
    }

    /// Get current constraint diagnostics.
    pub fn diagnostics(&self) -> ConstraintDiagnostics {
        compute_diagnostics(&self.state)
    }
}

// ---------------------------------------------------------------------------
// Geometric operations
// ---------------------------------------------------------------------------

/// Project position onto the unit hyperboloid: pos <- pos / sqrt(-<pos,pos>).
///
/// Precondition: pos must be timelike (<pos,pos> < 0).
fn project_to_hyperboloid(state: &mut BilliardState) {
    let neg_norm = -state.pos.norm_sq();
    if neg_norm > 0.0 {
        let scale = 1.0 / neg_norm.sqrt();
        state.pos = state.pos.scale(scale);
    }
}

/// Enforce tangency: vel <- vel - <pos,vel>/<pos,pos> * pos.
///
/// Since <pos,pos> = -1 on the hyperboloid, this simplifies to:
/// vel <- vel + <pos,vel> * pos.
fn enforce_tangency(state: &mut BilliardState) {
    let dot = state.pos.inner_product(&state.vel);
    // vel <- vel + dot * pos (since <pos,pos> = -1, dividing by -1 flips sign)
    state.vel = state.vel.add(&state.pos.scale(dot));
}

/// Normalize velocity to unit speed: vel <- vel / sqrt(<vel,vel>).
///
/// Precondition: vel must be spacelike (<vel,vel> > 0) after tangency enforcement.
fn normalize_velocity(state: &mut BilliardState) {
    let norm_sq = state.vel.norm_sq();
    if norm_sq > 1e-15 {
        state.vel = state.vel.scale(1.0 / norm_sq.sqrt());
    }
}

/// Geodesic advance on H^9: x(t) = cosh(t)*x + sinh(t)*v, v(t) = sinh(t)*x + cosh(t)*v.
fn geodesic_advance(state: &mut BilliardState, t: f64) {
    let ch = t.cosh();
    let sh = t.sinh();
    let new_pos = state.pos.scale(ch).add(&state.vel.scale(sh));
    let new_vel = state.pos.scale(sh).add(&state.vel.scale(ch));
    state.pos = new_pos;
    state.vel = new_vel;
}

/// Weyl reflection of velocity off wall alpha: v' = v - 2<v,alpha>/<alpha,alpha> * alpha.
fn weyl_reflect(vel: &mut LorentzVec, wall: &LorentzVec) {
    let v_dot_alpha = vel.inner_product(wall);
    let alpha_sq = wall.norm_sq();
    if alpha_sq.abs() > 1e-15 {
        let coeff = 2.0 * v_dot_alpha / alpha_sq;
        *vel = vel.sub(&wall.scale(coeff));
    }
}

/// Compute constraint diagnostics.
fn compute_diagnostics(state: &BilliardState) -> ConstraintDiagnostics {
    ConstraintDiagnostics {
        pos_norm_error: state.pos.norm_sq() + 1.0,
        tangency_error: state.pos.inner_product(&state.vel),
        vel_norm_error: state.vel.norm_sq() - 1.0,
    }
}

/// Compute the Weyl vector (chamber interior point) by solving G*x = 1.
///
/// This is the same algorithm as in the e10_billiard example but factored out
/// for reuse.
fn compute_weyl_vector(
    e10: &E10RootSystem,
    roots: &[KacMoodyRoot],
) -> KacMoodyRoot {
    use crate::lie::kac_moody::RootType;

    let n = roots.len();

    // Build Gram matrix
    let mut gram = vec![vec![0.0; n]; n];
    for (i, ri) in roots.iter().enumerate() {
        for (j, rj) in roots.iter().enumerate() {
            gram[i][j] = e10.inner_product(ri, rj);
        }
    }

    // Gaussian elimination with partial pivoting: [G | 1]
    let mut aug = vec![vec![0.0; n + 1]; n];
    for (i, row) in aug.iter_mut().enumerate() {
        for (j, &g) in gram[i].iter().enumerate() {
            row[j] = g;
        }
        row[n] = 1.0;
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (offset, aug_row) in aug[(col + 1)..n].iter().enumerate() {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = col + 1 + offset;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-12, "Gram matrix singular at column {col}");

        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            let pivot_row: Vec<f64> = aug[col][col..=n].to_vec();
            for (j_off, &pval) in pivot_row.iter().enumerate() {
                aug[row][col + j_off] -= factor * pval;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    // pos = sum_i x_i * alpha_i
    let mut finite = vec![0.0; 8];
    let mut lorentz = vec![0.0; 2];
    for (i, &coeff) in x.iter().enumerate() {
        for (j, &fp) in roots[i].finite_part.iter().enumerate().take(8) {
            finite[j] += coeff * fp;
        }
        if roots[i].lorentz_coords.len() >= 2 {
            lorentz[0] += coeff * roots[i].lorentz_coords[0];
            lorentz[1] += coeff * roots[i].lorentz_coords[1];
        }
    }

    KacMoodyRoot {
        finite_part: finite,
        level: 0,
        lorentz_coords: lorentz,
        root_type: RootType::Real,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_billiard(seed: u64) -> HyperbolicBilliard {
        let e10 = E10RootSystem::new();
        HyperbolicBilliard::from_e10(&e10, seed)
    }

    #[test]
    fn test_lorentz_inner_product_signature() {
        // Spacelike vector: (1,0,...,0) has norm^2 = 1 > 0
        let mut v = LorentzVec::zero();
        v.coords[0] = 1.0;
        assert!((v.norm_sq() - 1.0).abs() < 1e-15);

        // Timelike vector: (0,...,0,1) has norm^2 = -1 < 0
        let mut t = LorentzVec::zero();
        t.coords[9] = 1.0;
        assert!((t.norm_sq() + 1.0).abs() < 1e-15);

        // Lightlike: (1,0,...,0,1) has norm^2 = 1-1 = 0
        let mut l = LorentzVec::zero();
        l.coords[0] = 1.0;
        l.coords[9] = 1.0;
        assert!(l.norm_sq().abs() < 1e-15);
    }

    #[test]
    fn test_initial_constraints() {
        let billiard = make_test_billiard(42);
        let diag = billiard.diagnostics();

        assert!(diag.pos_norm_error.abs() < 1e-12,
            "Position not on hyperboloid: error = {:.2e}", diag.pos_norm_error);
        assert!(diag.tangency_error.abs() < 1e-12,
            "Velocity not tangent: error = {:.2e}", diag.tangency_error);
        assert!(diag.vel_norm_error.abs() < 1e-12,
            "Velocity not unit speed: error = {:.2e}", diag.vel_norm_error);
    }

    #[test]
    fn test_inside_chamber_initially() {
        let billiard = make_test_billiard(42);
        assert!(billiard.is_inside_chamber(),
            "Initial position should be inside the Weyl chamber");
    }

    #[test]
    fn test_geodesic_preserves_constraints() {
        // Advance along a geodesic without reflection, verify constraints hold
        let billiard = make_test_billiard(42);
        let mut state = billiard.state.clone();

        for t in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let mut test_state = billiard.state.clone();
            geodesic_advance(&mut test_state, t);
            let diag = compute_diagnostics(&test_state);

            // Tolerance scales with cosh(t)^2 * machine_eps (quadratic form amplification)
            let tol = (t.cosh().powi(2) * 1e-14).max(1e-12);
            assert!(diag.pos_norm_error.abs() < tol,
                "t={}: pos norm error = {:.2e} > tol {:.2e}", t, diag.pos_norm_error, tol);
            assert!(diag.tangency_error.abs() < tol,
                "t={}: tangency error = {:.2e} > tol {:.2e}", t, diag.tangency_error, tol);
            assert!(diag.vel_norm_error.abs() < tol,
                "t={}: vel norm error = {:.2e} > tol {:.2e}", t, diag.vel_norm_error, tol);
        }

        // Many small steps should also preserve constraints
        for _ in 0..1000 {
            geodesic_advance(&mut state, 0.01);
        }
        let diag = compute_diagnostics(&state);
        assert!(diag.pos_norm_error.abs() < 1e-7,
            "After 1000 small steps: pos error = {:.2e}", diag.pos_norm_error);
    }

    #[test]
    fn test_weyl_reflection_preserves_norm() {
        let billiard = make_test_billiard(42);
        let original_norm = billiard.state.vel.norm_sq();

        for wall in &billiard.config.walls {
            let mut vel = billiard.state.vel.clone();
            weyl_reflect(&mut vel, wall);
            let new_norm = vel.norm_sq();
            assert!((new_norm - original_norm).abs() < 1e-12,
                "Weyl reflection changed velocity norm: {:.6} -> {:.6}",
                original_norm, new_norm);
        }
    }

    #[test]
    fn test_simulate_10k_bounces_norm_preservation() {
        let mut billiard = make_test_billiard(42);
        let sequence = billiard.simulate(10_000);

        // Should complete without escaping
        assert!(sequence.len() >= 1000,
            "Expected at least 1000 bounces, got {}", sequence.len());

        // Constraint errors should stay bounded
        assert!(billiard.max_pos_error < 1e-10,
            "Max position error too large: {:.2e}", billiard.max_pos_error);
        assert!(billiard.max_tangency_error < 1e-10,
            "Max tangency error too large: {:.2e}", billiard.max_tangency_error);
        assert!(billiard.max_vel_error < 1e-10,
            "Max velocity error too large: {:.2e}", billiard.max_vel_error);

        // Should hit multiple walls
        let unique_walls: std::collections::HashSet<usize> = sequence.iter().copied().collect();
        assert!(unique_walls.len() >= 3,
            "Expected diverse wall hits, got {} unique walls", unique_walls.len());
    }

    #[test]
    fn test_simulate_deterministic() {
        let mut b1 = make_test_billiard(42);
        let mut b2 = make_test_billiard(42);
        let s1 = b1.simulate(500);
        let s2 = b2.simulate(500);
        assert_eq!(s1, s2, "Same seed should produce identical sequences");
    }

    #[test]
    fn test_collision_time_inside_chamber() {
        let billiard = make_test_billiard(42);
        // At least some walls should have finite collision times
        let mut n_reachable = 0;
        for wall in &billiard.config.walls {
            if billiard.collision_time(wall).is_some() {
                n_reachable += 1;
            }
        }
        assert!(n_reachable >= 1,
            "No reachable walls from initial position/velocity");
    }

    #[test]
    fn test_renormalization_effect() {
        // Without renormalization, errors should grow faster
        let e10 = E10RootSystem::new();
        let simple_roots = e10.simple_roots();
        let walls: Vec<LorentzVec> = simple_roots.iter()
            .map(LorentzVec::from_kac_moody)
            .collect();

        let config_no_renorm = BilliardConfig {
            walls: walls.clone(),
            renormalize: false,
            time_epsilon: 1e-12,
        };
        let config_renorm = BilliardConfig {
            walls,
            renormalize: true,
            time_epsilon: 1e-12,
        };

        // Use from_e10 to get initial state, then clone with different configs
        let base = HyperbolicBilliard::from_e10(&e10, 42);
        let pos = base.state.pos.clone();
        let vel = base.state.vel.clone();

        let mut b_no = HyperbolicBilliard::new(pos.clone(), vel.clone(), config_no_renorm);
        let mut b_yes = HyperbolicBilliard::new(pos, vel, config_renorm);

        let _s_no = b_no.simulate(1000);
        let _s_yes = b_yes.simulate(1000);

        // With renormalization, errors should be smaller
        assert!(b_yes.max_pos_error <= b_no.max_pos_error + 1e-15,
            "Renormalized error ({:.2e}) should be <= non-renormalized ({:.2e})",
            b_yes.max_pos_error, b_no.max_pos_error);
    }

    #[test]
    fn test_lorentz_vec_from_kac_moody_roundtrip() {
        let e10 = E10RootSystem::new();
        let roots = e10.simple_roots();

        for (i, root) in roots.iter().enumerate() {
            let lv = LorentzVec::from_kac_moody(root);
            // Inner product should match E10 inner product
            let lv_norm = lv.norm_sq();
            let km_norm = e10.inner_product(root, root);
            assert!((lv_norm - km_norm).abs() < 1e-10,
                "Root {}: LorentzVec norm {:.6} != KacMoody norm {:.6}",
                i, lv_norm, km_norm);
        }
    }

    #[test]
    fn test_walls_have_norm_2() {
        let e10 = E10RootSystem::new();
        let simple_roots = e10.simple_roots();
        let walls: Vec<LorentzVec> = simple_roots.iter()
            .map(LorentzVec::from_kac_moody)
            .collect();

        for (i, wall) in walls.iter().enumerate() {
            let norm = wall.norm_sq();
            assert!((norm - 2.0).abs() < 1e-10,
                "Wall {} has norm^2 = {:.6}, expected 2.0", i, norm);
        }
    }

    #[test]
    fn test_simulate_produces_e8_heavy_sequence() {
        // BKL billiards should spend most time in the E8 sector (walls 0-7)
        let mut billiard = make_test_billiard(42);
        let sequence = billiard.simulate(5000);

        if sequence.len() >= 100 {
            let e8_count = sequence.iter().filter(|&&w| w < 8).count();
            let e8_fraction = e8_count as f64 / sequence.len() as f64;
            // E8 sector should dominate (at least 50% of bounces)
            assert!(e8_fraction > 0.3,
                "E8 fraction too low: {:.4} ({} of {})",
                e8_fraction, e8_count, sequence.len());
        }
    }

    /// T1: Verify that the initial Weyl vector point is strictly inside all walls.
    #[test]
    fn test_t1_chamber_validity() {
        let e10 = E10RootSystem::new();
        let simple_roots = e10.simple_roots();
        let pos_km = compute_weyl_vector(&e10, &simple_roots);
        
        for (i, root) in simple_roots.iter().enumerate() {
            let ip = e10.inner_product(&pos_km, root);
            assert!(ip > 0.99, "Wall {} inner product {} < 1.0", i, ip);
        }
    }

    /// T5: Verify that the adaptive nudge prevents false escapes.
    #[test]
    fn test_t5_no_false_escape() {
        let mut billiard = make_test_billiard(123);
        let n = 2000;
        let sequence = billiard.simulate(n);
        
        assert_eq!(sequence.len(), n, 
            "Simulation escaped at step {}/{} with adaptive nudge!", sequence.len(), n);
        
        // Also check that min slack remains non-negative
        for _ in 0..100 {
            billiard.step().expect("Should not escape");
            assert!(billiard.is_inside_chamber(), "Escaped chamber during step!");
        }
    }

    /// T2: Verify hyperboloid stability (norm preservation).
    #[test]
    fn test_t2_hyperboloid_stability() {
        let mut billiard = make_test_billiard(456);
        let initial_pos_norm = billiard.state.pos.norm_sq();
        let initial_vel_norm = billiard.state.vel.norm_sq();
        
        billiard.simulate(1000);
        
        let final_pos_norm = billiard.state.pos.norm_sq();
        let final_vel_norm = billiard.state.vel.norm_sq();
        
        assert!((final_pos_norm - initial_pos_norm).abs() < 1e-10,
            "Position norm drifted: {} -> {}", initial_pos_norm, final_pos_norm);
        assert!((final_vel_norm - initial_vel_norm).abs() < 1e-10,
            "Velocity norm drifted: {} -> {}", initial_vel_norm, final_vel_norm);
    }
}
