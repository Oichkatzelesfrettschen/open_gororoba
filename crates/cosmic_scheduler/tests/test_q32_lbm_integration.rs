//! Integration test: LbmDomain<Q32_32> as TwoPhaseSystem backend.
//!
//! Demonstrates the cosmic scheduler driving a fixed-point LBM solver
//! with zero mass drift (Q32.32 integer accumulation is exact).
//! Cross-pollinated from steinmarder CUDA kernel kernels_q32_soa.cu.

use cosmic_scheduler::{ScheduleResult, TwoPhaseClockScheduler, TwoPhaseSystem};
use fixed_point_lbm::{Q32_32, solver::LbmDomain};

/// Adapter wrapping LbmDomain<Q32_32> as a TwoPhaseSystem.
/// Phi1 = collision, Phi2 = streaming.
struct Q32LbmSystem {
    domain: LbmDomain<Q32_32>,
    collision_count: u64,
    streaming_count: u64,
}

impl Q32LbmSystem {
    fn new(n: usize) -> Self {
        Self {
            domain: LbmDomain::<Q32_32>::new(n, n, n, 0.6),
            collision_count: 0,
            streaming_count: 0,
        }
    }
}

impl TwoPhaseSystem for Q32LbmSystem {
    /// Phi1: BGK collision (compute equilibrium, relax distributions).
    fn execute_phase1(&mut self) -> ScheduleResult<()> {
        self.domain.collide();
        self.collision_count += 1;
        Ok(())
    }

    /// Phi2: Streaming (propagate distributions to neighbors).
    fn execute_phase2(&mut self) -> ScheduleResult<()> {
        self.domain.stream();
        self.streaming_count += 1;
        Ok(())
    }

    /// Validate: check mass conservation.
    fn validate_state(&self) -> ScheduleResult<()> {
        let mass = self.domain.total_mass();
        let n = self.domain.n_cells() as f64;
        let drift = ((mass - n) / n).abs();
        if drift > 1e-6 {
            Err(cosmic_scheduler::ScheduleError::StateInvalid(
                format!("Mass drift {drift:.2e} exceeds tolerance"),
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn test_q32_lbm_cosmic_scheduler_50_cycles() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = Q32LbmSystem::new(8);

    let initial_mass = system.domain.total_mass();

    // Run 50 two-phase cycles via the cosmic scheduler
    scheduler
        .execute_cycles(&mut system, 50)
        .expect("50 cycles should succeed");

    let final_mass = system.domain.total_mass();
    let drift = ((final_mass - initial_mass) / initial_mass).abs();

    println!("Q32.32 LBM via cosmic_scheduler:");
    println!("  Initial mass: {initial_mass:.10}");
    println!("  Final mass:   {final_mass:.10}");
    println!("  Drift:        {drift:.2e}");
    println!("  Collisions:   {}", system.collision_count);
    println!("  Streamings:   {}", system.streaming_count);

    assert_eq!(system.collision_count, 50, "should have 50 collisions");
    assert_eq!(system.streaming_count, 50, "should have 50 streamings");
    assert!(drift < 1e-6, "Q32.32 should have near-zero mass drift, got {drift}");
}

#[test]
fn test_q32_lbm_validate_state() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = Q32LbmSystem::new(4);

    // Run 10 cycles then validate
    scheduler.execute_cycles(&mut system, 10).unwrap();
    system.validate_state().expect("state should be valid after 10 cycles");
}
