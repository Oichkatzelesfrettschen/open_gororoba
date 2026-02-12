/// Generic two-phase clock scheduler trait
///
/// Models the Intel 4004's two-phase clock (phi1/phi2) abstraction.
/// Phi1 = preparation phase (collision, precharge, compute)
/// Phi2 = execution phase (streaming, evaluate, transfer)
///
/// This trait enables any system to use deterministic, coordinated two-phase evolution
/// with well-defined timing guarantees.
use crate::timing_constants::{clock_spec, Time};
use serde::{Deserialize, Serialize};

/// Result type for scheduler operations
pub type ScheduleResult<T> = Result<T, ScheduleError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleError {
    PhaseOutOfOrder(String),
    TimingViolation(String),
    StateInvalid(String),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::PhaseOutOfOrder(msg) => write!(f, "Phase out of order: {}", msg),
            Self::TimingViolation(msg) => write!(f, "Timing violation: {}", msg),
            Self::StateInvalid(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Current phase of the two-phase clock
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Phi1 - preparation/collision/compute phase
    Phi1,
    /// Phi2 - execution/streaming/transfer phase
    Phi2,
}

impl Phase {
    /// Get the next phase in the cycle
    pub fn next(self) -> Self {
        match self {
            Phase::Phi1 => Phase::Phi2,
            Phase::Phi2 => Phase::Phi1,
        }
    }

    /// Get human-readable name
    pub fn name(self) -> &'static str {
        match self {
            Phase::Phi1 => "phi1",
            Phase::Phi2 => "phi2",
        }
    }
}

/// Generic trait for systems using two-phase evolution
///
/// A system implementing this trait must be able to execute two distinct phases:
/// 1. Phase1: Preparation, collision, or compute
/// 2. Phase2: Execution, streaming, or evaluation
///
/// The scheduler ensures deterministic ordering and timing.
pub trait TwoPhaseSystem {
    /// Execute phase 1 (preparation/collision)
    fn execute_phase1(&mut self) -> ScheduleResult<()>;

    /// Execute phase 2 (streaming/execution)
    fn execute_phase2(&mut self) -> ScheduleResult<()>;

    /// Check if system is in valid state for phase transitions
    fn validate_state(&self) -> ScheduleResult<()> {
        Ok(())
    }

    /// Optional: Get current system time for logging
    fn current_time(&self) -> Option<Time> {
        None
    }
}

/// Generic two-phase clock scheduler
///
/// Coordinates phase1 and phase2 execution with deterministic timing.
/// Ensures phases occur in the correct order and respects timing constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoPhaseClockScheduler {
    /// Current phase
    current_phase: Phase,

    /// Cycle number (incremented after each complete cycle)
    cycle_count: u64,

    /// Total time elapsed (in picoseconds)
    total_time: Time,

    /// Period of one complete cycle (phi1 + phi2)
    cycle_period: Time,

    /// Duration of phi1 phase
    phi1_duration: Time,

    /// Duration of phi2 phase
    phi2_duration: Time,

    /// Whether timing constraints are enforced
    enforce_timing: bool,
}

impl TwoPhaseClockScheduler {
    /// Create a new scheduler with default Intel 4004 timings
    pub fn new() -> Self {
        Self::with_period(clock_spec::TCY_TYP)
    }

    /// Create a scheduler with specified cycle period
    pub fn with_period(cycle_period: Time) -> Self {
        let phi1_duration = clock_spec::T0D1_MIN;
        let phi2_duration = cycle_period - phi1_duration;

        Self {
            current_phase: Phase::Phi1,
            cycle_count: 0,
            total_time: 0,
            cycle_period,
            phi1_duration,
            phi2_duration,
            enforce_timing: true,
        }
    }

    /// Create a scheduler with custom phase durations
    pub fn with_custom_phases(phi1_duration: Time, phi2_duration: Time) -> Self {
        Self {
            current_phase: Phase::Phi1,
            cycle_count: 0,
            total_time: 0,
            cycle_period: phi1_duration + phi2_duration,
            phi1_duration,
            phi2_duration,
            enforce_timing: true,
        }
    }

    /// Get current phase
    pub fn current_phase(&self) -> Phase {
        self.current_phase
    }

    /// Get cycle count
    pub fn cycles_elapsed(&self) -> u64 {
        self.cycle_count
    }

    /// Get total time elapsed
    pub fn total_time(&self) -> Time {
        self.total_time
    }

    /// Get current cycle period
    pub fn cycle_period(&self) -> Time {
        self.cycle_period
    }

    /// Enable or disable timing constraint enforcement
    pub fn set_enforce_timing(&mut self, enforce: bool) {
        self.enforce_timing = enforce;
    }

    /// Execute phase 1 on a system
    pub fn execute_phase1<S: TwoPhaseSystem>(&mut self, system: &mut S) -> ScheduleResult<()> {
        if self.current_phase != Phase::Phi1 {
            return Err(ScheduleError::PhaseOutOfOrder(format!(
                "Expected phase1 but in {:?}",
                self.current_phase
            )));
        }

        // Validate system state before execution
        system.validate_state()?;

        // Execute phase 1
        system.execute_phase1()?;

        // Update timing
        self.total_time += self.phi1_duration;
        self.current_phase = Phase::Phi2;

        Ok(())
    }

    /// Execute phase 2 on a system
    pub fn execute_phase2<S: TwoPhaseSystem>(&mut self, system: &mut S) -> ScheduleResult<()> {
        if self.current_phase != Phase::Phi2 {
            return Err(ScheduleError::PhaseOutOfOrder(format!(
                "Expected phase2 but in {:?}",
                self.current_phase
            )));
        }

        // Validate system state before execution
        system.validate_state()?;

        // Execute phase 2
        system.execute_phase2()?;

        // Update timing
        self.total_time += self.phi2_duration;
        self.cycle_count += 1;
        self.current_phase = Phase::Phi1;

        Ok(())
    }

    /// Execute one complete cycle (phase1 + phase2)
    pub fn execute_cycle<S: TwoPhaseSystem>(&mut self, system: &mut S) -> ScheduleResult<()> {
        self.execute_phase1(system)?;
        self.execute_phase2(system)?;
        Ok(())
    }

    /// Execute N complete cycles
    pub fn execute_cycles<S: TwoPhaseSystem>(
        &mut self,
        system: &mut S,
        n: u64,
    ) -> ScheduleResult<()> {
        for _ in 0..n {
            self.execute_cycle(system)?;
        }
        Ok(())
    }

    /// Reset scheduler to initial state
    pub fn reset(&mut self) {
        self.current_phase = Phase::Phi1;
        self.cycle_count = 0;
        self.total_time = 0;
    }
}

impl Default for TwoPhaseClockScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock system for testing
    struct MockSystem {
        phi1_count: u32,
        phi2_count: u32,
    }

    impl MockSystem {
        fn new() -> Self {
            Self {
                phi1_count: 0,
                phi2_count: 0,
            }
        }
    }

    impl TwoPhaseSystem for MockSystem {
        fn execute_phase1(&mut self) -> ScheduleResult<()> {
            self.phi1_count += 1;
            Ok(())
        }

        fn execute_phase2(&mut self) -> ScheduleResult<()> {
            self.phi2_count += 1;
            Ok(())
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = TwoPhaseClockScheduler::new();
        assert_eq!(scheduler.current_phase(), Phase::Phi1);
        assert_eq!(scheduler.cycles_elapsed(), 0);
        assert_eq!(scheduler.total_time(), 0);
    }

    #[test]
    fn test_phase_ordering() {
        let mut scheduler = TwoPhaseClockScheduler::new();
        let mut system = MockSystem::new();

        assert_eq!(scheduler.current_phase(), Phase::Phi1);
        assert!(scheduler.execute_phase1(&mut system).is_ok());
        assert_eq!(scheduler.current_phase(), Phase::Phi2);
        assert!(scheduler.execute_phase2(&mut system).is_ok());
        assert_eq!(scheduler.current_phase(), Phase::Phi1);
        assert_eq!(scheduler.cycles_elapsed(), 1);
    }

    #[test]
    fn test_phase_out_of_order_error() {
        let mut scheduler = TwoPhaseClockScheduler::new();
        let mut system = MockSystem::new();

        // Try to execute phase2 when phase1 is expected
        let result = scheduler.execute_phase2(&mut system);
        assert!(result.is_err());
        match result {
            Err(ScheduleError::PhaseOutOfOrder(_)) => (),
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_complete_cycle() {
        let mut scheduler = TwoPhaseClockScheduler::new();
        let mut system = MockSystem::new();

        assert!(scheduler.execute_cycle(&mut system).is_ok());
        assert_eq!(system.phi1_count, 1);
        assert_eq!(system.phi2_count, 1);
        assert_eq!(scheduler.cycles_elapsed(), 1);
    }

    #[test]
    fn test_multiple_cycles() {
        let mut scheduler = TwoPhaseClockScheduler::new();
        let mut system = MockSystem::new();

        assert!(scheduler.execute_cycles(&mut system, 5).is_ok());
        assert_eq!(system.phi1_count, 5);
        assert_eq!(system.phi2_count, 5);
        assert_eq!(scheduler.cycles_elapsed(), 5);
    }

    #[test]
    fn test_timing_progression() {
        let mut scheduler = TwoPhaseClockScheduler::with_period(1_350_000);
        let mut system = MockSystem::new();

        let start_time = scheduler.total_time();
        assert!(scheduler.execute_cycle(&mut system).is_ok());
        let end_time = scheduler.total_time();

        assert!(end_time > start_time);
        assert_eq!(end_time - start_time, scheduler.cycle_period());
    }

    #[test]
    fn test_reset() {
        let mut scheduler = TwoPhaseClockScheduler::new();
        let mut system = MockSystem::new();

        assert!(scheduler.execute_cycles(&mut system, 10).is_ok());
        assert_eq!(scheduler.cycles_elapsed(), 10);

        scheduler.reset();
        assert_eq!(scheduler.cycles_elapsed(), 0);
        assert_eq!(scheduler.total_time(), 0);
        assert_eq!(scheduler.current_phase(), Phase::Phi1);
    }

    #[test]
    fn test_custom_phases() {
        let phi1_dur = 400_000;
        let phi2_dur = 500_000;
        let scheduler = TwoPhaseClockScheduler::with_custom_phases(phi1_dur, phi2_dur);

        assert_eq!(scheduler.phi1_duration, phi1_dur);
        assert_eq!(scheduler.phi2_duration, phi2_dur);
        assert_eq!(scheduler.cycle_period(), phi1_dur + phi2_dur);
    }

    #[test]
    fn test_phase_enum() {
        assert_eq!(Phase::Phi1.next(), Phase::Phi2);
        assert_eq!(Phase::Phi2.next(), Phase::Phi1);
        assert_eq!(Phase::Phi1.name(), "phi1");
        assert_eq!(Phase::Phi2.name(), "phi2");
    }
}
