/// Unit tests for PhaseScheduler trait and TwoPhaseClockScheduler
use cosmic_scheduler::{
    Phase, ScheduleError, ScheduleResult, TwoPhaseClockScheduler, TwoPhaseSystem,
};

/// Simple mock system for testing
struct TestSystem {
    phase1_executions: usize,
    phase2_executions: usize,
}

impl TestSystem {
    fn new() -> Self {
        Self {
            phase1_executions: 0,
            phase2_executions: 0,
        }
    }
}

impl TwoPhaseSystem for TestSystem {
    fn execute_phase1(&mut self) -> ScheduleResult<()> {
        self.phase1_executions += 1;
        Ok(())
    }

    fn execute_phase2(&mut self) -> ScheduleResult<()> {
        self.phase2_executions += 1;
        Ok(())
    }
}

#[test]
fn test_scheduler_initialization() {
    let scheduler = TwoPhaseClockScheduler::new();
    assert_eq!(scheduler.current_phase(), Phase::Phi1);
    assert_eq!(scheduler.cycles_elapsed(), 0);
    assert_eq!(scheduler.total_time(), 0);
}

#[test]
fn test_phase_transition_sequence() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = TestSystem::new();

    // Start in Phi1
    assert_eq!(scheduler.current_phase(), Phase::Phi1);

    // Execute Phi1
    assert!(scheduler.execute_phase1(&mut system).is_ok());
    assert_eq!(scheduler.current_phase(), Phase::Phi2);
    assert_eq!(system.phase1_executions, 1);

    // Execute Phi2
    assert!(scheduler.execute_phase2(&mut system).is_ok());
    assert_eq!(scheduler.current_phase(), Phase::Phi1);
    assert_eq!(system.phase2_executions, 1);
}

#[test]
fn test_enforce_phase_ordering() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = TestSystem::new();

    // Try to execute Phi2 when expecting Phi1
    let result = scheduler.execute_phase2(&mut system);
    assert!(matches!(result, Err(ScheduleError::PhaseOutOfOrder(_))));
}

#[test]
fn test_cycle_count_increments() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = TestSystem::new();

    assert_eq!(scheduler.cycles_elapsed(), 0);

    scheduler.execute_cycle(&mut system).unwrap();
    assert_eq!(scheduler.cycles_elapsed(), 1);

    scheduler.execute_cycle(&mut system).unwrap();
    assert_eq!(scheduler.cycles_elapsed(), 2);
}

#[test]
fn test_multiple_cycles_execution() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = TestSystem::new();

    let cycle_count = 10;
    scheduler.execute_cycles(&mut system, cycle_count).unwrap();

    assert_eq!(scheduler.cycles_elapsed(), cycle_count);
    assert_eq!(system.phase1_executions, cycle_count as usize);
    assert_eq!(system.phase2_executions, cycle_count as usize);
}

#[test]
fn test_scheduler_reset() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = TestSystem::new();

    // Execute some cycles
    scheduler.execute_cycles(&mut system, 5).unwrap();
    assert_eq!(scheduler.cycles_elapsed(), 5);

    // Reset
    scheduler.reset();
    assert_eq!(scheduler.cycles_elapsed(), 0);
    assert_eq!(scheduler.total_time(), 0);
    assert_eq!(scheduler.current_phase(), Phase::Phi1);
}

#[test]
fn test_phase_enum_next() {
    assert_eq!(Phase::Phi1.next(), Phase::Phi2);
    assert_eq!(Phase::Phi2.next(), Phase::Phi1);
}

#[test]
fn test_phase_names() {
    assert_eq!(Phase::Phi1.name(), "phi1");
    assert_eq!(Phase::Phi2.name(), "phi2");
}

#[test]
fn test_custom_cycle_period() {
    let custom_period = 2_000_000;
    let scheduler = TwoPhaseClockScheduler::with_period(custom_period);
    assert_eq!(scheduler.cycle_period(), custom_period);
}

#[test]
fn test_custom_phase_durations() {
    let phi1_dur = 800_000;
    let phi2_dur = 600_000;
    let scheduler = TwoPhaseClockScheduler::with_custom_phases(phi1_dur, phi2_dur);

    assert_eq!(scheduler.cycle_period(), phi1_dur + phi2_dur);
}
