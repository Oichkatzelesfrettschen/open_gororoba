/// Tests for deterministic and reproducible evolution
/// This is critical for CE-002: two-phase clock coordination
use cosmic_scheduler::{ScheduleResult, TwoPhaseClockScheduler, TwoPhaseSystem};

/// Deterministic system with predictable behavior
struct DeterministicSystem {
    counter: u64,
    phase1_count: u64,
    phase2_count: u64,
}

impl DeterministicSystem {
    fn new() -> Self {
        Self {
            counter: 0,
            phase1_count: 0,
            phase2_count: 0,
        }
    }
}

impl TwoPhaseSystem for DeterministicSystem {
    fn execute_phase1(&mut self) -> ScheduleResult<()> {
        self.counter += 1;
        self.phase1_count += 1;
        Ok(())
    }

    fn execute_phase2(&mut self) -> ScheduleResult<()> {
        self.counter += 10;
        self.phase2_count += 1;
        Ok(())
    }
}

#[test]
fn test_deterministic_counter_sequence() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    // Execute known sequence and verify
    scheduler.execute_phase1(&mut system).unwrap();
    assert_eq!(system.counter, 1);

    scheduler.execute_phase2(&mut system).unwrap();
    assert_eq!(system.counter, 11);

    scheduler.execute_phase1(&mut system).unwrap();
    assert_eq!(system.counter, 12);

    scheduler.execute_phase2(&mut system).unwrap();
    assert_eq!(system.counter, 22);
}

#[test]
fn test_reproducible_state() {
    // Run 1
    let mut sched1 = TwoPhaseClockScheduler::new();
    let mut sys1 = DeterministicSystem::new();
    sched1.execute_cycles(&mut sys1, 100).unwrap();
    let final_counter1 = sys1.counter;

    // Run 2 - identical
    let mut sched2 = TwoPhaseClockScheduler::new();
    let mut sys2 = DeterministicSystem::new();
    sched2.execute_cycles(&mut sys2, 100).unwrap();
    let final_counter2 = sys2.counter;

    // Same initial conditions -> same final state
    assert_eq!(final_counter1, final_counter2);
    assert_eq!(sys1.phase1_count, sys2.phase1_count);
    assert_eq!(sys1.phase2_count, sys2.phase2_count);
}

#[test]
fn test_counter_increment_pattern() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    let mut expected = 0u64;

    // Each cycle: +1 (phase1) then +10 (phase2) = +11 total
    for cycle in 0..10 {
        scheduler.execute_cycle(&mut system).unwrap();
        expected += 11;
        assert_eq!(
            system.counter, expected,
            "Counter mismatch at cycle {}",
            cycle
        );
    }

    assert_eq!(system.counter, 110);
}

#[test]
fn test_cycle_count_matches_execution() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    let cycles = 50;
    scheduler.execute_cycles(&mut system, cycles).unwrap();

    assert_eq!(scheduler.cycles_elapsed(), cycles);
    assert_eq!(system.phase1_count, cycles);
    assert_eq!(system.phase2_count, cycles);
}

#[test]
fn test_time_accumulation() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    let start_time = scheduler.total_time();
    let cycle_period = scheduler.cycle_period();

    scheduler.execute_cycles(&mut system, 100).unwrap();

    let end_time = scheduler.total_time();
    let expected_time = start_time + (100 * cycle_period);

    assert_eq!(end_time, expected_time);
}

#[test]
fn test_reset_reproducibility() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    // Run 1: 50 cycles
    scheduler.execute_cycles(&mut system, 50).unwrap();
    let state_after_50 = system.counter;

    // Run more cycles
    scheduler.execute_cycles(&mut system, 50).unwrap();
    let state_after_100 = system.counter;

    // Reset and run fresh
    scheduler.reset();
    let mut system2 = DeterministicSystem::new();

    // Run 50 cycles on reset scheduler
    scheduler.execute_cycles(&mut system2, 50).unwrap();
    assert_eq!(system2.counter, state_after_50);

    // Continue to 100
    scheduler.execute_cycles(&mut system2, 50).unwrap();
    assert_eq!(system2.counter, state_after_100);
}

#[test]
fn test_phase_invariant() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut system = DeterministicSystem::new();

    // At start of each cycle, should be in Phi1
    for _ in 0..10 {
        assert_eq!(scheduler.current_phase(), cosmic_scheduler::Phase::Phi1);
        scheduler.execute_cycle(&mut system).unwrap();
    }

    // After N complete cycles, back to Phi1
    assert_eq!(scheduler.current_phase(), cosmic_scheduler::Phase::Phi1);
}

#[test]
fn test_manual_phase_execution_matches_cycle() {
    // Method 1: Execute cycle
    let mut sched1 = TwoPhaseClockScheduler::new();
    let mut sys1 = DeterministicSystem::new();
    sched1.execute_cycle(&mut sys1).unwrap();
    let counter1 = sys1.counter;

    // Method 2: Manual phase execution
    let mut sched2 = TwoPhaseClockScheduler::new();
    let mut sys2 = DeterministicSystem::new();
    sched2.execute_phase1(&mut sys2).unwrap();
    sched2.execute_phase2(&mut sys2).unwrap();
    let counter2 = sys2.counter;

    // Should be identical
    assert_eq!(counter1, counter2);
    assert_eq!(sched1.cycles_elapsed(), sched2.cycles_elapsed());
    assert_eq!(sched1.total_time(), sched2.total_time());
}

#[test]
fn test_custom_period_reproducibility() {
    let custom_period = 5_000_000u64;

    // Run 1
    let mut sched1 = TwoPhaseClockScheduler::with_period(custom_period);
    let mut sys1 = DeterministicSystem::new();
    sched1.execute_cycles(&mut sys1, 10).unwrap();
    let time1 = sched1.total_time();

    // Run 2
    let mut sched2 = TwoPhaseClockScheduler::with_period(custom_period);
    let mut sys2 = DeterministicSystem::new();
    sched2.execute_cycles(&mut sys2, 10).unwrap();
    let time2 = sched2.total_time();

    // Same cycle period -> same time advancement
    assert_eq!(time1, time2);
    assert_eq!(time1, custom_period * 10);
}
