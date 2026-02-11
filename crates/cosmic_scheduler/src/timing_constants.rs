/// Timing constants extracted from Intel 4004/4040 datasheet
/// and adapted for generic phase scheduling.
/// Source: /home/eirikr/Github/4-bit/mcs4-emu/crates/mcs4-core/src/timing.rs
///
/// Time in picoseconds (10^-12 seconds)
pub type Time = u64;

/// Propagation delay
pub type Delay = u64;

/// Time scale constants
pub const PICOSECOND: Time = 1;
pub const NANOSECOND: Time = 1_000;
pub const MICROSECOND: Time = 1_000_000;
pub const MILLISECOND: Time = 1_000_000_000;

/// Intel 4004 clock timing specifications
pub mod clock_spec {
    use super::*;

    /// Minimum clock period (1.35 us = 740 kHz)
    pub const TCY_MIN: Time = 1_350 * NANOSECOND;

    /// Maximum clock period (2.0 us = 500 kHz)
    pub const TCY_MAX: Time = 2_000 * NANOSECOND;

    /// Typical clock period for simulation
    pub const TCY_TYP: Time = 1_350 * NANOSECOND;

    /// Clock rise time (50 ns typical)
    pub const T0R: Time = 50 * NANOSECOND;

    /// Clock fall time (50 ns typical)
    pub const T0F: Time = 50 * NANOSECOND;

    /// Clock pulse width (380-480 ns)
    pub const T0PW_MIN: Time = 380 * NANOSECOND;
    pub const T0PW_MAX: Time = 480 * NANOSECOND;

    /// Delay phi1 to phi2 (400-550 ns)
    pub const T0D1_MIN: Time = 400 * NANOSECOND;
    pub const T0D1_MAX: Time = 550 * NANOSECOND;

    /// Delay phi2 to phi1 (150 ns min)
    pub const T0D2_MIN: Time = 150 * NANOSECOND;
}

/// Gate delay model parameters
pub mod gate_delay {
    use super::*;

    /// Base delay for 2-input NAND gate
    pub const NAND2_BASE: Delay = 5 * NANOSECOND;

    /// Base delay for inverter
    pub const INV_BASE: Delay = 3 * NANOSECOND;

    /// Additional delay per fanout (capacitive loading)
    pub const FANOUT_FACTOR: Delay = 500 * PICOSECOND;

    /// Calculate total propagation delay including fanout
    pub fn with_fanout(base: Delay, fanout: usize) -> Delay {
        base + (fanout as Delay * FANOUT_FACTOR)
    }
}

/// Format time to human-readable string
pub fn format_time(t: Time) -> String {
    if t >= MILLISECOND {
        format!("{:.3} ms", t as f64 / MILLISECOND as f64)
    } else if t >= MICROSECOND {
        format!("{:.3} us", t as f64 / MICROSECOND as f64)
    } else if t >= NANOSECOND {
        format!("{:.3} ns", t as f64 / NANOSECOND as f64)
    } else {
        format!("{t} ps")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_constants() {
        assert_eq!(1000 * PICOSECOND, NANOSECOND);
        assert_eq!(1000 * NANOSECOND, MICROSECOND);
        assert_eq!(1000 * MICROSECOND, MILLISECOND);
    }

    #[test]
    fn test_clock_period_range() {
        assert_eq!(clock_spec::TCY_MIN, 1_350_000);
        assert_eq!(clock_spec::TCY_MAX, 2_000_000);
    }

    #[test]
    fn test_fanout_delay() {
        let base = gate_delay::NAND2_BASE;
        assert_eq!(gate_delay::with_fanout(base, 0), base);
        assert_eq!(gate_delay::with_fanout(base, 1), base + 500);
        assert_eq!(gate_delay::with_fanout(base, 10), base + 5000);
    }

    #[test]
    fn test_format_time() {
        assert_eq!(format_time(500), "500 ps");
        assert_eq!(format_time(5000), "5.000 ns");
        assert_eq!(format_time(5_000_000), "5.000 us");
        assert_eq!(format_time(5_000_000_000), "5.000 ms");
    }
}
