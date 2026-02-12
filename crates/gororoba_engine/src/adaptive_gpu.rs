//! Adaptive backend chooser.

/// Compute backend for a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

/// Select backend from problem size and environment capability.
pub fn choose_backend(problem_size: usize, gpu_available: bool) -> ComputeBackend {
    if gpu_available && problem_size >= 10_000 {
        ComputeBackend::Gpu
    } else {
        ComputeBackend::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_backend_threshold() {
        assert_eq!(choose_backend(500, true), ComputeBackend::Cpu);
        assert_eq!(choose_backend(20_000, true), ComputeBackend::Gpu);
        assert_eq!(choose_backend(20_000, false), ComputeBackend::Cpu);
    }
}
