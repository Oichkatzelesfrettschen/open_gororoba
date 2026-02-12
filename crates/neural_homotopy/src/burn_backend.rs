//! Lightweight backend selection helpers.
//!
//! The crate can train with Burn backends, but these helpers keep backend
//! selection deterministic and testable even without device probing.

/// Compile-time backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
}

/// Return backend selected by crate features.
pub fn selected_backend() -> BackendKind {
    if cfg!(feature = "cuda-backend") {
        BackendKind::Cuda
    } else if cfg!(feature = "wgpu-backend") {
        BackendKind::Wgpu
    } else {
        BackendKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection_is_defined() {
        let backend = selected_backend();
        match backend {
            BackendKind::Cpu | BackendKind::Wgpu | BackendKind::Cuda => {}
        }
    }
}
