//! NVML (NVIDIA Management Library) telemetry.
//!
//! Lifts `gororoba_cli_warp/src/warp_telemetry.rs:110-211` into a
//! reusable helper. The original telemetry has TWO modes: this NVML
//! mode (Rust-native via nvml-wrapper) and a `nvidia-smi` subprocess
//! fallback. This crate exposes only the NVML mode; callers needing the
//! subprocess fallback retain the warp_telemetry implementation.

use nvml_wrapper::{
    Nvml,
    enum_wrappers::device::{Clock, TemperatureSensor},
};

use crate::error::{CudaError, Result};

/// A single NVML telemetry sample.
#[derive(Clone, Debug, Default)]
pub struct TelemetrySample {
    pub temperature_c: Option<u32>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub power_w: Option<f64>,
    pub gpu_util_pct: Option<u32>,
    pub mem_util_pct: Option<u32>,
}

/// NVML telemetry handle.
pub struct Telemetry {
    nvml: Nvml,
    device_index: u32,
}

impl Telemetry {
    /// Initialise NVML and bind to device 0.
    pub fn with_default_device() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialise NVML and bind to a specific device ordinal.
    pub fn with_device(device_index: u32) -> Result<Self> {
        let nvml = Nvml::init().map_err(|e| CudaError::Nvml(format!("NVML init: {e}")))?;
        // Validate the device exists before returning.
        let _ = nvml
            .device_by_index(device_index)
            .map_err(|e| CudaError::Nvml(format!("device_by_index({device_index}): {e}")))?;
        Ok(Self { nvml, device_index })
    }

    /// Sample current telemetry for the bound device. Each field is
    /// `Option<T>` because NVML may return errors per-field (e.g.
    /// power_usage unsupported on consumer cards).
    pub fn sample(&self) -> Result<TelemetrySample> {
        let device = self
            .nvml
            .device_by_index(self.device_index)
            .map_err(|e| CudaError::Nvml(format!("device_by_index sample: {e}")))?;
        let util = device.utilization_rates().ok();
        Ok(TelemetrySample {
            temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
            sm_clock_mhz: device.clock_info(Clock::SM).ok(),
            mem_clock_mhz: device.clock_info(Clock::Memory).ok(),
            power_w: device.power_usage().ok().map(|mw| mw as f64 / 1000.0),
            gpu_util_pct: util.as_ref().map(|u| u.gpu),
            mem_util_pct: util.as_ref().map(|u| u.memory),
        })
    }

    /// Bound device ordinal.
    pub fn device_index(&self) -> u32 {
        self.device_index
    }
}
