//! cubecl-wgpu backend for the Besag-Clifford `transform_viscosity` kernel.
//!
//! Pointwise transform (no atomics, no RNG, embarrassingly parallel):
//!
//!   tau[i] = 3 * nu_base * exp(lambda * (phi[i] - phi_mean)) + 0.5
//!
//! This is the simplest cubecl parity port in the matrix and closes the
//! pointwise-transform cell of #136 Phase 2.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::transform_viscosity_cpu::ViscosityTransformInputs;

#[cube(launch_unchecked)]
pub fn transform_viscosity_kernel(
    phi: &Array<f32>,
    tau_out: &mut Array<f32>,
    #[comptime] n_cells: u32,
    #[comptime] nu_base_bits: u32,
    #[comptime] lambda_bits: u32,
    #[comptime] phi_mean_bits: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= n_cells as usize {
        terminate!();
    }
    let nu_base = f32::reinterpret(nu_base_bits);
    let lambda = f32::reinterpret(lambda_bits);
    let phi_mean = f32::reinterpret(phi_mean_bits);
    let phi_i = phi[i];
    let nu_i = nu_base * f32::exp(lambda * (phi_i - phi_mean));
    tau_out[i] = 3.0_f32 * nu_i + 0.5_f32;
}

pub fn is_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let _client = WgpuRuntime::client(&device);
    })
    .is_ok()
}

#[derive(Debug, thiserror::Error)]
pub enum CubeclViscosityError {
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

pub fn transform_viscosity_cubecl(
    inputs: &ViscosityTransformInputs,
) -> Result<Vec<f32>, CubeclViscosityError> {
    if !is_available() {
        return Err(CubeclViscosityError::AdapterUnavailable);
    }
    let n = inputs.phi.len();
    if n == 0 {
        return Ok(vec![]);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let phi_bytes: &[u8] = bytemuck::cast_slice(&inputs.phi);
    let phi_handle = client.create_from_slice(phi_bytes);
    let tau_handle = client.empty(n * std::mem::size_of::<f32>());
    let tau_handle_readback = tau_handle.clone();

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n.div_ceil(256) as u32);

    // SAFETY: launch_unchecked needs the host to vouch for buffer sizes.
    // phi.len() == n == tau_out length. f32 scalars piped through
    // comptime u32 via to_bits() (ScalarArg removed from cubecl 0.10 prelude).
    unsafe {
        transform_viscosity_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(phi_handle, n),
            ArrayArg::from_raw_parts(tau_handle, n),
            n as u32,
            inputs.nu_base.to_bits(),
            inputs.lambda.to_bits(),
            inputs.phi_mean.to_bits(),
        );
    }

    let tau_bytes = client.read_one_unchecked(tau_handle_readback);
    let tau: &[f32] = bytemuck::cast_slice(&tau_bytes);
    Ok(tau.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    #[test]
    fn empty_input_returns_empty_without_gpu() {
        let inputs = ViscosityTransformInputs {
            phi: vec![],
            nu_base: 0.01,
            lambda: 1.0,
            phi_mean: 0.0,
        };
        let r = transform_viscosity_cubecl(&inputs);
        assert!(r.is_ok());
        assert!(r.unwrap().is_empty());
    }
}
