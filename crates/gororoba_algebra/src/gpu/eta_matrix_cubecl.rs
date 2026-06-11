//! cubecl-wgpu launcher for Cayley-Dickson eta matrix computation.
//!
//! The eta matrix is a binary `(dim / 2) * (dim / 2)` table. The cubecl
//! path writes one u32 flag per cell and narrows the readback to u8 on the
//! host after checking that every shader value is binary.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

#[cube(launch_unchecked)]
pub fn eta_matrix_kernel(eta_out: &mut Array<u32>, #[comptime] dim: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= eta_out.len() {
        terminate!();
    }

    let dim_half = (dim / 2u32) as usize;
    let i = (idx / dim_half) as u32;
    let j = (idx % dim_half) as u32;
    let psi_ij = psi(dim, i, j + dim / 2u32);
    let psi_ji = psi(dim, j, i + dim / 2u32);
    eta_out[idx] = psi_ij ^ psi_ji;
}

#[cube]
fn psi(dim: u32, i: u32, j: u32) -> u32 {
    cd_basis_mul_is_negative(dim, i, j)
}

#[cube]
fn cd_basis_mul_is_negative(dim: u32, p_input: u32, q_input: u32) -> u32 {
    let mut is_negative = 0u32;
    let mut p = p_input;
    let mut q = q_input;
    let mut half = dim / 2u32;

    while half > 0u32 {
        let p_hi = p >= half;
        let q_hi = q >= half;
        let mut next_half = half >> 1u32;

        if !p_hi && q_hi {
            let qh = q - half;
            q = p;
            p = qh;
        } else if p_hi && !q_hi {
            p -= half;
            if q != 0u32 {
                is_negative = 1u32 - is_negative;
            }
        } else if p_hi && q_hi {
            let qh = q - half;
            let ph = p - half;
            if qh == 0u32 {
                is_negative = 1u32 - is_negative;
                next_half = 0u32;
            } else {
                p = qh;
                q = ph;
            }
        }

        half = next_half;
    }

    is_negative
}

pub struct EtaMatrixCubeclKernel;

impl EtaMatrixCubeclKernel {
    pub fn is_available() -> bool {
        eta_matrix_cubecl_available()
    }

    pub fn output_len(dim: usize) -> Result<usize, String> {
        validate_dim(dim)?;
        let dim_half = dim / 2;
        dim_half
            .checked_mul(dim_half)
            .ok_or_else(|| format!("eta matrix dimension {dim} overflows usize"))
    }

    pub fn compute(dim: usize) -> Result<Vec<u8>, String> {
        let output_len = Self::output_len(dim)?;
        let dim_u32 =
            u32::try_from(dim).map_err(|_| format!("eta matrix dimension {dim} exceeds u32"))?;

        if !Self::is_available() {
            return Err("eta matrix cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let output_handle = client.empty(output_len * std::mem::size_of::<u32>());
        let output_handle_for_readback = output_handle.clone();
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(output_len.div_ceil(256) as u32);

        // SAFETY: launch_unchecked requires the host to prove buffer sizes.
        // `output_handle` has exactly output_len u32 slots and the kernel
        // terminates every thread with idx >= output_len.
        unsafe {
            eta_matrix_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(output_handle, output_len),
                dim_u32,
            );
        }

        let bytes = client.read_one_unchecked(output_handle_for_readback);
        decode_binary_u32_output(&bytes, output_len)
    }
}

pub fn eta_matrix_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

fn validate_dim(dim: usize) -> Result<(), String> {
    if dim < 2 {
        return Err(format!("eta matrix dimension must be >= 2, got {dim}"));
    }
    if !dim.is_power_of_two() {
        return Err(format!(
            "eta matrix dimension must be a power of two, got {dim}"
        ));
    }
    Ok(())
}

fn decode_binary_u32_output(bytes: &[u8], output_len: usize) -> Result<Vec<u8>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("eta matrix output length {output_len} overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "eta matrix cubecl readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    let mut eta = Vec::with_capacity(output_len);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        match value {
            0 => eta.push(0),
            1 => eta.push(1),
            other => {
                return Err(format!(
                    "eta matrix cubecl shader wrote non-binary value {other} at index {index}"
                ));
            }
        }
    }
    Ok(eta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_cubecl_available_does_not_panic() {
        let _ = EtaMatrixCubeclKernel::is_available();
    }

    #[test]
    fn eta_cubecl_output_len_matches_shape() {
        assert_eq!(EtaMatrixCubeclKernel::output_len(8).unwrap(), 16);
        assert_eq!(EtaMatrixCubeclKernel::output_len(64).unwrap(), 1024);
    }

    #[test]
    fn eta_cubecl_rejects_invalid_dimensions() {
        assert!(EtaMatrixCubeclKernel::output_len(0).is_err());
        assert!(EtaMatrixCubeclKernel::output_len(12).is_err());
    }

    #[test]
    fn eta_cubecl_decodes_binary_output() {
        let bytes = [0u32, 1, 1, 0]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_binary_u32_output(&bytes, 4).unwrap(),
            vec![0, 1, 1, 0]
        );
    }

    #[test]
    fn eta_cubecl_rejects_non_binary_output() {
        let bytes = [2u32]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert!(decode_binary_u32_output(&bytes, 1).is_err());
    }
}
