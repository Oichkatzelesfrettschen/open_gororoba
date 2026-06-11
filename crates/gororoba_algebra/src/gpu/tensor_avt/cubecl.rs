#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use super::TensorAVT;

const WORKGROUP_SIZE: u32 = 256;

#[cube(launch_unchecked)]
pub fn tensor_avt_mul_kernel(
    left: &Array<f32>,
    right: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] dim: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.len() {
        terminate!();
    }

    let dim_usize = dim as usize;
    let row = (idx % dim_usize) as u32;
    let batch = idx / dim_usize;
    let mut acc = 0.0f32;
    let mut j = 0u32;
    while j < dim {
        let src = row ^ j;
        let sign = cd_basis_mul_sign(dim, src, j);
        acc += left[src as usize] * sign * right[batch * dim_usize + (j as usize)];
        j += 1u32;
    }
    output[idx] = acc;
}

#[cube(launch_unchecked)]
pub fn tensor_avt_norm_kernel(vectors: &Array<f32>, norms: &mut Array<f32>, #[comptime] dim: u32) {
    let idx = ABSOLUTE_POS;
    if idx >= norms.len() {
        terminate!();
    }

    let dim_usize = dim as usize;
    let base = idx * dim_usize;
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < dim_usize {
        let value = vectors[base + j];
        acc += value * value;
        j += 1usize;
    }
    norms[idx] = acc;
}

#[cube]
fn cd_basis_mul_sign(dim: u32, p_input: u32, q_input: u32) -> f32 {
    let mut sign = 1.0f32;
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
                sign = -sign;
            }
        } else if p_hi && q_hi {
            let qh = q - half;
            let ph = p - half;
            if qh == 0u32 {
                sign = -sign;
                next_half = 0u32;
            } else {
                p = qh;
                q = ph;
            }
        }

        half = next_half;
    }

    sign
}

pub struct TensorAvtCubeclKernel;

impl TensorAvtCubeclKernel {
    pub fn is_available() -> bool {
        tensor_avt_cubecl_available()
    }

    pub fn compute_cd_mul_batch(
        dim: usize,
        left: &[f32],
        right: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        validate_mul_inputs(dim, left, right, batch_size)?;
        if !Self::is_available() {
            return Err("TensorAVT cubecl adapter unavailable".to_string());
        }
        let dim_u32 =
            u32::try_from(dim).map_err(|_| format!("TensorAVT cubecl dim {dim} exceeds u32"))?;
        let output_len = dim * batch_size;
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let left_handle = client.create_from_slice(&encode_f32_slice(left)?);
        let right_handle = client.create_from_slice(&encode_f32_slice(right)?);
        let output_handle = client.empty(output_len * std::mem::size_of::<f32>());
        let output_readback = output_handle.clone();
        let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let cube_count = CubeCount::new_1d(output_len.div_ceil(WORKGROUP_SIZE as usize) as u32);

        // SAFETY: each ArrayArg length is expressed in f32 elements and
        // matches the host allocation. The kernel terminates idx >= output_len.
        unsafe {
            tensor_avt_mul_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(left_handle, left.len()),
                ArrayArg::from_raw_parts(right_handle, right.len()),
                ArrayArg::from_raw_parts(output_handle, output_len),
                dim_u32,
            );
        }

        decode_f32_output(
            &client.read_one_unchecked(output_readback),
            output_len,
            "output",
        )
    }

    pub fn compute_norm_sq_batch(
        dim: usize,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        validate_norm_inputs(dim, vectors, n_vectors)?;
        if !Self::is_available() {
            return Err("TensorAVT cubecl adapter unavailable".to_string());
        }
        let dim_u32 =
            u32::try_from(dim).map_err(|_| format!("TensorAVT cubecl dim {dim} exceeds u32"))?;
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let vectors_handle = client.create_from_slice(&encode_f32_slice(vectors)?);
        let norms_handle = client.empty(n_vectors * std::mem::size_of::<f32>());
        let norms_readback = norms_handle.clone();
        let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
        let cube_count = CubeCount::new_1d(n_vectors.div_ceil(WORKGROUP_SIZE as usize) as u32);

        // SAFETY: buffers have exactly vectors.len() and n_vectors f32 slots.
        // The kernel terminates idx >= n_vectors.
        unsafe {
            tensor_avt_norm_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(vectors_handle, vectors.len()),
                ArrayArg::from_raw_parts(norms_handle, n_vectors),
                dim_u32,
            );
        }

        decode_f32_output(
            &client.read_one_unchecked(norms_readback),
            n_vectors,
            "norms",
        )
    }
}

pub fn tensor_avt_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

impl TensorAVT {
    pub fn compute_cd_mul_cubecl(&self, a: &[f32], x: &[f32]) -> Result<Vec<f32>, String> {
        TensorAvtCubeclKernel::compute_cd_mul_batch(self.dim, a, x, 1)
    }

    pub fn compute_cd_mul_batch_cubecl(
        &self,
        a: &[f32],
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, String> {
        TensorAvtCubeclKernel::compute_cd_mul_batch(self.dim, a, x_batch, batch_size)
    }

    pub fn compute_norm_sq_batch_cubecl(
        &self,
        vectors: &[f32],
        n_vectors: usize,
    ) -> Result<Vec<f32>, String> {
        TensorAvtCubeclKernel::compute_norm_sq_batch(self.dim, vectors, n_vectors)
    }
}

fn validate_dim(dim: usize) -> Result<(), String> {
    if dim < 16 || !dim.is_power_of_two() {
        return Err(format!(
            "TensorAVT cubecl dim must be a power of two >= 16, got {dim}"
        ));
    }
    Ok(())
}

fn validate_mul_inputs(
    dim: usize,
    left: &[f32],
    right: &[f32],
    batch_size: usize,
) -> Result<(), String> {
    validate_dim(dim)?;
    if batch_size == 0 {
        return Err("TensorAVT cubecl batch_size must be > 0".to_string());
    }
    if left.len() != dim {
        return Err(format!(
            "left input length {} must equal dim {}",
            left.len(),
            dim
        ));
    }
    let expected = dim * batch_size;
    if right.len() != expected {
        return Err(format!(
            "right input length {} must equal batch_size * dim {}",
            right.len(),
            expected
        ));
    }
    Ok(())
}

fn validate_norm_inputs(dim: usize, vectors: &[f32], n_vectors: usize) -> Result<(), String> {
    validate_dim(dim)?;
    if n_vectors == 0 {
        return Err("TensorAVT cubecl n_vectors must be > 0".to_string());
    }
    let expected = dim * n_vectors;
    if vectors.len() != expected {
        return Err(format!(
            "vectors length {} must equal n_vectors * dim {}",
            vectors.len(),
            expected
        ));
    }
    Ok(())
}

fn encode_f32_slice(values: &[f32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "TensorAVT cubecl buffer length overflows bytes".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_f32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("TensorAVT cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "TensorAVT cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_avt_cubecl_available_does_not_panic() {
        let _ = TensorAvtCubeclKernel::is_available();
    }

    #[test]
    fn tensor_avt_cubecl_rejects_invalid_inputs() {
        let left = vec![1.0f32; 16];
        let right = vec![1.0f32; 16];
        assert!(validate_mul_inputs(12, &left, &right, 1).is_err());
        assert!(validate_mul_inputs(16, &left[..15], &right, 1).is_err());
        assert!(validate_mul_inputs(16, &left, &right[..15], 1).is_err());
        assert!(validate_norm_inputs(16, &right, 0).is_err());
    }

    #[test]
    fn tensor_avt_cubecl_decodes_f32_output() {
        let bytes = [1.25f32, -0.5f32]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_f32_output(&bytes, 2, "test").unwrap(),
            vec![1.25, -0.5]
        );
    }
}
