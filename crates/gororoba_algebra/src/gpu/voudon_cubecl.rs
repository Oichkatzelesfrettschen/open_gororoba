//! cubecl-wgpu launcher for 256D Voudon frustration-field generation.
//!
//! The cubecl kernel mirrors the Vulkan count path: each thread computes one
//! cell's 32 deterministic basis-pair samples, writes a u32 frustration count,
//! and the host converts counts to public `f32` field values.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use super::voudon::Cd256FrustrationKernel;

#[cube(launch_unchecked)]
pub fn voudon_frustration_kernel(
    frustration_counts: &mut Array<u32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= frustration_counts.len() {
        terminate!();
    }

    let x = idx as u32 % nx;
    let y = (idx as u32 / nx) % ny;
    let z = idx as u32 / (nx * ny);
    let mut local_frustration = 0u32;
    let mut sample = 0u32;

    while sample < 32u32 {
        let i = spatial_index_256(x, y, z, seed ^ sample);
        let j = spatial_index_256(x, y, z, seed ^ (sample + 100u32));

        let s1_negative = cd_basis_mul_is_negative_256(i, i);
        let ij_idx = i ^ j;
        let s2_negative = cd_basis_mul_is_negative_256(i, j);
        let i_ij_negative = cd_basis_mul_is_negative_256(i, ij_idx) ^ s2_negative;

        if s1_negative != i_ij_negative {
            local_frustration += 1u32;
        }

        sample += 1u32;
    }

    frustration_counts[idx] = local_frustration;
}

#[cube]
fn spatial_index_256(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    ((spatial_hash(x, y, z, seed) & 0xffffu32) * 255u32) / 65_535u32
}

#[cube]
fn spatial_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut hash = seed ^ (x * 73_856_093u32) ^ (y * 19_349_663u32);
    hash ^= z * 83_492_791u32;
    hash = (hash >> 13u32) ^ hash;
    hash *= 0x5bd1_e995u32;
    (hash >> 15u32) ^ hash
}

#[cube]
fn cd_basis_mul_is_negative_256(p_input: u32, q_input: u32) -> u32 {
    let mut is_negative = 0u32;
    let mut p = p_input;
    let mut q = q_input;
    let mut half = 128u32;

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

pub struct VoudonCubeclKernel;

impl VoudonCubeclKernel {
    pub fn is_available() -> bool {
        voudon_cubecl_available()
    }

    pub fn compute_field(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Vec<f32>, String> {
        let prepared = PreparedVoudonField::new(nx, ny, nz, seed)?;
        if prepared.n_cells == 0 {
            return Ok(Vec::new());
        }
        if !Self::is_available() {
            return Err("Voudon cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let output_handle = client.empty(prepared.n_cells * std::mem::size_of::<u32>());
        let output_readback = output_handle.clone();
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.n_cells.div_ceil(256) as u32);

        // SAFETY: launch_unchecked requires exact buffer-size proof. The
        // output buffer has n_cells u32 entries, and n_cells is exactly
        // nx * ny * nz after u32 range validation.
        unsafe {
            voudon_frustration_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(output_handle, prepared.n_cells),
                prepared.nx,
                prepared.ny,
                prepared.seed,
            );
        }

        let bytes = client.read_one_unchecked(output_readback);
        let counts = decode_u32_output(&bytes, prepared.n_cells, "frustration_counts")?;
        counts_to_field(&counts)
    }
}

pub fn voudon_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

struct PreparedVoudonField {
    nx: u32,
    ny: u32,
    seed: u32,
    n_cells: usize,
}

impl PreparedVoudonField {
    fn new(nx: usize, ny: usize, nz: usize, seed: u32) -> Result<Self, String> {
        if nx > u32::MAX as usize || ny > u32::MAX as usize || nz > u32::MAX as usize {
            return Err(format!("Voudon field shape {nx}x{ny}x{nz} exceeds u32"));
        }
        let n_cells = nx
            .checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| format!("Voudon field shape {nx}x{ny}x{nz} overflows usize"))?;
        if n_cells > u32::MAX as usize {
            return Err(format!(
                "Voudon field cell count {n_cells} exceeds u32 dispatch"
            ));
        }

        Ok(Self {
            nx: nx as u32,
            ny: ny as u32,
            seed,
            n_cells,
        })
    }
}

fn counts_to_field(counts: &[u32]) -> Result<Vec<f32>, String> {
    counts
        .iter()
        .map(|&count| {
            if count > Cd256FrustrationKernel::SAMPLES_PER_CELL as u32 {
                Err(format!(
                    "Voudon cubecl count {count} exceeds sample count {}",
                    Cd256FrustrationKernel::SAMPLES_PER_CELL
                ))
            } else {
                Ok(count as f32 / Cd256FrustrationKernel::SAMPLES_PER_CELL as f32)
            }
        })
        .collect()
}

fn decode_u32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<u32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("Voudon cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "Voudon cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voudon_cubecl_available_does_not_panic() {
        let _ = VoudonCubeclKernel::is_available();
    }

    #[test]
    fn voudon_cubecl_prepares_field_shape() {
        let prepared = PreparedVoudonField::new(3, 2, 2, 42).unwrap();
        assert_eq!(prepared.nx, 3);
        assert_eq!(prepared.ny, 2);
        assert_eq!(prepared.seed, 42);
        assert_eq!(prepared.n_cells, 12);
    }

    #[test]
    fn voudon_cubecl_rejects_oversized_field() {
        assert!(PreparedVoudonField::new(u32::MAX as usize + 1, 1, 1, 0).is_err());
    }

    #[test]
    fn voudon_cubecl_decodes_counts() {
        let bytes = [0u32, 16, 32]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_u32_output(&bytes, 3, "counts").unwrap(),
            vec![0, 16, 32]
        );
        assert_eq!(counts_to_field(&[0, 16, 32]).unwrap(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn voudon_cubecl_rejects_out_of_range_counts() {
        assert!(counts_to_field(&[Cd256FrustrationKernel::SAMPLES_PER_CELL as u32 + 1]).is_err());
    }
}
