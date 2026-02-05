"""
Rigorous Warp Ring Animation V9 (Sapphire Matrix Edition).

Replaces Ice VIII with Al2O3 (Sapphire) as the dielectric host.
Material constants sourced from MaterialDatabase (Sellmeier / Drude-Lorentz
models) at 1550nm canonical wavelength.

Implements:
1. RK4 Ray Bending in complex-n GRIN media (rk4_step_absorbing).
2. Beer-Lambert Gold Absorption from Babar & Weaver (2015) tabulated data.
3. Smooth Grid Anti-Aliasing.

Scene wavelength WAVELENGTH_SU maps physical absorption to scene geometry:
with WAVELENGTH_SU=5.0 and gold k~10.35, skin depth ~ 0.08 SU, so rays
inside the gold torus (cross-section radius 0.5 SU) are fully absorbed
within a few RK4 steps.
"""

import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

# Internal Imports
sys.path.append(os.path.abspath("src"))
from gemini_physics.materials.database import MaterialDatabase
from gemini_physics.optics.grin_solver import rk4_step_absorbing

# --- Material Constants from Database (1550nm) ---
_db = MaterialDatabase()
_WL = 1550.0  # nm, canonical wavelength

N_SI = _db.get("Silicon").refractive_index(_WL).real        # ~3.476
N_SAPPHIRE = _db.get("Sapphire").refractive_index(_WL).real  # ~1.746
_n_gold = _db.get("Gold").refractive_index(_WL)
N_GOLD_REAL = _n_gold.real                                    # ~0.19
N_GOLD_IMAG = _n_gold.imag                                    # ~10.35
N_VAC = 1.0

# --- Configuration ---
TOTAL_FRAMES = 1
MAX_STEPS = 400

# Scene wavelength (scene units): controls Beer-Lambert absorption rate.
# lambda_SU = 5.0 gives skin depth ~ lambda/(4*pi*k) ~ 0.04 SU in gold.
WAVELENGTH_SU = 5.0

# Ring geometry (scene units)
RING_MAJOR_R = 4.0  # torus major radius
RING_MINOR_R = 0.5  # torus cross-section radius
RING_BLEND = 0.1    # smooth boundary transition width


# --- Scene Mapping (Complex n) ---

@njit(fastmath=True)
def map_n_complex(p):
    """
    Complex refractive index field n(p) = n_r + i*k.

    Dielectric region: Gaussian GRIN lens (Sapphire -> Silicon), k=0.
    Gold ring: torus at r=RING_MAJOR_R, complex n from Babar & Weaver.
    Smooth boundary transition over RING_BLEND width avoids gradient
    discontinuities that would break the central-difference gradient.
    """
    q_xy = math.sqrt(p[0]**2 + p[1]**2) - RING_MAJOR_R
    dist_sq = q_xy**2 + p[2]**2
    dist = math.sqrt(dist_sq)

    # Dielectric GRIN: Gaussian lens around the ring
    sigma = 1.5
    n_dielectric = (N_SAPPHIRE
                    + (N_SI - N_SAPPHIRE) * math.exp(-dist_sq / (2.0 * sigma**2)))

    if dist < RING_MINOR_R:
        # Inside the gold ring torus: smooth blend at boundary.
        # edge_dist = how far inside the boundary (0 at surface, max at center).
        edge_dist = RING_MINOR_R - dist
        if edge_dist < RING_BLEND:
            # Linear ramp: 0 at boundary -> 1 at RING_BLEND depth
            blend = edge_dist / RING_BLEND
        else:
            blend = 1.0
        n_r = N_GOLD_REAL * blend + n_dielectric * (1.0 - blend)
        n_i = N_GOLD_IMAG * blend
        return complex(n_r, n_i)

    return complex(n_dielectric, 0.0)


@njit(fastmath=True)
def get_grad_n_complex(p):
    """Central-difference gradient of Re(n), plus full complex n(p)."""
    eps = 0.01

    p_px = p.copy(); p_px[0] += eps
    p_mx = p.copy(); p_mx[0] -= eps
    nx = (map_n_complex(p_px).real - map_n_complex(p_mx).real) / (2 * eps)

    p_py = p.copy(); p_py[1] += eps
    p_my = p.copy(); p_my[1] -= eps
    ny = (map_n_complex(p_py).real - map_n_complex(p_my).real) / (2 * eps)

    p_pz = p.copy(); p_pz[2] += eps
    p_mz = p.copy(); p_mz[2] -= eps
    nz = (map_n_complex(p_pz).real - map_n_complex(p_mz).real) / (2 * eps)

    n0 = map_n_complex(p)
    return np.array([nx, ny, nz]), n0


# --- Simulation ---

@njit(parallel=True, fastmath=True)
def simulate_rays_v9(ro, rd, n_rays):
    """
    Simulates rays through the complex-n GRIN field.

    Ray paths follow grad(Re(n)) via RK4; amplitude decays via
    Beer-Lambert attenuation from Im(n) at each step.

    Returns: (final_pos, final_dir, energy_deposition)
    """
    out_pos = np.zeros((n_rays, 3), dtype=np.float64)
    out_dir = np.zeros((n_rays, 3), dtype=np.float64)
    deposition = np.zeros((n_rays, 3), dtype=np.float64)  # [theta, phi, energy]

    for i in prange(n_rays):
        p = ro[i].copy()
        d = rd[i].copy()
        energy = 1.0

        for _ in range(MAX_STEPS):
            # Adaptive step: smaller near high curvature
            grad_n, n_c = get_grad_n_complex(p)
            n_real = n_c.real
            if n_real < 1e-6:
                n_real = 1e-6
            curv = np.linalg.norm(grad_n) / n_real
            dt_step = max(0.005, min(0.5, 0.1 / (curv + 1e-6)))

            # RK4 step with complex n: ray path + Beer-Lambert attenuation
            p, d, atten, phase = rk4_step_absorbing(
                p, d, dt_step, get_grad_n_complex, WAVELENGTH_SU)

            # Track energy absorption
            absorbed = energy * (1.0 - atten)
            energy *= atten

            # Record deposition if significant absorption occurred
            if absorbed > 1e-6:
                q_xy = math.sqrt(p[0]**2 + p[1]**2) - RING_MAJOR_R
                theta = math.atan2(p[1], p[0])
                phi = math.atan2(p[2], q_xy)
                deposition[i, 0] = theta
                deposition[i, 1] = phi
                deposition[i, 2] += absorbed

            if energy < 0.01:
                break

            # Exit check
            if np.linalg.norm(p) > 20.0:
                break

        out_pos[i] = p
        out_dir[i] = d

    return out_pos, out_dir, deposition


def main():
    print("--- RIGOROUS WARP RENDER V9 (SAPPHIRE + BEER-LAMBERT) ---")
    print(f"    Si  n = {N_SI:.3f}")
    print(f"    Al2O3 n = {N_SAPPHIRE:.3f}")
    print(f"    Au  n = {N_GOLD_REAL:.2f} + {N_GOLD_IMAG:.2f}j")
    print(f"    Scene wavelength = {WAVELENGTH_SU:.1f} SU")

    out_dir = "data/artifacts/frames_v9"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize Rays (Perspective Camera)
    n_sqrt = 512
    n_rays = n_sqrt * n_sqrt

    # Use float64 for RK4 stability
    cam_pos = np.array([0.0, -12.0, 4.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 0, 1], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    ro = np.zeros((n_rays, 3), dtype=np.float64)
    rd = np.zeros((n_rays, 3), dtype=np.float64)

    for y in range(n_sqrt):
        for x in range(n_sqrt):
            idx = y * n_sqrt + x
            uv_x = (x - n_sqrt / 2) / (n_sqrt / 2)
            uv_y = (y - n_sqrt / 2) / (n_sqrt / 2)

            ro[idx] = cam_pos
            direction = fwd * 1.5 + right * uv_x + up * uv_y
            rd[idx] = direction / np.linalg.norm(direction)

    # 2. Run Physics
    t0 = time.time()
    pos, dirs, deps = simulate_rays_v9(ro, rd, n_rays)
    print(f"Physics complete in {time.time() - t0:.2f}s")

    # 3. Process Deposition (Plasmon Surface Field)
    # Binning depositions into a 2D surface map
    surf_res = 128
    plasmon_map = np.zeros((surf_res, surf_res))
    for i in range(n_rays):
        if deps[i, 2] > 0:
            # map theta, phi [-pi, pi] to [0, res]
            tx = int((deps[i, 0] + np.pi) / (2 * np.pi) * (surf_res - 1))
            px = int((deps[i, 1] + np.pi) / (2 * np.pi) * (surf_res - 1))
            plasmon_map[px, tx] += deps[i, 2]

    # 4. Render Final Image
    # (Simplified: Show the background grid distortion)
    img = np.zeros((n_sqrt, n_sqrt, 3))

    for i in range(n_rays):
        y = i // n_sqrt
        x = i % n_sqrt

        # Grid texture on the "celestial sphere"
        d = dirs[i]
        phi = math.atan2(d[1], d[0])
        theta = math.acos(max(-1.0, min(1.0, d[2])))

        # Smooth Grid (Anti-aliased)
        grid_w = 0.05
        g1 = abs(math.sin(phi * 10))
        g2 = abs(math.sin(theta * 20))

        grid_val = (max(0.0, 1.0 - abs(g1 - 0.9) / grid_w)
                    + max(0.0, 1.0 - abs(g2 - 0.9) / grid_w))

        if grid_val > 0.1:
            img[y, x, 2] += min(1.0, grid_val) * 0.5  # Blue grid

        # Add deposition glow (Projected back to image space)
        if deps[i, 2] > 0:
            img[y, x, 0] += deps[i, 2] * 5.0  # Red/Gold glow
            img[y, x, 1] += deps[i, 2] * 3.0

    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(img, origin='lower')
    plt.axis('off')
    plt.title("Rigorous Warp Ring V9 (Sapphire + Beer-Lambert Au)", color='white')

    out_path = f"{out_dir}/frame_0000.png"
    plt.savefig(out_path, facecolor='black')
    print(f"Saved test frame to {out_path}")


if __name__ == "__main__":
    main()
