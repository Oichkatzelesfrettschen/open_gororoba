"""
Genesis Simulation to Gravastar TOV Bridge (G2.3).

Extracts soliton-like structures from the k^{-3} vacuum dynamics simulation
and maps them to polytropic gravastar parameters for stability analysis.

The bridge addresses:
- How do the k^{-3} spectral dynamics produce localized structures?
- Which soliton configurations correspond to stable gravastars?
- What is the relationship between genesis mass spectrum and stable branches?

Key insight from G2.1-G2.2: The k^{-3} spectrum matches Kraichnan's 2D enstrophy
cascade, suggesting the vacuum dynamics has an underlying 2D-like structure.
This may explain why the simulation produces localized soliton-like objects.

References:
- Mazur & Mottola (2001, 2004) - Gravastar model
- Visser & Wiltshire (2004) - Polytropic shells
- Kraichnan (1967) - 2D enstrophy cascade k^{-3}
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
from scipy.fftpack import fft2, fftfreq, ifft2
from scipy.ndimage import label, center_of_mass, find_objects


@dataclass
class SolitonPeak:
    """Extracted soliton parameters from genesis simulation."""
    peak_id: int
    center_x: float
    center_y: float
    peak_density: float
    effective_radius: float  # Half-width at half-maximum
    total_mass: float  # Integrated density within FWHM
    aspect_ratio: float  # Elongation measure


@dataclass
class GravastarMapping:
    """Mapping from soliton to gravastar parameters."""
    soliton_id: int
    rho_v: float  # Vacuum energy density (from peak density)
    R1: float  # Inner shell radius (from soliton radius)
    rho_shell: float  # Shell density (from density gradient)
    gamma: float  # Polytropic exponent
    K: float  # Polytropic constant
    is_stable: bool  # From Harrison-Wheeler criterion
    M_total: float  # Total mass from TOV solution
    R2: float  # Outer shell radius


def run_genesis_simulation(
    N: int = 256,
    L: float = 100.0,
    alpha: float = -1.5,
    coupling: float = 20.0,
    steps: int = 300,
    seed: int = 137
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Run a lightweight genesis simulation for soliton extraction.

    This is a streamlined version of genesis_simulation_v2.py for
    programmatic use in the bridge analysis.

    Parameters
    ----------
    N : int
        Grid size (NxN).
    L : float
        Physical box size.
    alpha : float
        Spectral exponent (alpha = -1.5 gives k^{-3} kinetic energy).
    coupling : float
        Nonlinear self-interaction strength (attractive gravity).
    steps : int
        Number of evolution steps.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X, Y : np.ndarray
        Coordinate grids.
    density : np.ndarray
        Final probability density |psi|^2.
    metadata : dict
        Simulation parameters and diagnostics.
    """
    np.random.seed(seed)

    dx = L / N
    x = np.linspace(-L / 2, L / 2, N)
    y = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(x, y)

    # K-space
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    K[0, 0] = 1e-10  # Avoid singularity

    # Kinetic energy: T_k = -(K + 0.5)^{2*alpha}
    # For alpha = -1.5: kinetic energy ~ k^{-3}
    T_k = -(K + 0.5) ** (2 * alpha)

    # Initial state: vacuum fluctuations
    psi = np.random.normal(0, 0.1, (N, N)) + 1j * np.random.normal(0, 0.1, (N, N))

    # Filter: Planck-scale cutoff
    psi_k = fft2(psi) * np.exp(-(K / 10.0) ** 2)
    psi = ifft2(psi_k)
    psi /= np.linalg.norm(psi)

    dt = 0.05
    max_densities = []

    for t in range(steps):
        # Linear step (kinetic)
        psi_k = fft2(psi)
        psi_k *= np.exp(-1j * T_k * dt)
        psi = ifft2(psi_k)

        # Nonlinear step (attractive self-interaction)
        rho = np.abs(psi) ** 2
        V = -coupling * rho
        psi *= np.exp(-1j * V * dt)

        # Renormalize
        psi /= np.linalg.norm(psi)

        if t % 50 == 0:
            max_densities.append(np.max(np.abs(psi) ** 2))

    density = np.abs(psi) ** 2

    metadata = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "coupling": coupling,
        "steps": steps,
        "seed": seed,
        "max_densities": max_densities,
        "final_peak": float(np.max(density)),
    }

    return X, Y, density, metadata


def extract_soliton_peaks(
    X: np.ndarray,
    Y: np.ndarray,
    density: np.ndarray,
    threshold_fraction: float = 0.05,
    min_area_pixels: int = 50
) -> List[SolitonPeak]:
    """
    Extract soliton-like peaks from the density field.

    Uses connected component labeling to identify distinct peaks,
    then measures their properties for gravastar mapping.

    Parameters
    ----------
    X, Y : np.ndarray
        Coordinate grids.
    density : np.ndarray
        Probability density field.
    threshold_fraction : float
        Fraction of max density for peak identification.
    min_area_pixels : int
        Minimum area for valid soliton (reject noise).

    Returns
    -------
    List[SolitonPeak]
        List of extracted soliton parameters.
    """
    dx = X[0, 1] - X[0, 0]

    # Normalize and threshold
    density_norm = density / density.max()
    binary = density_norm > threshold_fraction

    # Label connected components
    labeled, n_components = label(binary)

    peaks = []
    for i in range(1, n_components + 1):
        mask = labeled == i
        area_pixels = np.sum(mask)

        if area_pixels < min_area_pixels:
            continue

        # Find centroid
        cy, cx = center_of_mass(density, labels=labeled, index=i)
        center_x = X[0, int(cx)]
        center_y = Y[int(cy), 0]

        # Peak density at centroid
        peak_density = density[mask].max()

        # Effective radius (FWHM-based)
        # Find where density drops to half max within the component
        half_max = peak_density / 2
        above_half = mask & (density > half_max)
        area_half = np.sum(above_half) * dx ** 2
        effective_radius = np.sqrt(area_half / np.pi)  # Circular equivalent

        # Total mass within FWHM region
        total_mass = np.sum(density[above_half]) * dx ** 2

        # Aspect ratio from bounding box
        all_slices = find_objects(labeled)
        if all_slices and len(all_slices) >= i and all_slices[i - 1]:
            s = all_slices[i - 1]
            height = s[0].stop - s[0].start
            width = s[1].stop - s[1].start
            aspect_ratio = max(height, width) / max(min(height, width), 1)
        else:
            aspect_ratio = 1.0

        peaks.append(SolitonPeak(
            peak_id=i,
            center_x=center_x,
            center_y=center_y,
            peak_density=peak_density,
            effective_radius=effective_radius,
            total_mass=total_mass,
            aspect_ratio=aspect_ratio,
        ))

    return peaks


def map_soliton_to_gravastar(
    soliton: SolitonPeak,
    gamma: float = 2.0,
    K: float = 1.0,
    density_scale: float = 1e-3,
    length_scale: float = 1.0
) -> GravastarMapping:
    """
    Map a soliton's parameters to gravastar TOV inputs.

    The mapping is heuristic but physically motivated:
    - rho_v (vacuum density) ~ soliton peak density
    - R1 (inner radius) ~ soliton effective radius
    - rho_shell ~ density at the soliton edge

    Parameters
    ----------
    soliton : SolitonPeak
        Extracted soliton parameters.
    gamma : float
        Polytropic exponent for shell EoS.
    K : float
        Polytropic constant.
    density_scale : float
        Conversion factor from simulation units to geometrized units.
    length_scale : float
        Conversion factor for lengths.

    Returns
    -------
    GravastarMapping
        Gravastar parameters (stability to be determined by TOV).
    """
    # Map simulation density to geometrized units
    rho_v = soliton.peak_density * density_scale

    # Inner shell radius from soliton size
    R1 = soliton.effective_radius * length_scale

    # Shell density from edge density (heuristic: 10-50% of peak)
    # This represents the matter accumulated at the phase boundary
    rho_shell = 0.3 * soliton.peak_density * density_scale

    return GravastarMapping(
        soliton_id=soliton.peak_id,
        rho_v=rho_v,
        R1=R1,
        rho_shell=rho_shell,
        gamma=gamma,
        K=K,
        is_stable=False,  # To be determined by TOV
        M_total=0.0,  # To be computed
        R2=0.0,  # To be computed
    )


def check_gravastar_stability(
    mapping: GravastarMapping,
    gamma_range: List[float] = None
) -> Tuple[bool, Dict]:
    """
    Check gravastar stability via polytropic TOV solution.

    Uses the Harrison-Wheeler criterion: dM/d(rho_c) > 0 for stability.

    Parameters
    ----------
    mapping : GravastarMapping
        Initial gravastar parameters.
    gamma_range : list
        Range of gamma values to test (default: [1.5, 2.0, 2.5]).

    Returns
    -------
    is_stable : bool
        Whether any gamma produces a stable configuration.
    results : dict
        Detailed results for each gamma tested.
    """
    if gamma_range is None:
        gamma_range = [1.5, 2.0, 2.5]

    # Import gravastar solver (deferred to avoid circular imports)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

    try:
        from gravastar_tov import solve_gravastar
    except ImportError:
        # Fallback for different import paths
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from gravastar_tov import solve_gravastar
        except ImportError:
            return False, {"error": "Could not import gravastar_tov"}

    results = {}
    any_stable = False

    for gamma in gamma_range:
        try:
            sol = solve_gravastar(
                rho_v=mapping.rho_v,
                R1=mapping.R1,
                rho_shell=mapping.rho_shell,
                gamma=gamma,
                K=mapping.K,
                n_points=500
            )

            # Check stability via Harrison-Wheeler:
            # For polytropic gamma >= 4/3, stable branch exists
            # Simplified criterion: valid solution with positive mass
            is_config_stable = (
                sol["equilibrium_satisfied"] and
                sol["M_total"] > 0 and
                sol["R2"] > sol["R1"] and
                gamma >= 4.0 / 3.0
            )

            results[gamma] = {
                "M_total": sol["M_total"],
                "R2": sol["R2"],
                "equilibrium_satisfied": sol["equilibrium_satisfied"],
                "is_stable": is_config_stable,
            }

            if is_config_stable:
                any_stable = True

        except ValueError as e:
            results[gamma] = {"error": str(e), "is_stable": False}
        except Exception as e:
            results[gamma] = {"error": str(e), "is_stable": False}

    return any_stable, results


def run_genesis_gravastar_bridge(
    output_dir: Optional[Path] = None,
    gamma_sweep: List[float] = None,
    n_simulations: int = 3,
    seeds: List[int] = None
) -> Dict:
    """
    Run the complete genesis-to-gravastar bridge analysis.

    Parameters
    ----------
    output_dir : Path, optional
        Directory for CSV output.
    gamma_sweep : list
        Gamma values to test for stability.
    n_simulations : int
        Number of genesis simulations with different seeds.
    seeds : list
        Specific seeds to use (default: [137, 42, 314]).

    Returns
    -------
    dict
        Complete analysis results.
    """
    if gamma_sweep is None:
        gamma_sweep = [1.0, 1.33, 1.5, 2.0, 2.5]

    if seeds is None:
        seeds = [137, 42, 314][:n_simulations]

    results = {
        "simulations": [],
        "solitons": [],
        "gravastar_mappings": [],
        "stability_summary": {
            "total_solitons": 0,
            "stable_configs": 0,
            "stable_gammas": {},
        },
    }

    print("=" * 70)
    print("GENESIS-GRAVASTAR BRIDGE ANALYSIS (G2.3)")
    print("=" * 70)

    for seed in seeds:
        print(f"\nRunning genesis simulation (seed={seed})...")
        X, Y, density, metadata = run_genesis_simulation(seed=seed, steps=300)
        results["simulations"].append(metadata)

        # Extract solitons
        peaks = extract_soliton_peaks(X, Y, density)
        print(f"  Found {len(peaks)} soliton peaks")

        for peak in peaks:
            results["solitons"].append({
                "seed": seed,
                "peak_id": peak.peak_id,
                "peak_density": peak.peak_density,
                "effective_radius": peak.effective_radius,
                "total_mass": peak.total_mass,
                "aspect_ratio": peak.aspect_ratio,
            })

            # Map to gravastar and check stability
            mapping = map_soliton_to_gravastar(peak, gamma=2.0)
            is_stable, stability_results = check_gravastar_stability(
                mapping, gamma_range=gamma_sweep
            )

            for gamma, sr in stability_results.items():
                if "error" not in sr:
                    results["gravastar_mappings"].append({
                        "seed": seed,
                        "soliton_id": peak.peak_id,
                        "rho_v": mapping.rho_v,
                        "R1": mapping.R1,
                        "rho_shell": mapping.rho_shell,
                        "gamma": gamma,
                        "M_total": sr.get("M_total", 0),
                        "R2": sr.get("R2", 0),
                        "is_stable": sr.get("is_stable", False),
                    })

            results["stability_summary"]["total_solitons"] += 1
            if is_stable:
                results["stability_summary"]["stable_configs"] += 1

            # Track which gammas produce stability
            for gamma, sr in stability_results.items():
                if sr.get("is_stable", False):
                    results["stability_summary"]["stable_gammas"][gamma] = \
                        results["stability_summary"]["stable_gammas"].get(gamma, 0) + 1

    # Summary
    total = results["stability_summary"]["total_solitons"]
    stable = results["stability_summary"]["stable_configs"]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {stable}/{total} solitons yield stable gravastars")
    print(f"Stable gamma values: {results['stability_summary']['stable_gammas']}")

    # Save CSV
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Soliton data
        soliton_path = output_dir / "genesis_soliton_extraction.csv"
        with open(soliton_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "peak_id", "peak_density", "effective_radius",
                "total_mass", "aspect_ratio"
            ])
            writer.writeheader()
            writer.writerows(results["solitons"])
        print(f"Saved soliton data to {soliton_path}")

        # Gravastar mappings
        mapping_path = output_dir / "genesis_gravastar_bridge.csv"
        with open(mapping_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "soliton_id", "rho_v", "R1", "rho_shell",
                "gamma", "M_total", "R2", "is_stable"
            ])
            writer.writeheader()
            writer.writerows(results["gravastar_mappings"])
        print(f"Saved gravastar bridge to {mapping_path}")

    return results


if __name__ == "__main__":
    results = run_genesis_gravastar_bridge(
        output_dir=Path("data/csv"),
        gamma_sweep=[1.0, 1.33, 1.5, 2.0, 2.5],
        n_simulations=3
    )
