"""
Fractal cosmology and spectral dimension analysis.

Implements Calcagni's multi-fractional spacetime model and compares various
spectral exponent predictions against the k^{-3} ansatz.

Key results:
- Calcagni 2010: Spectral dimension d_S flows from 2 (UV) to 4 (IR)
- Kraichnan 1967: 2D enstrophy cascade gives E(k) ~ k^{-3}
- Kolmogorov 1941: 3D energy cascade gives E(k) ~ k^{-5/3}
- Parisi-Sourlas 1979: Dimensional reduction D -> D-2 in random fields

References:
- Calcagni, PRL 104 (2010) 251301 [arXiv:0912.3142]
- Calcagni, JHEP 2012:65 [arXiv:1107.5041]
- Kraichnan, Phys. Fluids 10 (1967) 1417
- Kolmogorov, Dokl. Akad. Nauk SSSR 30 (1941) 299
- Parisi & Sourlas, PRL 43 (1979) 744
- Ambjorn et al., PRL 95 (2005) 171301 [CDT]
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv


def calcagni_spectral_dimension(k: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Calcagni's running spectral dimension d_S(k).

    In multi-fractional spacetime, the spectral dimension interpolates:
    - d_S -> 2 as k -> infinity (UV, Planck scale)
    - d_S -> 4 as k -> 0 (IR, macroscopic scale)

    The interpolation follows:
        d_S(k) = 4 - 2 / (1 + (k/k_*)^{-alpha})

    where k_* is the transition scale (set to 1 in natural units).

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array.
    alpha : float
        Interpolation exponent (default 0.5 from Calcagni).

    Returns
    -------
    np.ndarray
        Spectral dimension at each k.
    """
    k_star = 1.0  # Transition scale in natural units
    ratio = (k / k_star) ** (-alpha)
    d_S = 4.0 - 2.0 / (1.0 + ratio)
    return d_S


def calcagni_spectral_density(k: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Spectral density P(k) from Calcagni's running dimension.

    In d_S dimensions, the spectral density scales as:
        P(k) ~ k^{d_S(k) - 1}

    This gives a k-dependent power law.

    Returns
    -------
    np.ndarray
        Spectral density at each k (normalized to P(1) = 1).
    """
    d_S = calcagni_spectral_dimension(k, alpha)
    # P(k) ~ k^{d_S - 1}
    P = k ** (d_S - 1)
    # Normalize
    return P / P[np.argmin(np.abs(k - 1.0))]


def k_minus_3_spectrum(k: np.ndarray) -> np.ndarray:
    """
    The k^{-3} spectral ansatz from the project.

    This corresponds to a specific power law E(k) ~ k^{-3}.

    Returns
    -------
    np.ndarray
        Spectral density (normalized to P(1) = 1).
    """
    P = k ** (-3)
    return P / P[np.argmin(np.abs(k - 1.0))]


def kolmogorov_spectrum(k: np.ndarray) -> np.ndarray:
    """
    Kolmogorov 1941 energy cascade spectrum: E(k) ~ k^{-5/3}.

    This is the universal spectrum for 3D homogeneous isotropic turbulence
    in the inertial range.

    Returns
    -------
    np.ndarray
        Spectral density (normalized).
    """
    P = k ** (-5.0 / 3.0)
    return P / P[np.argmin(np.abs(k - 1.0))]


def kraichnan_enstrophy_spectrum(k: np.ndarray) -> np.ndarray:
    """
    Kraichnan 1967 enstrophy cascade spectrum: E(k) ~ k^{-3}.

    In 2D turbulence, there are TWO cascade regimes:
    - Inverse energy cascade: E(k) ~ k^{-5/3} (large scales)
    - Forward enstrophy cascade: E(k) ~ k^{-3} (small scales)

    The k^{-3} spectrum arises from enstrophy (vorticity squared) conservation.

    This is the EXACT match for the project's k^{-3} ansatz!

    Returns
    -------
    np.ndarray
        Spectral density (normalized).
    """
    return k_minus_3_spectrum(k)  # Same as k^{-3}


def parisi_sourlas_effective_dimension(D: int) -> int:
    """
    Parisi-Sourlas dimensional reduction: D -> D - 2.

    In the presence of quenched random disorder, certain field theories
    exhibit supersymmetric structure that reduces the effective dimension.

    For D = 4, the effective dimension is D_eff = 2.

    Parameters
    ----------
    D : int
        Physical dimension.

    Returns
    -------
    int
        Effective dimension after Parisi-Sourlas reduction.
    """
    return max(D - 2, 0)


def parisi_sourlas_spectrum_exponent(D: int) -> float:
    """
    Spectral exponent from Parisi-Sourlas effective dimension.

    If D_eff = D - 2, and spectral density P(k) ~ k^{D_eff - 1}, then:
    - D = 4: D_eff = 2, P(k) ~ k^1 (NOT k^{-3})
    - D = 5: D_eff = 3, P(k) ~ k^2

    Parisi-Sourlas does NOT produce k^{-3}.

    Returns
    -------
    float
        Spectral exponent (d_eff - 1).
    """
    D_eff = parisi_sourlas_effective_dimension(D)
    return D_eff - 1


def cdt_spectral_dimension(k: np.ndarray, k_pl: float = 1.0) -> np.ndarray:
    """
    Spectral dimension from Causal Dynamical Triangulation (CDT).

    CDT simulations (Ambjorn et al. 2005) show:
    - d_S -> 2 as k -> infinity (UV, Planck scale)
    - d_S -> 4 as k -> 0 (IR)

    This matches Calcagni's fractal model qualitatively.

    The interpolation is:
        d_S(k) = 4 - 2 * exp(-k_pl / k)

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array.
    k_pl : float
        Planck wavenumber scale.

    Returns
    -------
    np.ndarray
        Spectral dimension.
    """
    d_S = 4.0 - 2.0 * np.exp(-k_pl / k)
    return d_S


def compare_spectra(
    k_min: float = 0.01,
    k_max: float = 100.0,
    n_points: int = 1000
) -> Dict[str, np.ndarray]:
    """
    Compare all spectral models.

    Returns
    -------
    dict
        Dictionary with k and all spectral densities.
    """
    k = np.geomspace(k_min, k_max, n_points)

    return {
        "k": k,
        "k_minus_3": k_minus_3_spectrum(k),
        "kolmogorov": kolmogorov_spectrum(k),
        "kraichnan_enstrophy": kraichnan_enstrophy_spectrum(k),
        "calcagni": calcagni_spectral_density(k),
        "calcagni_d_S": calcagni_spectral_dimension(k),
        "cdt_d_S": cdt_spectral_dimension(k),
    }


def compute_rms_deviation(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
    log_space: bool = True
) -> float:
    """
    Compute RMS deviation between two spectra.

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        Spectral densities to compare.
    log_space : bool
        If True, compare in log space (for power laws).

    Returns
    -------
    float
        RMS deviation.
    """
    if log_space:
        # Avoid log(0)
        s1 = np.log10(np.clip(spectrum1, 1e-30, None))
        s2 = np.log10(np.clip(spectrum2, 1e-30, None))
    else:
        s1, s2 = spectrum1, spectrum2

    return float(np.sqrt(np.mean((s1 - s2) ** 2)))


def analyze_k_minus_3_origin() -> Dict[str, any]:
    """
    Analyze the physical origin of the k^{-3} spectrum.

    Key question: Is k^{-3} consistent with any published framework?

    Results:
    1. Calcagni fractal cosmology: NO (running dimension gives varying exponent)
    2. Kolmogorov 3D turbulence: NO (gives k^{-5/3})
    3. Kraichnan 2D enstrophy cascade: YES (exact match!)
    4. Parisi-Sourlas: NO (gives positive exponents)
    5. CDT: NO (similar to Calcagni)

    Conclusion: k^{-3} is the 2D enstrophy cascade spectrum (Kraichnan 1967).

    Returns
    -------
    dict
        Analysis results.
    """
    spectra = compare_spectra()
    k = spectra["k"]

    results = {
        "framework_comparison": {},
        "rms_deviations": {},
        "conclusion": "",
    }

    # Compare k^{-3} against each framework
    k_minus_3 = spectra["k_minus_3"]

    # Kolmogorov
    kolm_dev = compute_rms_deviation(k_minus_3, spectra["kolmogorov"])
    results["rms_deviations"]["kolmogorov"] = kolm_dev
    results["framework_comparison"]["kolmogorov"] = {
        "exponent": -5/3,
        "matches_k_minus_3": False,
        "deviation": kolm_dev,
    }

    # Kraichnan enstrophy (should be exact match)
    kraich_dev = compute_rms_deviation(k_minus_3, spectra["kraichnan_enstrophy"])
    results["rms_deviations"]["kraichnan_enstrophy"] = kraich_dev
    results["framework_comparison"]["kraichnan_enstrophy"] = {
        "exponent": -3,
        "matches_k_minus_3": True,
        "deviation": kraich_dev,
    }

    # Calcagni
    calcagni_dev = compute_rms_deviation(k_minus_3, spectra["calcagni"])
    results["rms_deviations"]["calcagni"] = calcagni_dev
    results["framework_comparison"]["calcagni"] = {
        "exponent": "running (1 to 3)",
        "matches_k_minus_3": False,
        "deviation": calcagni_dev,
    }

    # Parisi-Sourlas
    ps_exp = parisi_sourlas_spectrum_exponent(4)
    results["framework_comparison"]["parisi_sourlas_D4"] = {
        "exponent": ps_exp,
        "matches_k_minus_3": False,
        "note": f"D=4 -> D_eff=2 gives k^{ps_exp}, NOT k^{-3}",
    }

    # Final conclusion
    results["conclusion"] = (
        "The k^{-3} spectral exponent matches EXACTLY the Kraichnan (1967) "
        "enstrophy cascade spectrum in 2D turbulence. This suggests the "
        "project's vacuum dynamics model may have an underlying 2D structure "
        "or be related to enstrophy conservation. Neither Calcagni fractal "
        "cosmology nor Parisi-Sourlas dimensional reduction produces k^{-3}."
    )

    return results


def run_fractal_cosmology_analysis(
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Run complete fractal cosmology analysis.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save CSV artifacts.

    Returns
    -------
    dict
        Full analysis results.
    """
    results = {}

    # Step 1: Compare spectra
    print("Comparing spectral models...")
    spectra = compare_spectra()
    results["spectra"] = spectra

    # Step 2: Analyze k^{-3} origin
    print("Analyzing k^{-3} origin...")
    origin_analysis = analyze_k_minus_3_origin()
    results["origin_analysis"] = origin_analysis

    # Step 3: UV/IR behavior
    print("Analyzing UV/IR limits...")
    k_uv = 100.0
    k_ir = 0.01

    results["uv_ir_analysis"] = {
        "calcagni_d_S_UV": float(calcagni_spectral_dimension(np.array([k_uv]))[0]),
        "calcagni_d_S_IR": float(calcagni_spectral_dimension(np.array([k_ir]))[0]),
        "cdt_d_S_UV": float(cdt_spectral_dimension(np.array([k_uv]))[0]),
        "cdt_d_S_IR": float(cdt_spectral_dimension(np.array([k_ir]))[0]),
        "expected_UV": 2.0,
        "expected_IR": 4.0,
    }

    # Save CSV if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save spectral comparison
        csv_path = output_dir / "fractal_cosmology_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "k_minus_3", "kolmogorov", "kraichnan", "calcagni", "calcagni_d_S", "cdt_d_S"])
            for i in range(len(spectra["k"])):
                writer.writerow([
                    spectra["k"][i],
                    spectra["k_minus_3"][i],
                    spectra["kolmogorov"][i],
                    spectra["kraichnan_enstrophy"][i],
                    spectra["calcagni"][i],
                    spectra["calcagni_d_S"][i],
                    spectra["cdt_d_S"][i],
                ])
        print(f"Saved spectral comparison to {csv_path}")

        # Save RMS deviations
        dev_path = output_dir / "spectral_exponent_comparisons.csv"
        with open(dev_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["framework", "exponent", "rms_deviation_from_k_minus_3", "matches"])
            for fw, data in origin_analysis["framework_comparison"].items():
                exp = data.get("exponent", "N/A")
                dev = origin_analysis["rms_deviations"].get(fw, "N/A")
                matches = data.get("matches_k_minus_3", False)
                writer.writerow([fw, exp, dev, matches])
        print(f"Saved exponent comparisons to {dev_path}")

    return results


if __name__ == "__main__":
    results = run_fractal_cosmology_analysis(output_dir=Path("data/csv"))

    print("\n" + "=" * 70)
    print("FRACTAL COSMOLOGY ANALYSIS RESULTS")
    print("=" * 70)

    print("\nSpectral Dimension UV/IR Limits:")
    uv_ir = results["uv_ir_analysis"]
    print(f"  Calcagni d_S(UV): {uv_ir['calcagni_d_S_UV']:.3f} (expected: 2)")
    print(f"  Calcagni d_S(IR): {uv_ir['calcagni_d_S_IR']:.3f} (expected: 4)")
    print(f"  CDT d_S(UV): {uv_ir['cdt_d_S_UV']:.3f}")
    print(f"  CDT d_S(IR): {uv_ir['cdt_d_S_IR']:.3f}")

    print("\nRMS Deviations from k^{-3}:")
    for fw, dev in results["origin_analysis"]["rms_deviations"].items():
        print(f"  {fw}: {dev:.4f}")

    print("\nConclusion:")
    print(f"  {results['origin_analysis']['conclusion']}")
