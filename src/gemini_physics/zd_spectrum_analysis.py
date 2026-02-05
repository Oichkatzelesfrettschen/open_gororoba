"""
Zero-divisor spectrum analysis for Cayley-Dickson algebras.

Implements G3.4 from the Phase 7 plan: analysis of pathion (32D) general-form
zero-divisor spectrum, scaling laws, and classification.

References:
- de Marrais (box-kites): Zero-divisor structure in sedenions
- Moreno & Recht (2007): ZD characterization in CD algebras
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv


# Try to import Rust accelerated kernels
try:
    import gororoba_kernels
    HAS_RUST_KERNELS = True
except ImportError:
    HAS_RUST_KERNELS = False


def count_2blade_zero_divisors(dim: int, atol: float = 1e-10) -> int:
    """
    Count 2-blade zero-divisors in dim-dimensional CD algebra.

    A 2-blade is an element of form e_i + e_j.
    """
    if HAS_RUST_KERNELS and dim >= 16:
        zd_pairs = gororoba_kernels.find_zero_divisors(dim, atol)
        return len(zd_pairs)
    else:
        raise NotImplementedError("Requires Rust kernels for dim >= 16")


def zd_spectrum_histogram(
    dim: int,
    n_samples: int = 10000,
    n_bins: int = 50,
    seed: int = 42
) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute histogram of product norms ||a*b|| for random pairs.

    Returns
    -------
    min_norm : float
        Minimum observed norm.
    max_norm : float
        Maximum observed norm.
    mean_norm : float
        Mean observed norm.
    histogram : np.ndarray
        Bin counts.
    """
    if HAS_RUST_KERNELS:
        min_n, max_n, mean_n, hist = gororoba_kernels.zd_spectrum_analysis(
            dim, n_samples, n_bins, seed
        )
        return min_n, max_n, mean_n, np.array(hist)
    else:
        raise NotImplementedError("Requires Rust kernels")


def zd_scaling_law(dims: List[int], atol: float = 1e-10) -> Dict[int, int]:
    """
    Compute ZD counts for multiple dimensions to analyze scaling.

    Returns dict mapping dim -> n_zero_divisors.
    """
    result = {}
    for d in dims:
        if d >= 16:
            result[d] = count_2blade_zero_divisors(d, atol)
        else:
            result[d] = 0  # No ZDs below sedenions
    return result


def classify_zd_by_indices(
    dim: int,
    atol: float = 1e-10
) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
    """
    Classify 2-blade ZD pairs by their index pattern.

    Categories:
    - "same_half": both indices in same half of basis (e.g., both < dim/2)
    - "cross_half": indices span both halves
    - "consecutive": indices differ by 1
    - "power_of_2": index difference is a power of 2
    """
    if not HAS_RUST_KERNELS or dim < 16:
        raise NotImplementedError("Requires Rust kernels for dim >= 16")

    zd_pairs = gororoba_kernels.find_zero_divisors(dim, atol)
    half = dim // 2

    categories = {
        "same_half": [],
        "cross_half": [],
        "consecutive": [],
        "power_of_2": [],
        "other": [],
    }

    def is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    for pair in zd_pairs:
        i, j, k, l, norm = pair

        # Check categories
        a_same_half = (i < half and j < half) or (i >= half and j >= half)
        b_same_half = (k < half and l < half) or (k >= half and l >= half)

        if abs(i - j) == 1 or abs(k - l) == 1:
            categories["consecutive"].append(pair)
        elif is_power_of_2(abs(i - j)) or is_power_of_2(abs(k - l)):
            categories["power_of_2"].append(pair)
        elif a_same_half and b_same_half:
            categories["same_half"].append(pair)
        elif not a_same_half or not b_same_half:
            categories["cross_half"].append(pair)
        else:
            categories["other"].append(pair)

    return categories


def run_pathion_zd_analysis(
    seed: int = 42,
    n_samples: int = 10000,
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Run complete pathion (32D) zero-divisor analysis.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of samples for spectrum analysis.
    output_dir : Path, optional
        Directory to save CSV artifacts.

    Returns
    -------
    dict
        Analysis results including counts, spectrum, and classification.
    """
    results = {
        "seed": seed,
        "n_samples": n_samples,
        "has_rust_kernels": HAS_RUST_KERNELS,
    }

    if not HAS_RUST_KERNELS:
        print("WARNING: Rust kernels not available. Analysis will be limited.")
        return results

    # Step 1: ZD scaling law analysis (sedenion vs pathion)
    print("Analyzing ZD scaling law...")
    scaling = zd_scaling_law([16, 32], atol=1e-10)
    results["scaling_law"] = scaling
    results["sedenion_2blade_zd"] = scaling[16]
    results["pathion_2blade_zd"] = scaling[32]
    results["pathion_sedenion_ratio"] = scaling[32] / max(scaling[16], 1)

    # Step 2: Spectrum analysis for sedenion and pathion
    print("Computing spectrum histograms...")
    sed_min, sed_max, sed_mean, sed_hist = zd_spectrum_histogram(16, n_samples, 50, seed)
    path_min, path_max, path_mean, path_hist = zd_spectrum_histogram(32, n_samples, 50, seed)

    results["sedenion_spectrum"] = {
        "min_norm": sed_min,
        "max_norm": sed_max,
        "mean_norm": sed_mean,
        "histogram": sed_hist.tolist(),
    }
    results["pathion_spectrum"] = {
        "min_norm": path_min,
        "max_norm": path_max,
        "mean_norm": path_mean,
        "histogram": path_hist.tolist(),
    }

    # Step 3: ZD classification (sedenion only - pathion too slow)
    print("Classifying sedenion ZD pairs...")
    sed_classes = classify_zd_by_indices(16, atol=1e-10)
    results["sedenion_classification"] = {k: len(v) for k, v in sed_classes.items()}

    # Save CSV if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save scaling law
        scaling_path = output_dir / "zd_scaling_law.csv"
        with open(scaling_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dim", "n_2blade_zd"])
            for dim, count in sorted(scaling.items()):
                writer.writerow([dim, count])
        print(f"Saved scaling law to {scaling_path}")

        # Save ZD spectrum
        spectrum_path = output_dir / "pathion_zd_general_form_spectrum.csv"
        with open(spectrum_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "sedenion", "pathion"])
            writer.writerow(["min_norm", sed_min, path_min])
            writer.writerow(["max_norm", sed_max, path_max])
            writer.writerow(["mean_norm", sed_mean, path_mean])
        print(f"Saved spectrum to {spectrum_path}")

        # Save classification
        class_path = output_dir / "sedenion_zd_classification.csv"
        with open(class_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "count"])
            for cat, pairs in sed_classes.items():
                writer.writerow([cat, len(pairs)])
        print(f"Saved classification to {class_path}")

    return results


if __name__ == "__main__":
    results = run_pathion_zd_analysis(
        seed=42,
        n_samples=10000,
        output_dir=Path("data/csv")
    )

    print("\n" + "=" * 60)
    print("PATHION ZD SPECTRUM ANALYSIS RESULTS")
    print("=" * 60)

    print("\nZD Scaling Law (2-blade):")
    print(f"  Sedenion (16D): {results.get('sedenion_2blade_zd', 'N/A')}")
    print(f"  Pathion (32D):  {results.get('pathion_2blade_zd', 'N/A')}")
    ratio = results.get('pathion_sedenion_ratio', 'N/A')
    print(f"  Ratio (32/16): {ratio:.2f}x" if isinstance(ratio, float) else f"  Ratio: {ratio}")

    if "sedenion_spectrum" in results:
        print("\nSedenion Spectrum:")
        spec = results["sedenion_spectrum"]
        print(f"  Min norm:  {spec['min_norm']:.4f}")
        print(f"  Max norm:  {spec['max_norm']:.4f}")
        print(f"  Mean norm: {spec['mean_norm']:.4f}")

    if "pathion_spectrum" in results:
        print("\nPathion Spectrum:")
        spec = results["pathion_spectrum"]
        print(f"  Min norm:  {spec['min_norm']:.4f}")
        print(f"  Max norm:  {spec['max_norm']:.4f}")
        print(f"  Mean norm: {spec['mean_norm']:.4f}")

    if "sedenion_classification" in results:
        print("\nSedenion ZD Classification:")
        for cat, count in results["sedenion_classification"].items():
            print(f"  {cat}: {count}")
