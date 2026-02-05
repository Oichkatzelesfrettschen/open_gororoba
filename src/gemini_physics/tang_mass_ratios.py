"""
Tang & Tang 2023 sedenion-to-SU(5) mass ratio analysis.

Investigates the hypothesis that associator norms ||[a,b,c]|| in sedenion algebra
correlate with Standard Model particle masses.

References:
- Tang & Tang (2023): arXiv:2308.14768 - Sedenion SU(5) unification
- Gresnigt (2023): arXiv:2307.02505 - Unified sedenion lepton model
- Moreno & Recht (2007): Zero divisors in Cayley-Dickson algebras

Implementation notes:
- Uses Rust gororoba_kernels for batch associator computation when available
- Falls back to pure Python/NumPy implementation otherwise
- Seeded RNG for reproducibility
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


def cd_multiply_python(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pure Python Cayley-Dickson multiplication for fallback.

    Uses the standard recursive formula:
    (a,b)(c,d) = (ac - d*b, da + bc*)
    where * denotes conjugation.
    """
    dim = len(a)
    if dim == 1:
        return np.array([a[0] * b[0]])

    half = dim // 2
    aL, aR = a[:half], a[half:]
    cL, cR = b[:half], b[half:]

    # Conjugate: negate all but first component
    def conj(x):
        res = x.copy()
        res[1:] = -res[1:]
        return res

    # L = ac - d*b
    L = cd_multiply_python(aL, cL) - cd_multiply_python(conj(cR), aR)
    # R = da + bc*
    R = cd_multiply_python(cR, aL) + cd_multiply_python(aR, conj(cL))

    return np.concatenate([L, R])


def associator_python(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute associator A(a,b,c) = (ab)c - a(bc) using pure Python.
    """
    ab = cd_multiply_python(a, b)
    ab_c = cd_multiply_python(ab, c)
    bc = cd_multiply_python(b, c)
    a_bc = cd_multiply_python(a, bc)
    return ab_c - a_bc


def associator_norm(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute ||A(a,b,c)|| using Rust kernels if available.
    """
    dim = len(a)
    if HAS_RUST_KERNELS:
        return gororoba_kernels.cd_associator_norm(a, b, c, dim)
    else:
        assoc = associator_python(a, b, c)
        return float(np.linalg.norm(assoc))


def batch_associator_norms(
    a_flat: np.ndarray, b_flat: np.ndarray, c_flat: np.ndarray,
    dim: int, n_triples: int
) -> np.ndarray:
    """
    Batch compute associator norms using Rust kernels if available.
    """
    if HAS_RUST_KERNELS:
        return np.asarray(gororoba_kernels.batch_associator_norms(
            a_flat, b_flat, c_flat, dim, n_triples
        ))
    else:
        norms = []
        for i in range(n_triples):
            start = i * dim
            end = start + dim
            norm = associator_norm(
                a_flat[start:end],
                b_flat[start:end],
                c_flat[start:end]
            )
            norms.append(norm)
        return np.array(norms)


def sedenion_basis_vector(i: int, dim: int = 16) -> np.ndarray:
    """Return the i-th basis vector e_i in dim-dimensional CD algebra."""
    e = np.zeros(dim)
    e[i] = 1.0
    return e


def tang_particle_assignment() -> Dict[str, List[int]]:
    """
    Particle-to-basis-element assignment inspired by Tang & Tang 2023.

    NOTE: This is a simplified/hypothetical mapping. The original paper
    uses a more complex embedding via SU(5) -> sedenion automorphisms.
    Since the explicit mapping is not fully extractable from the paper,
    we document this as "Literature claim, mapping reconstructed."

    Assignment scheme (following sedenion -> gauge group decomposition):
    - e0: Scalar/vacuum
    - e1-e8: Color sector (gluon-like)
    - e9-e11: Weak sector
    - e12: Hypercharge
    - e13-e15: Lepton/BSM sector

    For leptons, we hypothesize:
    - electron ~ e13
    - muon ~ e14
    - tau ~ e15
    """
    return {
        # Leptons (charged)
        "electron": [13],
        "muon": [14],
        "tau": [15],
        # Leptons (neutral)
        "nu_e": [9],
        "nu_mu": [10],
        "nu_tau": [11],
        # Up-type quarks (simplified - would need color indices)
        "u": [1, 2, 3],  # 3 colors
        "c": [4, 5, 6],
        "t": [7, 8, 12],  # Uses hypercharge index
        # Down-type quarks
        "d": [1, 4, 7],
        "s": [2, 5, 8],
        "b": [3, 6, 12],
    }


def compute_lepton_associator_norms(
    rng: np.random.Generator,
    n_samples: int = 1000,
    dim: int = 16
) -> Dict[str, np.ndarray]:
    """
    Compute associator norms for triples involving lepton basis elements.

    For each lepton, samples random triples where one element is the
    lepton's basis vector, and computes the associator norm.

    Returns dict mapping lepton name to array of norms.
    """
    assignment = tang_particle_assignment()
    leptons = ["electron", "muon", "tau"]

    results = {}

    for lepton in leptons:
        indices = assignment[lepton]
        norms = []

        for idx in indices:
            e_i = sedenion_basis_vector(idx, dim)

            # Sample random partners
            for _ in range(n_samples // len(indices)):
                # Random b and c
                b = rng.uniform(-1, 1, dim)
                c = rng.uniform(-1, 1, dim)

                # Compute ||A(e_i, b, c)||
                norm = associator_norm(e_i, b, c)
                norms.append(norm)

        results[lepton] = np.array(norms)

    return results


def compute_mass_ratio_prediction(
    lepton_norms: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute predicted mass ratios from associator norm statistics.

    Hypothesis: m_i / m_j ~ mean(||A_i||) / mean(||A_j||)

    Returns predicted ratios and comparison to observed values.
    """
    observed = {
        "m_e": 0.511,  # MeV
        "m_mu": 105.66,
        "m_tau": 1776.86,
    }

    # Compute mean norms
    mean_norms = {k: np.mean(v) for k, v in lepton_norms.items()}

    # Normalize by electron
    if mean_norms["electron"] > 0:
        predicted_mu_e = mean_norms["muon"] / mean_norms["electron"]
        predicted_tau_e = mean_norms["tau"] / mean_norms["electron"]
    else:
        predicted_mu_e = 0.0
        predicted_tau_e = 0.0

    observed_mu_e = observed["m_mu"] / observed["m_e"]
    observed_tau_e = observed["m_tau"] / observed["m_e"]

    return {
        "mean_norm_e": mean_norms["electron"],
        "mean_norm_mu": mean_norms["muon"],
        "mean_norm_tau": mean_norms["tau"],
        "predicted_ratio_mu_e": predicted_mu_e,
        "predicted_ratio_tau_e": predicted_tau_e,
        "observed_ratio_mu_e": observed_mu_e,
        "observed_ratio_tau_e": observed_tau_e,
        "error_mu_e": abs(predicted_mu_e - observed_mu_e) / observed_mu_e if observed_mu_e > 0 else 0,
        "error_tau_e": abs(predicted_tau_e - observed_tau_e) / observed_tau_e if observed_tau_e > 0 else 0,
    }


def null_test_random_associators(
    rng: np.random.Generator,
    n_particles: int = 3,
    n_samples: int = 1000,
    n_permutations: int = 1000,
    dim: int = 16
) -> Dict[str, float]:
    """
    Null test: compare structured assignment vs random assignment.

    Generates random particle-to-basis mappings and computes associator
    norm statistics. If the Tang assignment is no better than random,
    the p-value will be high.

    Returns p-value and test statistics.
    """
    # First compute the "real" assignment statistics
    real_norms = compute_lepton_associator_norms(rng, n_samples, dim)
    real_stats = compute_mass_ratio_prediction(real_norms)

    # Compute error metric for real assignment
    real_error = real_stats["error_mu_e"] + real_stats["error_tau_e"]

    # Now test random assignments
    random_errors = []

    for _ in range(n_permutations):
        # Random assignment: pick 3 random basis indices for leptons
        random_indices = rng.choice(range(1, dim), size=3, replace=False)

        # Compute norms for random assignment
        random_norms = {}
        for i, lepton in enumerate(["electron", "muon", "tau"]):
            e_i = sedenion_basis_vector(int(random_indices[i]), dim)
            norms = []
            for _ in range(n_samples // 3):
                b = rng.uniform(-1, 1, dim)
                c = rng.uniform(-1, 1, dim)
                norm = associator_norm(e_i, b, c)
                norms.append(norm)
            random_norms[lepton] = np.array(norms)

        random_stats = compute_mass_ratio_prediction(random_norms)
        random_error = random_stats["error_mu_e"] + random_stats["error_tau_e"]
        random_errors.append(random_error)

    random_errors = np.array(random_errors)

    # p-value: fraction of random assignments with smaller error
    p_value = np.mean(random_errors <= real_error)

    return {
        "real_error": real_error,
        "random_error_mean": np.mean(random_errors),
        "random_error_std": np.std(random_errors),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "conclusion": "structured" if p_value < 0.05 else "not_significant",
    }


def associator_norm_gap_analysis(
    rng: np.random.Generator,
    dim: int = 16,
    n_samples: int = 5000
) -> Dict[str, float]:
    """
    Analyze the "gap" in associator norms between subalgebras.

    For quaternion (dim=4) and octonion (dim=8) subalgebras embedded
    in sedenions, the associator should be exactly zero for elements
    within the subalgebra.

    For generic sedenion elements, there should be a minimal nonzero
    associator norm delta > 0.

    This tests claim C-026.
    """
    results = {}

    # Test quaternion subalgebra (e0, e1, e2, e3)
    quat_norms = []
    for _ in range(n_samples):
        a = np.zeros(dim)
        b = np.zeros(dim)
        c = np.zeros(dim)
        a[:4] = rng.uniform(-1, 1, 4)
        b[:4] = rng.uniform(-1, 1, 4)
        c[:4] = rng.uniform(-1, 1, 4)
        norm = associator_norm(a, b, c)
        quat_norms.append(norm)

    results["quat_max_norm"] = float(np.max(quat_norms))
    results["quat_mean_norm"] = float(np.mean(quat_norms))
    results["quat_is_associative"] = results["quat_max_norm"] < 1e-10

    # Test octonion subalgebra (e0..e7)
    oct_norms = []
    for _ in range(n_samples):
        a = np.zeros(dim)
        b = np.zeros(dim)
        c = np.zeros(dim)
        a[:8] = rng.uniform(-1, 1, 8)
        b[:8] = rng.uniform(-1, 1, 8)
        c[:8] = rng.uniform(-1, 1, 8)
        norm = associator_norm(a, b, c)
        oct_norms.append(norm)

    results["oct_max_norm"] = float(np.max(oct_norms))
    results["oct_mean_norm"] = float(np.mean(oct_norms))
    # Octonions are alternative but not associative
    results["oct_is_associative"] = results["oct_max_norm"] < 1e-10

    # Test generic sedenion elements
    sed_norms = []
    for _ in range(n_samples):
        a = rng.uniform(-1, 1, dim)
        b = rng.uniform(-1, 1, dim)
        c = rng.uniform(-1, 1, dim)
        norm = associator_norm(a, b, c)
        sed_norms.append(norm)

    results["sed_min_norm"] = float(np.min(sed_norms))
    results["sed_max_norm"] = float(np.max(sed_norms))
    results["sed_mean_norm"] = float(np.mean(sed_norms))

    # The "gap" is the minimum nonzero associator norm for generic elements
    nonzero_sed = [n for n in sed_norms if n > 1e-10]
    if nonzero_sed:
        results["sed_gap_delta"] = float(np.min(nonzero_sed))
    else:
        results["sed_gap_delta"] = 0.0

    return results


def run_tang_analysis(
    seed: int = 42,
    n_samples: int = 1000,
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Run the complete Tang mass ratio analysis.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_samples : int
        Number of samples per lepton.
    output_dir : Path, optional
        Directory to save CSV artifacts.

    Returns
    -------
    dict
        Analysis results including predictions, null test, and gap analysis.
    """
    rng = np.random.default_rng(seed)

    results = {
        "seed": seed,
        "n_samples": n_samples,
        "has_rust_kernels": HAS_RUST_KERNELS,
    }

    # Step 1: Compute lepton associator norms
    print("Computing lepton associator norms...")
    lepton_norms = compute_lepton_associator_norms(rng, n_samples)

    # Step 2: Compute mass ratio predictions
    print("Computing mass ratio predictions...")
    predictions = compute_mass_ratio_prediction(lepton_norms)
    results["predictions"] = predictions

    # Step 3: Run null test (smaller sample for speed)
    print("Running null test...")
    null_test = null_test_random_associators(
        rng, n_samples=n_samples // 10, n_permutations=100
    )
    results["null_test"] = null_test

    # Step 4: Gap analysis
    print("Running gap analysis...")
    gap = associator_norm_gap_analysis(rng, n_samples=n_samples)
    results["gap_analysis"] = gap

    # Save CSV if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        csv_path = output_dir / "tang_lepton_mass_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in predictions.items():
                writer.writerow([k, v])
        print(f"Saved predictions to {csv_path}")

        # Save gap analysis
        gap_path = output_dir / "associator_gap_analysis.csv"
        with open(gap_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in gap.items():
                writer.writerow([k, v])
        print(f"Saved gap analysis to {gap_path}")

    return results


if __name__ == "__main__":
    results = run_tang_analysis(
        seed=42,
        n_samples=1000,
        output_dir=Path("data/csv")
    )

    print("\n" + "=" * 60)
    print("TANG MASS RATIO ANALYSIS RESULTS")
    print("=" * 60)

    print("\nMass Ratio Predictions:")
    pred = results["predictions"]
    print(f"  Predicted mu/e ratio: {pred['predicted_ratio_mu_e']:.4f}")
    print(f"  Observed mu/e ratio:  {pred['observed_ratio_mu_e']:.4f}")
    print(f"  Error: {pred['error_mu_e']*100:.1f}%")
    print(f"  Predicted tau/e ratio: {pred['predicted_ratio_tau_e']:.4f}")
    print(f"  Observed tau/e ratio:  {pred['observed_ratio_tau_e']:.4f}")
    print(f"  Error: {pred['error_tau_e']*100:.1f}%")

    print("\nNull Test:")
    null = results["null_test"]
    print(f"  p-value: {null['p_value']:.4f}")
    print(f"  Conclusion: {null['conclusion']}")

    print("\nGap Analysis:")
    gap = results["gap_analysis"]
    print(f"  Quaternion max ||A||: {gap['quat_max_norm']:.2e}")
    print(f"  Octonion max ||A||: {gap['oct_max_norm']:.2e}")
    print(f"  Sedenion gap delta: {gap['sed_gap_delta']:.4f}")
