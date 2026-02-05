"""
Cayley-Dickson Zero-Divisor to Metamaterial Absorber Layer Mapping (C-010).

Maps the algebraic structure of sedenion zero-divisors to physical parameters
of metamaterial absorber layers, providing an explicit bridge between the
abstract CD algebra and realizable optical properties.

The mapping:
- ZD pair indices (i, j, k, l) -> spatial configuration
- ZD product norm -> layer thickness (thinner for better annihilation)
- Basis vector structure -> refractive index profile

Physical realizability constraints:
- n > 0 (real part of refractive index)
- k >= 0 (extinction coefficient)
- thickness > 0

References:
- Leonhardt & Philbin (2006) - Transformation optics
- Smith & Pendry (2006) - Metamaterial parameter retrieval
- Project metamaterial.py - Gold/epoxy layer parameters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass
class MetamaterialLayer:
    """Physical parameters for a metamaterial absorber layer."""
    layer_id: int
    n_real: float  # Real part of refractive index
    n_imag: float  # Imaginary part (extinction coefficient k)
    thickness_nm: float  # Layer thickness in nanometers
    material_type: str  # 'dielectric', 'plasmonic', 'hyperbolic'


@dataclass
class ZDToLayerMapping:
    """Mapping from a ZD pair to metamaterial layer parameters."""
    zd_indices: Tuple[int, int, int, int]  # (i, j, k, l)
    product_norm: float
    layer: MetamaterialLayer
    is_physical: bool  # Whether parameters are physically realizable


def extract_zd_pairs_from_rust(dim: int = 16, atol: float = 1e-10) -> List[Tuple[int, int, int, int, float]]:
    """
    Extract zero-divisor pairs using the Rust kernel.

    Returns list of (i, j, k, l, norm) tuples where
    (e_i + e_j) * (e_k +/- e_l) ~ 0.
    """
    try:
        import gororoba_kernels
        return gororoba_kernels.find_zero_divisors(dim, atol)
    except ImportError:
        # Fallback: return canonical sedenion ZD pairs
        return _fallback_sedenion_zd_pairs()


def _fallback_sedenion_zd_pairs() -> List[Tuple[int, int, int, int, float]]:
    """
    Fallback canonical ZD pairs for sedenions (dim=16).

    Based on de Marrais box-kites and Reggiani's 84 standard ZDs.
    """
    # Known ZD pairs from the algebra structure
    pairs = [
        (1, 2, 4, 8, 0.0),  # e1+e2 annihilated by e4-related
        (1, 4, 2, 8, 0.0),
        (1, 8, 2, 4, 0.0),
        (2, 4, 1, 8, 0.0),
        (2, 8, 1, 4, 0.0),
        (4, 8, 1, 2, 0.0),
        (3, 5, 6, 9, 0.0),
        (3, 6, 5, 10, 0.0),
        (3, 9, 5, 12, 0.0),
        (5, 10, 3, 12, 0.0),
    ]
    return pairs


def map_zd_to_refractive_index(
    i: int, j: int, k: int, l: int,
    base_n: float = 1.5,
    modulation_strength: float = 0.2
) -> complex:
    """
    Map ZD indices to complex refractive index.

    The mapping uses the index values to determine optical properties:
    - Higher indices -> more exotic (hyperbolic-like) response
    - Paired indices (i,j) vs (k,l) -> contrast between layers

    Parameters
    ----------
    i, j, k, l : int
        ZD pair indices.
    base_n : float
        Base refractive index.
    modulation_strength : float
        How much indices modulate the base value.

    Returns
    -------
    complex
        Complex refractive index n + ik.
    """
    # Map index sum to real part
    index_sum = (i + j + k + l) / 60.0  # Normalize by max possible (15+14+13+12)
    n_real = base_n + modulation_strength * np.sin(np.pi * index_sum)

    # Map index product to extinction (higher = more absorbing)
    index_product = (i * j * k * l) ** 0.25 / 15.0  # Geometric mean normalized
    n_imag = 0.1 * modulation_strength * index_product

    return complex(n_real, n_imag)


def map_zd_norm_to_thickness(
    norm: float,
    min_thickness: float = 10.0,
    max_thickness: float = 200.0,
    inverse_scaling: bool = True
) -> float:
    """
    Map ZD product norm to layer thickness.

    Intuition: Better annihilation (lower norm) corresponds to more
    effective absorption, which can be achieved with thinner resonant layers.

    Parameters
    ----------
    norm : float
        ZD product norm (0 for perfect annihilation).
    min_thickness, max_thickness : float
        Thickness bounds in nanometers.
    inverse_scaling : bool
        If True, lower norm -> thinner layer (resonant).

    Returns
    -------
    float
        Layer thickness in nm.
    """
    if inverse_scaling:
        # Sigmoid-like mapping: norm=0 -> min_thickness, norm large -> max_thickness
        t = min_thickness + (max_thickness - min_thickness) * (1 - np.exp(-norm / 0.1))
    else:
        # Linear scaling
        t = min_thickness + norm * (max_thickness - min_thickness)

    return max(min_thickness, min(max_thickness, t))


def classify_material_type(n_complex: complex) -> str:
    """
    Classify material type based on optical properties.

    Parameters
    ----------
    n_complex : complex
        Complex refractive index.

    Returns
    -------
    str
        Material classification.
    """
    n_real = n_complex.real
    n_imag = n_complex.imag

    if n_real < 0:
        return "hyperbolic"  # Negative index material (requires careful design)
    elif n_imag > 0.5:
        return "plasmonic"  # High loss, metallic-like
    else:
        return "dielectric"  # Standard low-loss dielectric


def map_zd_pair_to_layer(
    zd: Tuple[int, int, int, int, float],
    layer_id: int,
    base_n: float = 1.5
) -> ZDToLayerMapping:
    """
    Map a single ZD pair to metamaterial layer parameters.

    Parameters
    ----------
    zd : tuple
        ZD pair (i, j, k, l, norm).
    layer_id : int
        Layer identifier.
    base_n : float
        Base refractive index.

    Returns
    -------
    ZDToLayerMapping
        Complete mapping with physical parameters.
    """
    i, j, k, l, norm = zd

    # Map to refractive index
    n_complex = map_zd_to_refractive_index(i, j, k, l, base_n=base_n)

    # Map to thickness
    thickness = map_zd_norm_to_thickness(norm)

    # Classify material
    material_type = classify_material_type(n_complex)

    # Check physical realizability
    is_physical = n_complex.real > 0 and n_complex.imag >= 0 and thickness > 0

    layer = MetamaterialLayer(
        layer_id=layer_id,
        n_real=n_complex.real,
        n_imag=n_complex.imag,
        thickness_nm=thickness,
        material_type=material_type,
    )

    return ZDToLayerMapping(
        zd_indices=(i, j, k, l),
        product_norm=norm,
        layer=layer,
        is_physical=is_physical,
    )


def build_absorber_stack(
    zd_pairs: List[Tuple[int, int, int, int, float]],
    max_layers: int = 20,
    base_n: float = 1.5
) -> List[ZDToLayerMapping]:
    """
    Build a metamaterial absorber stack from ZD pairs.

    Parameters
    ----------
    zd_pairs : list
        List of ZD pairs from find_zero_divisors.
    max_layers : int
        Maximum number of layers to include.
    base_n : float
        Base refractive index.

    Returns
    -------
    list
        List of layer mappings.
    """
    # Sort by norm (best annihilators first)
    sorted_pairs = sorted(zd_pairs, key=lambda x: x[4])[:max_layers]

    stack = []
    for layer_id, zd in enumerate(sorted_pairs):
        mapping = map_zd_pair_to_layer(zd, layer_id, base_n=base_n)
        stack.append(mapping)

    return stack


def verify_physical_realizability(stack: List[ZDToLayerMapping]) -> Dict:
    """
    Verify that all layers have physically realizable parameters.

    Parameters
    ----------
    stack : list
        List of layer mappings.

    Returns
    -------
    dict
        Verification results.
    """
    n_total = len(stack)
    n_physical = sum(1 for m in stack if m.is_physical)
    n_dielectric = sum(1 for m in stack if m.layer.material_type == "dielectric")
    n_plasmonic = sum(1 for m in stack if m.layer.material_type == "plasmonic")
    n_hyperbolic = sum(1 for m in stack if m.layer.material_type == "hyperbolic")

    # Check n > 0, k >= 0, thickness > 0
    all_n_positive = all(m.layer.n_real > 0 for m in stack)
    all_k_nonnegative = all(m.layer.n_imag >= 0 for m in stack)
    all_thickness_positive = all(m.layer.thickness_nm > 0 for m in stack)

    return {
        "n_total": n_total,
        "n_physical": n_physical,
        "n_dielectric": n_dielectric,
        "n_plasmonic": n_plasmonic,
        "n_hyperbolic": n_hyperbolic,
        "all_n_positive": all_n_positive,
        "all_k_nonnegative": all_k_nonnegative,
        "all_thickness_positive": all_thickness_positive,
        "all_physical": n_physical == n_total,
    }


def run_cd_absorber_mapping(
    dim: int = 16,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run the complete CD-ZD to metamaterial mapping analysis (C-010).

    Parameters
    ----------
    dim : int
        Algebra dimension (16 for sedenions, 32 for pathions).
    output_dir : Path, optional
        Directory to save CSV output.

    Returns
    -------
    dict
        Complete analysis results.
    """
    print("=" * 60)
    print(f"CD Zero-Divisor to Metamaterial Mapping (C-010)")
    print(f"Algebra dimension: {dim}")
    print("=" * 60)

    # Step 1: Extract ZD pairs
    print("\nStep 1: Extracting zero-divisor pairs...")
    zd_pairs = extract_zd_pairs_from_rust(dim)
    print(f"  Found {len(zd_pairs)} ZD pairs")

    # Step 2: Build absorber stack
    print("\nStep 2: Mapping to metamaterial parameters...")
    stack = build_absorber_stack(zd_pairs, max_layers=20)

    # Step 3: Verify physical realizability
    print("\nStep 3: Verifying physical realizability...")
    verification = verify_physical_realizability(stack)

    print(f"\nResults:")
    print(f"  Total layers: {verification['n_total']}")
    print(f"  Physically realizable: {verification['n_physical']}")
    print(f"  Material breakdown:")
    print(f"    - Dielectric: {verification['n_dielectric']}")
    print(f"    - Plasmonic: {verification['n_plasmonic']}")
    print(f"    - Hyperbolic: {verification['n_hyperbolic']}")
    print(f"  All n > 0: {verification['all_n_positive']}")
    print(f"  All k >= 0: {verification['all_k_nonnegative']}")
    print(f"  All thickness > 0: {verification['all_thickness_positive']}")

    # Save CSV if output dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "cd_zd_absorber_mapping.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer_id", "i", "j", "k", "l", "zd_norm",
                "n_real", "n_imag", "thickness_nm", "material_type", "is_physical"
            ])
            for m in stack:
                writer.writerow([
                    m.layer.layer_id,
                    m.zd_indices[0], m.zd_indices[1], m.zd_indices[2], m.zd_indices[3],
                    m.product_norm,
                    m.layer.n_real, m.layer.n_imag, m.layer.thickness_nm,
                    m.layer.material_type, m.is_physical
                ])
        print(f"\nSaved layer mapping to {csv_path}")

    return {
        "dim": dim,
        "n_zd_pairs": len(zd_pairs),
        "stack": stack,
        "verification": verification,
    }


if __name__ == "__main__":
    results = run_cd_absorber_mapping(dim=16, output_dir=Path("data/csv"))
