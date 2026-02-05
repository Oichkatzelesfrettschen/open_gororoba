import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Re-using the vector algebra from sedenion_field_sim.py concepts,
# but making it a standalone searcher.

def cd_conj(x):
    """Conjugate of a Cayley-Dickson element."""
    n = x.shape[0]
    if n == 1:
        return x
    half = n // 2
    a, b = x[:half], x[half:]
    # (a, b)* = (a*, -b)
    return np.concatenate((cd_conj(a), -b))

def cd_mul(x, y):
    """Recursive Cayley-Dickson product."""
    n = x.shape[0]
    if n == 1:
        return x * y
    half = n // 2
    a, b = x[:half], x[half:]
    c, d = y[:half], y[half:]

    # (a,b)(c,d) = (ac - d*b, da + bc*)
    # Note: d* is conj(d)

    ac = cd_mul(a, c)
    d_star = cd_conj(d)
    db = cd_mul(d_star, b)

    da = cd_mul(d, a)
    c_star = cd_conj(c)
    bc = cd_mul(b, c_star)

    return np.concatenate((ac - db, da + bc))

def find_nilpotents(dim=16, samples=1000):
    """
    Search for elements x such that x*x = 0 (Nilpotency index 2).
    In standard Euclidean Sedenions, norm(x*x) = norm(x)^2 (mostly),
    but we know Sedenions are not composition algebras.
    However, for x != 0, x*x=0 implies zero divisors are present in the multiplication chain.
    Strictly speaking, in R^16, x*x=0 implies x=0 if the norm is multiplicative.
    But Sedenions fail norm multiplicativity.
    """
    print(f"Searching for Nilpotents (x^2 ~ 0) in {dim}D...")

    candidates = []

    # 1. Search Basis pairs (Zero Divisors)
    # We know (e_i + e_j) * (e_k + e_l) = 0 exists.
    # Do we have x^2 = 0?
    # x = e_i + e_j. x^2 = (e_i + e_j)(e_i + e_j) = e_i^2 + e_i e_j + e_j e_i + e_j^2
    # = -1 + e_i e_j - e_i e_j - 1 = -2.
    # So sum of two basis elements squares to -2 (scalar).

    # We need x such that x^2 = 0.
    # This requires 'split' signature usually.
    # But let's check if the 'Associator Anomaly' allows for effective nilpotency in projections.

    # Let's search for "Generator-like" zero divisors: pairs (u, v) such that uv=0.
    # We will treat these pairs as the 'roots'.

    # Load the Zero Divisor Graph edges if available, else generate
    # Generating fresh for standalone script

    basis_indices = range(1, dim)

    # Monte Carlo search for x*y=0
    # Fix x, find y

    found_pairs = []

    for _ in range(samples):
        # Construct random integer vector (-1, 0, 1)
        # to simulate discrete algebraic elements
        ix = np.random.choice(basis_indices, 2, replace=False)
        x = np.zeros(dim)
        x[ix[0]] = 1; x[ix[1]] = 1 # e_i + e_j

        # Try to find y
        iy = np.random.choice(basis_indices, 2, replace=False)
        y = np.zeros(dim)
        y[iy[0]] = 1; y[iy[1]] = -1 # e_k - e_l (try signs)

        prod = cd_mul(x, y)
        if np.linalg.norm(prod) < 1e-9:
            found_pairs.append((ix, iy))

    return found_pairs

def map_to_e6_roots(pairs):
    """
    E6 has 72 roots. We check if our zero-divisor set has similar cardinality
    or structure (symmetry).
    """
    unique_elements = set()
    for p in pairs:
        unique_elements.add(tuple(p[0]))
        unique_elements.add(tuple(p[1]))

    print(f"Unique Zero-Divisor Components found: {len(unique_elements)}")
    print(f"E6 Root Count Target: 72")

    return list(unique_elements)

if __name__ == "__main__":
    # Increasing samples significantly to ensure we capture the full 48-component structure.
    # We also use a larger search space for pairs.
    pairs = find_nilpotents(16, 200000)
    roots = map_to_e6_roots(pairs)

    # Save to CSV
    df = pd.DataFrame(pairs, columns=["Element_A_Indices", "Element_B_Indices"])
    df.to_csv("data/csv/sedenion_nilpotent_candidates.csv", index=False)
    print("Saved candidates to data/csv/sedenion_nilpotent_candidates.csv")
