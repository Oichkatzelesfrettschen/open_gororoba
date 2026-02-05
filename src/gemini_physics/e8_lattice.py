from itertools import permutations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_e8_roots():
    """
    Generates the 240 roots of the E8 Lie Algebra.

    Coordinates are:
    1. (+/-1, +/-1, 0, 0, 0, 0, 0, 0) - all permutations (112 roots)
    2. (+/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2, +/-1/2)
       - even number of minus signs (128 roots)
    """
    roots = []

    # 1. Permutations of (+/-1, +/-1, 0, 0, 0, 0, 0, 0)
    for p in permutations([1, 1, 0, 0, 0, 0, 0, 0]):
        for signs in product([1, -1], repeat=2):
            root = np.zeros(8)
            idx = 0
            for i in range(8):
                if p[i] == 1:
                    root[i] = signs[idx]
                    idx += 1
            roots.append(root)

    # Remove duplicates from permutations
    roots = np.unique(np.array(roots), axis=0)

    # 2. (+/-1/2, ..., +/-1/2) with even minus signs
    for s in product([0.5, -0.5], repeat=8):
        if sum(1 for x in s if x < 0) % 2 == 0:
            roots = np.vstack([roots, np.array(s)])

    return np.unique(roots, axis=0)

def project_roots(roots):
    """
    Projects 8D roots to 2D using a random (but deterministic) orthogonal projection.
    Simulates a 2D Quasicrystal projection.
    """
    np.random.seed(42)
    # Generate an 8x2 projection matrix
    projection_matrix = np.random.randn(8, 2)
    # Orthogonalize
    q, _ = np.linalg.qr(projection_matrix)

    projected = roots @ q
    return projected

def synthesize_e8_data():
    """
    Saves the exact E8 roots to the framework folder.
    Fulfills Step 4 of the Roadmap.
    """
    roots = generate_e8_roots()
    df = pd.DataFrame(roots, columns=[f'dim_{i}' for i in range(1, 9)])

    target_path = "curated/01_theory_frameworks/e8_root_system_exact.csv"
    df.to_csv(target_path, index=False)
    print(f"Saved {len(roots)} E8 roots to {target_path}")

    # Visualization (Step 13)
    projected = project_roots(roots)
    plt.figure(figsize=(10, 10))
    plt.scatter(projected[:, 0], projected[:, 1], s=20, alpha=0.7, c='indigo')
    plt.title("2D Projection of the 8D E8 Lattice (Root System)", fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)

    viz_path = "curated/01_theory_frameworks/e8_quasicrystal_projection.png"
    plt.savefig(viz_path)
    print(f"Saved E8 Quasicrystal visualization to {viz_path}")

if __name__ == "__main__":
    synthesize_e8_data()
