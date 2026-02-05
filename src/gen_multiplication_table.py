import os
import sys

import numpy as np
import pandas as pd

# Import the engine from cd_hpl_example
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from cd_hpl_example import CDAlgebra
except ImportError:
    # Fallback if running from root
    from src.cd_hpl_example import CDAlgebra

def generate_multiplication_table(n=3):
    """
    Generates the full multiplication table for dimension 2^n.
    n=3 -> Octonions (8x8)
    n=4 -> Sedenions (16x16)
    """
    dim = 2**n
    alg = CDAlgebra(n)

    # Basis labels
    labels = ["1"] + [f"e{i}" for i in range(1, dim)]

    matrix = []

    print(f"Generating {dim}x{dim} Multiplication Table...")

    for i in range(dim):
        row = []
        basis_i = alg.basis(i)
        for j in range(dim):
            basis_j = alg.basis(j)

            # Compute product
            res = alg.multiply(basis_i, basis_j)

            # Identification
            # Result should be +/- e_k
            # Find k
            k = np.argmax(np.abs(res))
            sign = np.sign(res[k])

            val_str = labels[k]
            if sign < 0:
                val_str = "-" + val_str

            row.append(val_str)
        matrix.append(row)

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    output_path = f"data/csv/multiplication_table_{dim}D.csv"
    df.to_csv(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_multiplication_table(3) # Octonions
    generate_multiplication_table(4) # Sedenions
