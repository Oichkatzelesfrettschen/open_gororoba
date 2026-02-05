import os

import numpy as np
import pandas as pd


def check_convergence(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (File not found)")
        return

    print(f"\nChecking convergence for: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        # Check if rows converge (variance decreases across columns or rows)
        # Assuming columns are iteration steps

        # Calculate row-wise variance
        row_vars = df.var(axis=1)
        mean_variance = row_vars.mean()

        # Calculate iteration-to-iteration delta
        diffs = df.diff(axis=1).abs().mean(axis=0)

        print(f"Mean Row Variance: {mean_variance:.4f}")
        print("Average Step-to-Step Change:")
        print(diffs.head(5).to_string())

        if diffs.iloc[-1] < diffs.iloc[1]:
            print(">> TREND: CONVERGING")
        else:
            print(">> TREND: DIVERGING or STABLE NOISE")

    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    base_path = "curated/02_simulations_pde_quantum"
    files = [
        "AI-Optimized_Recursive_Tensor_Quantum_Field_Renormalization_in_High-Energy_Physics.csv"
    ]

    for f in files:
        check_convergence(os.path.join(base_path, f))
