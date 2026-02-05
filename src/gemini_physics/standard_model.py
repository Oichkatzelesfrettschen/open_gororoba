import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .optimized_algebra import cd_multiply_jit
except ImportError:
    # Fallback if running as script
    import sys
    sys.path.append(os.path.dirname(__file__))
    from optimized_algebra import cd_multiply_jit

class SedenionStandardModel:
    """
    Maps the 16D Sedenion Algebra to the Standard Model Gauge Groups.

    Hypothesis:
    - e0: Scalar (Higgs / Vacuum)
    - e1..e8: SU(3) Color Sector (Gluons)
    - e9..e11: SU(2) Weak Sector (W+, W-, Z0)
    - e12: U(1) Hypercharge (Photon/B)
    - e13..e15: BSM (Beyond Standard Model) / Dark Sector
    """
    def __init__(self):
        self.dim = 16
        self.labels = {
            0: "Scalar",
            1: "Gluon_1", 2: "Gluon_2", 3: "Gluon_3", 4: "Gluon_4",
            5: "Gluon_5", 6: "Gluon_6", 7: "Gluon_7", 8: "Gluon_8",
            9: "Weak_1", 10: "Weak_2", 11: "Weak_3",
            12: "Hypercharge",
            13: "BSM_1", 14: "BSM_2", 15: "BSM_3"
        }

    def project_flux(self, element):
        """Calculates the magnitude of the element in each sector."""
        mag = element**2
        sectors = {
            "Scalar": mag[0],
            "QCD (Gluons)": np.sum(mag[1:9]),
            "Weak (SU2)": np.sum(mag[9:12]),
            "QED (U1)": mag[12],
            "Dark Sector": np.sum(mag[13:])
        }
        return sectors

    def simulate_interaction(self, n_steps=1000):
        """
        Simulates random interactions (multiplications) and tracks
        where the 'Associator Energy' leaks.

        If (ab)c != a(bc), energy is effectively 'created' or 'lost'
        relative to a standard associative theory. We track which sector
        this non-associative flux enters.
        """
        print(f"Simulating {n_steps} Sedenion Interactions mapped to SM...")

        flux_history = []

        for _ in range(n_steps):
            # Random input vectors (fields)
            a = np.random.uniform(-1, 1, 16)
            b = np.random.uniform(-1, 1, 16)
            c = np.random.uniform(-1, 1, 16)

            # Calculate Associator: [a,b,c] = (ab)c - a(bc)
            ab_c = cd_multiply_jit(cd_multiply_jit(a, b, 16), c, 16)
            a_bc = cd_multiply_jit(a, cd_multiply_jit(b, c, 16), 16)

            associator = ab_c - a_bc

            # Where does the non-associativity live?
            flux = self.project_flux(associator)
            flux_history.append(flux)

        return pd.DataFrame(flux_history)

def run_standard_model_sim():
    sm = SedenionStandardModel()
    df = sm.simulate_interaction(5000)

    # Calculate average leakage per sector
    means = df.mean().sort_values(ascending=False)
    print("\nAverage Non-Associative Flux per Sector:")
    print(means)

    # Visualization
    plt.figure(figsize=(10, 6))
    means.plot(kind='bar', color=['red', 'orange', 'green', 'blue', 'purple'])
    plt.title("Violation of Associativity by Particle Sector (Sedenion Model)", fontsize=14)
    plt.ylabel("Mean Associator Magnitude (|ab)c - a(bc)|^2)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    viz_path = "curated/01_theory_frameworks/standard_model_associativity_leak.png"
    plt.savefig(viz_path)
    print(f"Saved SM visualization to {viz_path}")

    # Save Data
    csv_path = "curated/01_theory_frameworks/sedenion_standard_model_flux.csv"
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_standard_model_sim()
