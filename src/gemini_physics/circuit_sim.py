import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .optimized_algebra import cd_multiply_jit
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from optimized_algebra import cd_multiply_jit

class SedenionCircuit:
    def __init__(self, dim=16):
        self.dim = dim

    def process_signal_chain(self, signals, structure='left_associated'):
        result = signals[0]
        if structure == 'left_associated':
            for s in signals[1:]:
                result = cd_multiply_jit(result, s, self.dim)
        elif structure == 'right_associated':
            result = signals[-1]
            for s in reversed(signals[:-1]):
                result = cd_multiply_jit(s, result, self.dim)
        return result

def run_circuit_simulation():
    print("Simulating Non-Associative Circuit Topology...")
    circuit = SedenionCircuit(16)
    diffs = []

    for _ in range(100):
        s1 = np.random.uniform(-1, 1, 16)
        s2 = np.random.uniform(-1, 1, 16)
        s3 = np.random.uniform(-1, 1, 16)

        out_A = circuit.process_signal_chain([s1, s2, s3], 'left_associated')
        out_B = circuit.process_signal_chain([s1, s2, s3], 'right_associated')

        diff = np.linalg.norm(out_A - out_B)
        diffs.append(diff)

    plt.figure(figsize=(10, 6))
    plt.plot(diffs, color='crimson')
    plt.title("Non-Reciprocity in Sedenion Circuits", fontsize=14)
    plt.savefig("curated/02_simulations_pde_quantum/non_associative_circuit_response.png")

    print(f"Average Topology Difference: {np.mean(diffs):.4f}")

if __name__ == "__main__":
    run_circuit_simulation()
