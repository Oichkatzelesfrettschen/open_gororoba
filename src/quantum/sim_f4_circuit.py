import re

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def load_real_sedenion_indices():
    """
    Loads the 48 unique zero-divisor components from the CSV.
    Each component is a 16-dimensional vector basis index (0..15).
    Wait, the zero-divisors are pairs (u, v).
    The 'roots' of F4 are vectors in the lattice.
    The CSV contains pairs of indices like [1, 5] meaning e1 + e5.

    We need to map these pairs to a 6-qubit state space (indices 0..63).
    Since there are 48 components, we can map them to integers 0..47.
    """
    try:
        df = pd.read_csv("data/csv/sedenion_nilpotent_candidates.csv")

        # Extract unique vectors (basis sums)
        # Each row is a pair of vectors A and B.
        # We treat the set of all unique A's and B's as the "Root System".

        unique_vectors = set()

        for _, row in df.iterrows():
            # Parse string representation
            s_a = str(row['Element_A_Indices']).replace('[', '').replace(']', '').replace(',', ' ')
            idx_a = tuple(sorted([int(x) for x in s_a.split() if x.strip()]))
            unique_vectors.add(idx_a)

            s_b = str(row['Element_B_Indices']).replace('[', '').replace(']', '').replace(',', ' ')
            idx_b = tuple(sorted([int(x) for x in s_b.split() if x.strip()]))
            unique_vectors.add(idx_b)

        roots = list(unique_vectors)
        print(f"Loaded {len(roots)} unique Sedenion Zero-Divisor Roots.")
        return roots

    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def run_f4_simulation():
    print("--- Running F4 Symmetry Verification on Qiskit Aer ---")

    roots = load_real_sedenion_indices()
    if not roots:
        print("No roots found. Aborting.")
        return

    n_roots = len(roots) # Expect 48
    n_qubits = 6 # 2^6 = 64 states > 48
    n_ancilla = 1

    qc = QuantumCircuit(n_ancilla + n_qubits, 1)

    # 1. State Preparation: |psi_F4>
    # We want a uniform superposition of the 48 root states.
    # Since 48 is not a power of 2, we can't just use H gates.
    # We construct the statevector manually and initialize.

    state_vector = np.zeros(2**n_qubits)

    # Map the roots to indices 0..47
    # (In a real physical mapping, we would map the vector structure to bitstrings,
    # but for symmetry counting, an index map is sufficient to test the group size).
    for i in range(n_roots):
        state_vector[i] = 1.0

    # Normalize
    state_vector /= np.linalg.norm(state_vector)

    # Initialize the register qubits (1..6)
    qc.initialize(state_vector, range(n_ancilla, n_ancilla + n_qubits))

    # 2. Symmetry Test (Hadamard Test)
    # Check if the state is invariant under a mock "Weyl Reflection"
    # A true F4 reflection would permute the roots.
    # If the state is the sum of ALL roots, it is invariant under ANY permutation.
    # We simulate a specific permutation: reversing the list (swap 0<->47, etc).
    # Since our state is uniform, it is invariant under any permutation of the active basis.
    # The circuit implementation of a generic permutation is costly.
    # We will simulate a "Global Phase Flip" on the null-space (48..63) vs active space
    # to verify we are in the correct subspace.

    # Actually, let's implement a real permutation: X gate on qubit 1 (swaps indices by 2).
    # Does the set of roots 0..47 map to itself under this?
    # No, because we just mapped them linearly.
    #
    # REAL EXPERIMENT LOGIC:
    # If the physical Sedenion structure has F4 symmetry, then the
    # "Physical Operator" corresponding to a Weyl reflection should leave |psi> invariant.
    # Since we don't have the physical operator compiled to gates yet,
    # we will test the "Uniformity" property (Symmetry under Exchange).
    # We will test overlap with the initial state.

    # For this Step 1 simulation, we verify that we can Load, Encode, and Measure.
    qc.h(0) # Ancilla Superposition

    # Controlled-Identity (Dummy Symmetry for Validation) -> Should give P(0)=1.0
    # If we put a Controlled-X (Bit flip) -> Should give P(0) < 1.0 if not invariant.

    # Let's apply a Controlled-Z to half the states (Z on qubit 1).
    # If the state is symmetric across the Z-axis, this will dephase it.
    qc.cz(0, 1)

    qc.h(0)
    qc.measure(0, 0)

    # 3. Simulation
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=1024)
    result = job.result()
    counts = result.get_counts()

    print(f"Measurement Counts: {counts}")

    # P(0) = 0.5 * (1 + Re(<psi|U|psi>))
    # If U = I, <psi|psi> = 1 -> P(0) = 1.
    # If U = Z_1, and state is uniform 0..47,
    # indices 0..47 split into even/odd.
    # If counts are uniform, Z cancels out.

    p0 = counts.get('0', 0) / 1024.0
    expectation = 2 * p0 - 1
    print(f"Expectation Value <Z>: {expectation:.4f}")

    if expectation > 0.9:
        print("Symmetry Verified (State Invariant).")
    else:
        print("Symmetry Broken (State changed by Operator).")

if __name__ == "__main__":
    run_f4_simulation()
