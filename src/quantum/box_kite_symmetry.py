import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def load_cluster_0():
    """
    Loads the 6 roots belonging to Cluster 0 (The first Box-Kite).
    """
    try:
        df = pd.read_csv("data/csv/sedenion_box_kites_clustered.csv")
        cluster_0 = df[df['Cluster_ID'] == 0]

        roots = []
        for _, row in cluster_0.iterrows():
            s = str(row['Root_Indices']).replace('(', '').replace(')', '').replace(',', ' ')
            idx = tuple(sorted([int(x) for x in s.split() if x.strip()]))
            roots.append(idx)

        print(f"Loaded {len(roots)} roots for Box-Kite 0.")
        return roots
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return []


def design_box_kite_symmetry_circuit():
    """
    Tests for Z3 x Z2 symmetry within a single Box-Kite (6 elements).
    The 6 elements of a Box-Kite typically form an Octahedron.
    Symmetries to test:
    1. Antipodal Swap (Inversion): x -> -x (or conjugate pair swap)
    2. Cyclic Permutation of 3 axes (Rotation)
    """

    roots = load_cluster_0()
    if len(roots) != 6:
        print("Error: Box-Kite must have 6 elements.")
        return None

    # Mapping: Assign an integer 0..5 to each root
    # We need 3 qubits to encode 6 states (0..5). States 6,7 are "garbage".

    n_qubits = 3
    n_ancilla = 1

    qc = QuantumCircuit(n_ancilla + n_qubits, 1)

    # 1. State Prep: Uniform Superposition of |0>..|5>
    # Hard to do perfectly with just H gates.
    # Approx: Ry rotations to split amplitudes.
    # For now, let's just create a superposition of |0> and |1> to test a single swap.
    # Actually, let's load the full superposition via statevector init.

    psi = np.zeros(2**n_qubits)
    for i in range(6):
        psi[i] = 1.0 / np.sqrt(6)

    qc.initialize(psi, range(n_ancilla, n_ancilla + n_qubits))

    # 2. Symmetry Operator: "The Antipodal Swap"
    # In an octahedron, every vertex i has an opposite vertex i+3 (mod 6).
    # We want to swap 0<->3, 1<->4, 2<->5.
    # In binary:
    # 0 (000) <-> 3 (011)
    # 1 (001) <-> 4 (100)
    # 2 (010) <-> 5 (101)

    # This isn't a simple bit flip. It's a specific permutation.
    # Let's try a simpler symmetry: The Z2 swap of the first pair.
    # Or, let's define the operator U_swap that implements 0<->3 etc.
    # This operator is our "Hypothesis" for the symmetry group of the Box-Kite.

    # Implementing 0<->3 (000 <-> 011)
    # This requires flipping bits 0 and 1 conditional on bit 2 being 0? No.
    # It's a permutation matrix.

    # Let's use a standard Controlled-SWAP test logic but for a specific permutation matrix.
    # For simulation speed, we will apply the unitary directly if we assume the
    # gate compilation is solved (we are testing the algebra, not the compiler).
    # Actually, Qiskit allows UnitaryGate.

    # Define Permutation Matrix P for Antipodal Map
    # Map: 0->3, 1->4, 2->5, 3->0, 4->1, 5->2, 6->6, 7->7 (identity on garbage)
    P = np.eye(8)
    # Swap 0,3
    P[0, 0] = 0
    P[3, 3] = 0
    P[0, 3] = 1
    P[3, 0] = 1
    # Swap 1,4
    P[1, 1] = 0
    P[4, 4] = 0
    P[1, 4] = 1
    P[4, 1] = 1
    # Swap 2,5
    P[2, 2] = 0
    P[5, 5] = 0
    P[2, 5] = 1
    P[5, 2] = 1

    from qiskit.circuit.library import UnitaryGate

    U_sym = UnitaryGate(P, label="Antipodal Swap")

    # Hadamard Test
    qc.h(0) # Ancilla

    # Controlled-U
    # Qiskit's UnitaryGate .control() method
    c_U_sym = U_sym.control(1)

    qc.append(c_U_sym, [0, 1, 2, 3]) # Ancilla + 3 qubits

    qc.h(0)
    qc.measure(0, 0)

    return qc


def run_simulation():
    qc = design_box_kite_symmetry_circuit()
    if qc is None:
        return

    print("Running Box-Kite Symmetry Test (Antipodal Swap)...")
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=2048)
    res = job.result().get_counts()

    p0 = res.get('0', 0) / 2048.0
    val = 2*p0 - 1
    print(f"Expectation Value <Z>: {val:.4f}")

    if val > 0.9:
        print("Symmetry CONFIRMED: The Box-Kite state is invariant under Antipodal Swap.")
    else:
        print("Symmetry BROKEN: The Box-Kite state is NOT invariant (Phase/Structure mismatch).")

if __name__ == "__main__":
    run_simulation()
