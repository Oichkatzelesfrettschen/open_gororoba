import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def test_local_gpu_sim():
    print("--- Qiskit Local GPU/CPU Test ---")

    # 1. Define a heavy circuit (Quantum Volume-like) to stress test
    n_qubits = 20 # 2^20 states = ~1MB (Small, but good for logic check)
    depth = 10

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    rng = np.random.default_rng(42)

    for _ in range(depth):
        for i in range(0, n_qubits-1, 2):
            qc.cx(i, i+1)
            qc.rz(rng.random() * 2 * np.pi, i+1)

    qc.measure_all()

    print(f"Circuit Created: {n_qubits} qubits, depth {depth}")

    # 2. Check Available Simulators
    simulators = AerSimulator().available_methods()
    print(f"Available Aer Methods: {simulators}")

    # 3. Run Simulation (CPU Baseline)
    print("Running CPU Simulation...")
    sim_cpu = AerSimulator(method='statevector', device='CPU')
    t0 = time.time()
    result_cpu = sim_cpu.run(transpile(qc, sim_cpu), shots=100).result()
    t_cpu = time.time() - t0
    print(f"CPU Time: {t_cpu:.4f}s")

    # 4. Run Simulation (GPU Attempt)
    # Note: This might fallback if no GPU binary is linked, but we try.
    print("Attempting GPU Simulation...")
    try:
        sim_gpu = AerSimulator(method='statevector', device='GPU')
        t0 = time.time()
        result_gpu = sim_gpu.run(transpile(qc, sim_gpu), shots=100).result()
        t_gpu = time.time() - t0
        print(f"GPU Time: {t_gpu:.4f}s")
        print(f"Speedup: {t_cpu / t_gpu:.2f}x")
    except Exception as e:
        print(f"GPU Simulation Failed (Expected if CUDA libs missing in venv): {e}")
        print("Continuing with CPU-only for now.")

if __name__ == "__main__":
    test_local_gpu_sim()
