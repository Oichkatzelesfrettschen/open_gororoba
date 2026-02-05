import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def modular_chaos_sim(N=1000, steps=100, alpha=1.5):
    """
    Simulates modular evolution on a large discrete state space (amplitude vector).
    U = DiagonalPhase * Fourier (approximating Modular S, T)
    """
    # State vector (complex amplitudes)
    psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)

    # Operators
    # T-like: Quadratic phase
    # exp(i * alpha * k^2 / N)
    k = np.arange(N)
    T_op = np.exp(1j * alpha * (k**2) * np.pi / N)

    entropy_history = []

    print(f"Running Modular Chaos Sim (N={N})...")

    for t in range(steps):
        # Apply T
        psi = psi * T_op

        # Apply S (Discrete Fourier Transform)
        psi = np.fft.fft(psi) / np.sqrt(N)

        # Measure Entropy (Shannon entropy of probability distribution)
        prob = np.abs(psi)**2
        # Clip for numerical safety
        prob = np.clip(prob, 1e-15, 1.0)
        prob = prob / np.sum(prob)

        H = -np.sum(prob * np.log2(prob))
        entropy_history.append(H)

    return entropy_history

if __name__ == "__main__":
    # Compare Prime vs Composite N

    results = {}
    Ns = [256, 257] # Power of 2 vs Prime

    plt.figure(figsize=(10, 6))

    for n_val in Ns:
        H = modular_chaos_sim(N=n_val)
        results[f'N={n_val}'] = H
        plt.plot(H, label=f'N={n_val} {"(Prime)" if n_val%2!=0 else "(Composite)"}')

        # Save CSV
        pd.DataFrame(results).to_csv(f'data/csv/modular_chaos_N{n_val}.csv', index=False)

    plt.axhline(np.log2(256), color='k', linestyle='--', alpha=0.3, label='Max Entropy')
    plt.title('Modular Iteration: Scrambling & Number Theoretic Resonance')
    plt.xlabel('Steps')
    plt.ylabel('Shannon Entropy')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/artifacts/modular_chaos_plot.png')
    print("Modular Chaos Sim Complete.")
