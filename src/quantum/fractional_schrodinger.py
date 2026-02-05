# Fractional Schrodinger Equation with Potential
# Solved via Strang Split-Operator Method
# i dpsi/dt = D |k|^alpha psi + V(x)psi

import matplotlib.pyplot as plt
import numpy as np


def solve_fractional_schrodinger_potential():
    print("--- Simulating Fractional Schrodinger with Potential ---")

    # Grid
    N = 1024
    L = 100.0
    dx = (2*L)/N
    x = np.linspace(-L, L-dx, N)
    k = np.fft.fftfreq(N, d=dx) * 2*np.pi

    # Potential: Optical Lattice + Harmonic Trap
    V0 = 2.0
    Klat = 1.0
    Omega = 0.05
    V = V0 * (1.0 - np.cos(Klat * x)) + 0.5 * Omega**2 * x**2

    # Initial State: Gaussian
    x0 = -20.0
    k0 = 1.5
    sigma = 4.0
    psi = np.exp(-0.5*((x-x0)/sigma)**2) * np.exp(1j*k0*x)
    psi /= np.linalg.norm(psi)

    # Parameters
    alpha_vals = [2.0, 1.5, 1.2] # Standard vs Fractional
    D = 0.5
    T = 40.0
    dt = 0.05
    steps = int(T/dt)

    results = []

    for alpha in alpha_vals:
        print(f"Evolving alpha={alpha}...")
        p = psi.copy()

        # Precompute operators
        expV_half = np.exp(-1j * V * (dt/2.0))
        kinetic_k = np.exp(-1j * D * (np.abs(k)**alpha) * dt)

        for _ in range(steps):
            # Strang Splitting: V/2 -> T -> V/2
            p *= expV_half
            p_k = np.fft.fft(p)
            p_k *= kinetic_k
            p = np.fft.ifft(p_k)
            p *= expV_half

            # Renormalize (numerical drift)
            p /= np.linalg.norm(p)

        results.append(np.abs(p)**2)

    # Visualization (Dark Mode)
    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#1f2937",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "font.size": 14
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    offsets = [0.0, 0.02, 0.04]
    colors = ['#60a5fa', '#34d399', '#f87171']

    for dens, alpha, off, col in zip(results, alpha_vals, offsets, colors):
        ax.plot(x, dens + off, linewidth=2.0, color=col, label=rf"$\alpha={alpha}$")

    # Plot Potential (scaled)
    ax.plot(x, V * 0.01 - 0.01, color='gray', linestyle=':', alpha=0.5, label="V(x) (scaled)")

    ax.set_title(f"Fractional Schrodinger Dynamics (Strang Split, T={T})", fontsize=20, color='white')
    ax.set_xlabel("Position x")
    ax.set_ylabel("Probability Density (Offset)")
    ax.set_xlim(-60, 60)
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.savefig("data/artifacts/images/fractional_schrodinger_potential_highres.png", dpi=300)
    print("Saved Potential Simulation.")

if __name__ == "__main__":
    solve_fractional_schrodinger_potential()
