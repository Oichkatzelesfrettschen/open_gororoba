# EXPERIMENT 2D -- Absorber Pareto front: edge mass vs interior distortion (fractional free, \alpha=1.5)
# Sweep m in {4,6,8}, eta in {2e-4,5e-4,1e-3}, x_c in {110, 120, 130}; T=160, aggressive IC.
# Metrics:
#  - Edge mass M_edge = int_{|x|>x_c+\delta} |\psi(x,T)|^2 dx, \delta=5
#  - Interior distortion E_int = ||\psi_abs - \psi_free||_{L2(|x|<x_c-\delta)} at T
# Single scatter chart, dark mode, 3160x2820, labeled points.

import sys

import matplotlib.pyplot as plt
import numpy as np


# Compatibility wrapper for numpy integration
def integrate(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def run_experiment():
    print("--- Running Absorber Pareto Sweep (Robust) ---")

    try:
        # Rendering
        W, H = 3160, 2820
        dpi = 100
        fig_w, fig_h = W / dpi, H / dpi
        plt.rcParams.update({
            "figure.facecolor": "#0d0f14",
            "axes.facecolor": "#0d0f14",
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "font.size": 16
        })

        # Grid and parameters
        N = 1024
        L = 200.0
        dx = (2 * L) / N
        x = np.linspace(-L, L - dx, N)
        k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
        alpha = 1.5
        D = 0.5
        T = 160.0
        dt = 0.25
        steps = int(np.round(T / dt))
        delta = 5.0

        # Aggressive IC: right-moving packet
        x0 = -120.0
        k0 = 1.2
        sig = 8.0
        psi0 = np.exp(-0.5 * ((x - x0) / sig) ** 2) * np.exp(1j * k0 * x)

        # Normalize
        norm = np.sqrt(integrate(np.abs(psi0) ** 2, x))
        if norm == 0:
            raise ValueError("Normalization factor is zero.")
        psi0 /= norm

        phase_k = np.exp(-1j * D * (np.abs(k) ** alpha) * dt)

        # Free final
        print("Computing free evolution baseline...")
        psi_free = psi0.copy()
        for _ in range(steps):
            psi_k = np.fft.fft(psi_free)
            psi_k *= phase_k
            psi_free = np.fft.ifft(psi_k)

        def run_abs(eta, m, x_c):
            absorb = np.ones_like(x)
            mask = np.abs(x) > x_c
            absorb[mask] = np.exp(-eta*(np.abs(x[mask]) - x_c)**m)
            psi = psi0.copy()
            for _ in range(steps):
                psi_k = np.fft.fft(psi)
                psi_k *= phase_k
                psi = np.fft.ifft(psi_k)
                psi *= absorb
            return psi

        # Sweep Parameters
        orders = [4,6,8]
        etas = [2e-4, 5e-4, 1e-3]
        xcs = [110.0, 120.0, 130.0] # Missing in previous version

        points = []
        total_runs = len(orders) * len(etas) * len(xcs)
        count = 0

        print(f"Starting parameter sweep ({total_runs} combinations)...")

        for m in orders:
            for eta in etas:
                for xc in xcs:
                    psi_abs = run_abs(eta, m, xc)
                    # Edge mass
                    emask = np.abs(x) > (xc + delta)
                    M_edge = float(integrate(np.abs(psi_abs[emask])**2, x[emask]))
                    # Interior distortion
                    imask = np.abs(x) < (xc - delta)
                    diff_sq = np.abs(psi_abs[imask] - psi_free[imask])**2
                    E_int = float(np.sqrt(integrate(diff_sq, x[imask])))
                    points.append((M_edge, E_int, m, eta, xc))

                    count += 1
                    if count % 5 == 0:
                        print(f"Progress: {count}/{total_runs}")

        # Scatter plot with text labels
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        for (Me, Ei, m, eta, xc) in points:
            ax.scatter(Me, Ei, s=45, alpha=0.9)
            label = f"m={m},eta={eta:.0e},x_c={int(xc)}"
            ax.text(Me, Ei, label, fontsize=9, ha="left", va="bottom", color="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Edge mass M_edge (down better)")
        ax.set_ylabel("Interior distortion E_int (down better)")
        ax.set_title("Absorber Pareto: Edge mass vs Interior distortion (\alpha=1.5)\n"
                     fr"T={T}, dt={dt}, IC aggressive", fontsize=28, loc="left")
        ax.grid(True, which="both", alpha=0.2)

        outp = "data/artifacts/images/absorber_pareto_3160x2820.png"
        plt.savefig(outp, dpi=dpi, facecolor="#0d0f14")
        print(f"Saved: {outp}")

    except Exception as e:
        print(f"CRITICAL FAILURE in absorber_pareto: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_experiment()
