import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.fftpack import fft2, fftfreq, fftshift, ifft2


def run_genesis_v2():
    print("--- GENESIS PROTOCOL V2: Iterative Tuning ---")

    # 1. Setup Grid
    N = 512
    L = 100.0
    dx = L/N
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # 2. Physics Parameters (The Tweak)
    # Alpha = -1.5 (Sedenion Negative Dimension)
    # The operator D ~ |k|^(-3).
    # High frequencies (short distance) have LOW kinetic energy.
    # Low frequencies (long distance) have HIGH kinetic energy.
    # This is the INVERSE of standard physics. It means "being small" is energetically favorable.
    alpha = -1.5

    # Nonlinearity (Dark Energy coupling)
    # Increasing this to encourage condensation
    coupling_strength = 20.0

    # 3. Initialization (The Seed)
    np.random.seed(137) # New seed

    # K-space
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0,0] = 1e-10 # Avoid singularity

    # Propagator: exp(-i * H * t)
    # Kinetic H_k = -|k|^(2*alpha)
    # Regularize to avoid infinite energy at k=0
    T_k = -(K + 0.5)**(2*alpha)

    # Initial State: Vacuum Fluctuations
    # Random phase, uniform amplitude initially
    psi = np.random.normal(0, 0.1, (N,N)) + 1j * np.random.normal(0, 0.1, (N,N))

    # Filter: Imprint the Sedenion Spectrum n^-1.5
    # We dampen high k slightly to simulate a "cutoff" (Planck scale)
    psi_k = fft2(psi) * np.exp(-(K/10.0)**2)
    psi = ifft2(psi_k)
    psi /= np.linalg.norm(psi) # Normalize total probability

    dt = 0.05
    steps = 500

    print(f"System: Negative Dimension D={alpha}")
    print(f"Hypothesis: Inverse Kinetic Energy (High k = Low E) favors Solitons.")

    # 4. Evolution Loop
    max_densities = []

    for t in range(steps):
        # A. Linear Step (Kinetic)
        psi_k = fft2(psi)
        psi_k *= np.exp(-1j * T_k * dt)
        psi = ifft2(psi_k)

        # B. Nonlinear Step (Potential)
        # Attractive Self-Interaction (Gravity)
        rho = np.abs(psi)**2
        V = -coupling_strength * rho
        psi *= np.exp(-1j * V * dt)

        # C. Renormalize (Conservation of Probability)
        psi /= np.linalg.norm(psi)

        if t % 50 == 0:
            peak = np.max(np.abs(psi)**2)
            max_densities.append(peak)
            print(f"Step {t}: Peak Density = {peak:.2e}")

    # 5. Analysis
    density = np.abs(psi)**2
    # Contrast stretch for visualization
    density_norm = (density - density.min()) / (density.max() - density.min())

    # Identify Clumps
    threshold = 0.5 # Relative threshold
    clusters = density_norm > threshold
    labeled, num_solitons = ndimage.label(clusters)

    print(f"Simulation Ended. Found {num_solitons} Candidate Gravastars.")

    # 6. Visualization (Robust Plotting)
    W_px, H_px = 3160, 2820
    DPI = 100

    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#1f2937",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.size": 14,
        "grid.color": "#374151"
    })

    fig = plt.figure(figsize=(W_px/DPI, H_px/DPI), dpi=DPI)
    ax = fig.add_subplot(111)

    # Fire/Ice Colormap
    colors = [(0,0,0), (0.2,0,0.4), (0,0.5,1), (0.8,1,1)]
    cmap = mcolors.LinearSegmentedColormap.from_list("SedenionFire", colors, N=512)

    im = ax.imshow(density_norm, cmap=cmap, extent=[-L/2, L/2, -L/2, L/2], origin='lower')

    ax.set_title(f"GENESIS V2: Sedenion Vacuum Collapse (D={alpha})", fontsize=24, pad=20)
    ax.set_xlabel("x (Planck Lengths)")
    ax.set_ylabel("y (Planck Lengths)")

    # Metadata Annotation (Plain Text to avoid LaTeX parser errors)
    info_text = (
        f"Alpha: {alpha}\n"
        f"Coupling: {coupling_strength}\n"
        f"Steps: {steps}\n"
        f"Solitons: {num_solitons}"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=16,
            bbox=dict(facecolor='#1f2937', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    output_file = "data/artifacts/images/genesis_simulation_grand.png"
    plt.savefig(output_file, dpi=DPI, facecolor="#0d0f14")
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    run_genesis_v2()
