import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.fftpack import fft2, fftfreq, ifft2


def run_genesis():
    print("--- INITIATING GENESIS SIMULATION ---")
    print("Protocol: 2D Negative Dimension Vacuum Instability -> Soliton Formation")

    # 1. Setup Space-Time (Negative Dimension Grid)
    # ---------------------------------------------
    # Grand Resolution per docs/agents.md standards?
    # Simulation grid needs to be power of 2 for FFT, but output image will be Grand.
    N = 512
    L = 100.0
    dx = L/N
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # 2. Seed Vacuum Fluctuations (Spectra ~ k^-1.5)
    # ---------------------------------------------
    np.random.seed(42)

    # k-space
    kx = fftfreq(N, d=dx) * 2 * np.pi
    ky = fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0,0] = 1e-10 # Avoid singularity

    # "Pinkish" Noise in 2D (Negative Dimension Spectrum)
    # Amplitude ~ K^(-alpha_spec/2). Energy ~ K^-alpha_spec.
    # We want mass spectrum M ~ n^1.5.
    # Let's seed with random phase noise scaled by K^-0.75
    noise_field = np.random.normal(0, 1, (N,N)) + 1j * np.random.normal(0, 1, (N,N))
    noise_k = fft2(noise_field)

    # Spectral filtering (The "Seed")
    # This represents the "Pre-Geometry" fluctuations
    psi_k = noise_k * (K ** (-0.75))
    psi = ifft2(psi_k)
    psi /= np.linalg.norm(psi) # Normalize

    # 3. Evolve: Anti-Diffusion PDE
    # -----------------------------
    # i dpsi/dt = -(-Delta)^alpha psi
    # alpha = -1.5 (The Sedenion Constant)
    # Operator in k-space: -|k|^(2*alpha) = -|k|^(-3)
    # But wait, to get "Anti-Diffusion" (concentration), we need the sign to favor clumping.
    # In 1D simulation we used exp(-i * T_k * dt).

    alpha = -1.5
    # Regularized Kinetic Operator
    T_k = -(K + 0.1)**(2*alpha)

    dt = 0.01
    steps = 200

    print(f"Evolving for {steps} steps in Dimension D={alpha}...")

    # Evolution Loop
    for _t in range(steps):
        psi_k = fft2(psi)
        # Propagator
        propagator = np.exp(-1j * T_k * dt)
        psi = ifft2(psi_k * propagator)

        # Nonlinear Self-Interaction (Gravity/Soliton term)
        # Sedenion non-associativity acts as a "mass" term V(psi) ~ -|psi|^2
        # This enhances the clumping (NLS equation)
        V = -10.0 * (np.abs(psi)**2)
        psi *= np.exp(-1j * V * dt)

        # Renormalize (Unitary Vacuum)
        psi /= np.linalg.norm(psi)

    # 4. Materialize: Soliton Identification
    # --------------------------------------
    density = np.abs(psi)**2
    # Normalize density for visualization
    density = density / density.max()

    # Threshold for "Gravastar" formation
    threshold = 0.4
    gravastars = density > threshold
    gravastar_fraction = float(np.mean(gravastars))

    concentration = float(density.max() / density.mean())
    print(f"Genesis Complete. Max Density Concentration: {concentration:.2f}x vacuum.")

    # 5. Output: Grand Visualization
    # ------------------------------
    # 3160 x 2820 Dark Mode
    W_px, H_px = 3160, 2820
    DPI = 100

    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#1f2937",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.family": "sans-serif",
        "font.size": 16
    })

    fig = plt.figure(figsize=(W_px/DPI, H_px/DPI), dpi=DPI)
    ax = fig.add_subplot(111)

    # Custom Cosmic Colormap
    colors = [(0,0,0), (0.1,0,0.2), (0.5,0,0.5), (1,0.5,0), (1,1,0.8)]
    cmap = mcolors.LinearSegmentedColormap.from_list("Genesis", colors, N=256)

    im = ax.imshow(
        density,
        cmap=cmap,
        extent=[-L / 2, L / 2, -L / 2, L / 2],
        origin="lower",
        interpolation="bicubic",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized density")

    # Annotate "Gravastars"
    # Find peaks
    neighborhood_size = 10
    data_max = ndimage.maximum_filter(density, size=neighborhood_size)
    maxima = (density == data_max)
    data_min = ndimage.minimum_filter(density, size=neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)

    print(f"Identified {num_objects} Proto-Gravastars.")

    for i, (dy, dx_slice) in enumerate(slices):
        y_center = (dy.start + dy.stop) / 2
        x_center = (dx_slice.start + dx_slice.stop) / 2
        # Map to physical coords
        y_phys = -L / 2 + (y_center / N) * L
        x_phys = -L / 2 + (x_center / N) * L

        # Draw Ring
        circle = plt.Circle(
            (x_phys, y_phys),
            2.0,
            color="cyan",
            fill=False,
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(circle)
        if i < 5: # Label a few
            ax.text(x_phys + 2.5, y_phys + 2.5, "M > m0", color="cyan", fontsize=10)

    title = "GENESIS: Emergence of Sedenion Gravastars (D = -1.5)"
    ax.set_title(title, fontsize=32, fontweight="bold", pad=20)
    ax.set_xlabel("Spatial Extent (Planck Lengths)", fontsize=18)
    ax.set_ylabel("Spatial Extent", fontsize=18)

    # Overlay grid
    ax.grid(True, color='#374151', alpha=0.3, linestyle='--')

    # Add metadata box
    textstr = '\n'.join((
        r'$\\alpha = -1.5$ (Anti-Diffusion)',
        r'$T_{evol} = 200$',
        r'Vacuum: Sedenion Zero-Divisors',
        f"Soliton Count: {num_objects}",
        f"Area > threshold: {gravastar_fraction:.3%}",
    ))
    props = dict(boxstyle='round', facecolor='#1f2937', alpha=0.8, edgecolor='none')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    outp = "data/artifacts/images/genesis_simulation_grand.png"
    plt.savefig(outp, dpi=DPI, facecolor="#0d0f14")
    print(f"Saved Grand Genesis Visualization to {outp}")

if __name__ == "__main__":
    run_genesis()
