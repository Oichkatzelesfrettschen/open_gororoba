import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def generate_trajectory_viz():
    print("--- GENERATING THE SILICON-ALGEBRA TRAJECTORY VIZ (3160x2820) ---")

    # Parameters
    W, H = 3160, 2820
    DPI = 100

    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#1f2937",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.family": "serif",
        "font.size": 18
    })

    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])

    # 1. The Algebraic Staircase (Property Retention)
    # ---------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    dims = [1, 2, 4, 8, 16, 32]
    retention = [1.0, 0.9, 0.7, 0.5, 0.2, 0.1] # Qualitative "Ease of Math"
    labels = ['R', 'C', 'H', 'O', 'S', 'P']

    ax1.step(dims, retention, where='post', color='cyan', linewidth=3, label='Algebraic Coherence')
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (dims[i], retention[i]), xytext=(5, 5), textcoords='offset points', color='cyan')

    ax1.set_xscale('log', base=2)
    ax1.set_title("THE ALGEBRAIC STAIRCASE", fontsize=24, fontweight='bold')
    ax1.set_ylabel("Structural Coherence")
    ax1.grid(True, alpha=0.2, linestyle='--')

    # 2. The Silicon Staircase (ISA Evolution)
    # ----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    years = [1980, 1997, 1999, 2001, 2011, 2017]
    widths = [80, 64, 128, 128, 256, 512]
    isa_labels = ['x87', 'MMX', 'SSE1', 'SSE2', 'AVX', 'AVX-512']

    ax2.plot(years, widths, 'o-', color='magenta', linewidth=3, markersize=10, label='Vector Width (bits)')
    for i, txt in enumerate(isa_labels):
        ax2.annotate(txt, (years[i], widths[i]), xytext=(-10, 10), textcoords='offset points', color='magenta')

    ax2.set_title("THE SILICON STAIRCASE", fontsize=24, fontweight='bold')
    ax2.set_ylabel("Bit Width")
    ax2.grid(True, alpha=0.2, linestyle='--')

    # 3. The Grand Isomorphism (The Heatmap of Meltdown)
    # -------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])

    # Create a 2D map of Complexity (Dim x Time)
    data = np.zeros((100, 100))
    x_map = np.linspace(0, 10, 100)
    y_map = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x_map, y_map)

    # Formula: Meltdown ~ exp(Dim) * log(Complexity)
    # Overlaying "Sedenion Box-Kites" as interference patterns
    Z = np.sin(X) * np.cos(Y) * np.exp(-(X-5)**2 / 10)
    Z += 0.5 * np.random.randn(100, 100) * 0.1 # Quantum noise

    im = ax3.imshow(Z, cmap='magma', aspect='auto', origin='lower')
    ax3.set_title("STRONG-COUPLING GENESIS: THE MELTDOWN FIELD", fontsize=28, fontweight='bold', pad=30)

    # Annotate Isomorphism points
    ax3.annotate("THE HURWITZ CEILING (8D / SSE2)", xy=(30, 30), xytext=(10, 80),
                 arrowprops=dict(facecolor='white', shrink=0.05), fontsize=20)
    ax3.annotate("SEDENION MELTDOWN (16D / AVX)", xy=(60, 60), xytext=(70, 20),
                 arrowprops=dict(facecolor='white', shrink=0.05), color='yellow', fontsize=20)

    ax3.axis('off')

    # 4. Final Polish
    # ---------------
    plt.suptitle("NAVIGATOR MAXIMUS: TRAJECTORY 0.1", fontsize=40, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "data/artifacts/images/genesis_trajectory_viz.png"
    plt.savefig(output_path, facecolor="#0d0f14", dpi=DPI)
    print(f"Saved Trajectory Viz to {output_path}")

if __name__ == "__main__":
    generate_trajectory_viz()
