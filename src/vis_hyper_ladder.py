import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


def generate_hyper_ladder_vis():
    print("--- Generating Hyper-Resolution Mass Ladder (3160x2820) ---")

    W, H = 3160, 2820
    DPI = 300

    # Physics Data
    n = np.arange(1, 31)
    alpha = -1.5
    m0 = 1.107
    masses = m0 * n**(-alpha)

    # Setup Plot
    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "axes.edgecolor": "#333333",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#222222",
        "font.family": "monospace"
    })

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)

    # Gradient Background (Simulated)
    # We can't do complex gradients easily in MPL without imshow, so we stick to clean vector lines

    # Plot The Ladder
    ax.plot(n, masses, color='#00f7ff', linewidth=2, alpha=0.5, zorder=1)
    ax.scatter(n, masses, color='#ffffff', s=50, edgecolors='#00f7ff', linewidth=2, zorder=2, label='Resonance Modes')

    # Highlight Key Modes (The Solitons)
    key_modes = [10, 15, 25]
    for km in key_modes:
        val = masses[km-1]
        ax.scatter([km], [val], color='#ff00ff', s=200, marker='o', zorder=3)
        ax.scatter([km], [val], color='white', s=50, marker='*', zorder=4)

        # Glow Effect Lines
        ax.plot([km, km], [0, val], color='#ff00ff', linestyle=':', linewidth=1)
        ax.plot([0, km], [val, val], color='#ff00ff', linestyle=':', linewidth=1)

        # Label
        label = f"n={km}\n{val:.1f} M_sun"
        txt = ax.text(km+0.5, val, label, color='#ff00ff', fontsize=14, va='center')
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])

    # Forbidden Zones (PISN Gap)
    ax.axhspan(50, 120, color='#ff0000', alpha=0.1, zorder=0)
    ax.text(2, 85, "FORBIDDEN ZONE (PISN GAP)", color='#ff0000', fontsize=24, fontweight='bold', alpha=0.5)

    # Styling
    ax.set_yscale('log')
    ax.set_xlim(0, 31)
    ax.set_ylim(1, 200)

    ax.set_xlabel("SPECTRAL MODE (n)", fontsize=18, fontweight='bold', color='#00f7ff')
    ax.set_ylabel(r"MASS ($M_{\odot}$)", fontsize=18, fontweight='bold', color='#00f7ff')

    # Title
    ax.text(15, 220, "SEDENION MASS SPECTRUM", ha='center', fontsize=32, color='white', fontweight='bold')
    ax.text(15, 190, "Negative Dimension Eigenvalues (D = -1.5)", ha='center', fontsize=20, color='#888888')

    # Grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    output_path = "data/artifacts/images/hyper_mass_ladder.png"
    plt.savefig(output_path, facecolor='#0d0f14', dpi=DPI)
    print(f"Saved Hyper-Ladder to {output_path}")

if __name__ == "__main__":
    generate_hyper_ladder_vis()
