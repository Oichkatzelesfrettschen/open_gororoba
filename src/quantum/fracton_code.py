# Fracton-Like CSS Stabilizer Prototype
# Using Sierpinski Mask for Logical Operators

import matplotlib.pyplot as plt
import numpy as np


def generate_fracton_code():
    print("--- Generating Fracton Stabilizer Code Visualization ---")

    L = 65 # Power of 2 + 1 for clean Sierpinski
    grid = np.zeros((L, L), dtype=int)

    # Seed Rule-90
    grid[0, L//2] = 1
    for r in range(1, L):
        left = np.roll(grid[r-1], 1, axis=0)
        right = np.roll(grid[r-1], -1, axis=0)
        grid[r] = left ^ right

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

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='magma', origin='upper', interpolation='nearest')

    ax.set_title("Sierpinski Fractal Stabilizer Support (L=65)", fontsize=18, color='white')
    ax.axis('off')

    plt.savefig("data/artifacts/images/fracton_stabilizer_mask.png", dpi=300)
    print("Saved Fracton Mask.")

if __name__ == "__main__":
    generate_fracton_code()
