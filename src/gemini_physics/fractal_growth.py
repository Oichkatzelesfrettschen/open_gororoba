import random

import matplotlib.pyplot as plt
import numpy as np


def simulate_dla(size=100, particles=1000):
    """
    Simulates Diffusion-Limited Aggregation (DLA).
    This provides a physical basis for 'Fractal Tensor Growth'.
    """
    grid = np.zeros((size, size), dtype=int)
    # Seed at the center
    grid[size//2, size//2] = 1

    # Boundary check helper
    def is_adjacent(x, y):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if grid[nx, ny] == 1:
                    return True
        return False

    print(f"Simulating DLA cluster with {particles} particles...")
    for _ in range(particles):
        # Start particle at a random point on a circle
        angle = random.uniform(0, 2*np.pi)
        radius = size // 2 - 2
        px = int(size//2 + radius * np.cos(angle))
        py = int(size//2 + radius * np.sin(angle))

        while True:
            # Random walk
            dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
            px += dx
            py += dy

            # Keep in bounds
            if not (0 <= px < size and 0 <= py < size):
                break

            # Check for sticking
            if is_adjacent(px, py):
                grid[px, py] = 1
                break

            # Don't wander too far from center
            if np.sqrt((px-size//2)**2 + (py-size//2)**2) > size//2:
                break

    return grid

def synthesize_fractal_data():
    """
    Saves the DLA cluster and generates a visualization.
    Fulfills Step 8 of the Roadmap.
    """
    cluster = simulate_dla(size=150, particles=3000)

    plt.figure(figsize=(10, 10))
    plt.imshow(cluster, cmap='inferno', interpolation='nearest')
    plt.title("DLA Fractal Cluster: Model for Non-Linear Material Growth", fontsize=14)
    plt.axis('off')

    viz_path = "curated/02_simulations_pde_quantum/fractal_dla_growth.png"
    plt.savefig(viz_path)
    print(f"Saved DLA visualization to {viz_path}")

    # Save raw cluster as CSV
    df = pd.DataFrame(cluster)
    csv_path = "curated/02_simulations_pde_quantum/fractal_dla_cluster_matrix.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved cluster matrix to {csv_path}")

if __name__ == "__main__":
    import pandas as pd
    synthesize_fractal_data()
