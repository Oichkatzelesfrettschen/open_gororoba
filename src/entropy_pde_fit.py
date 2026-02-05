import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace

# Extended PDE Fit for N-Dimensions

def run_pde_sim_nd(shape, depth, theta_twist, D, gamma, alpha3):
    # shape: tuple (L,) or (L, W) or (L, W, H)...
    n_sites = np.prod(shape)

    # Initialize entropy field S
    # Initial state: localized low entropy or random?
    # Let's assume a "quench" scenario where S starts low and grows.
    S = np.zeros(shape)

    # Add some initial noise/seed
    # Center seed
    center = tuple(s // 2 for s in shape)
    S[center] = 0.1

    history = [S.copy()]
    dt = 0.1

    # Source term J (constant injection or boundary?)
    # Let's assume J is implicit in growth or constant 0.1 global
    J = 0.05

    for t in range(depth):
        # Laplacian
        # wrap boundary conditions
        lap = laplace(S, mode='wrap')

        # Reaction
        # dS/dt = D Lap S - gamma S - alpha S^3 + J
        dS = D * lap - gamma * S - alpha3 * S**3 + J

        S = S + dS * dt
        # Clamp S to positive and saturation?
        S = np.maximum(S, 0)
        # Saturation proxy (soft)

        history.append(S.copy())

    return np.array(history)

def main():
    # Parameters
    depth = 50
    theta = 0.5

    # Map theta to D (heuristic)
    D = np.sin(theta)**2
    gamma = 0.01
    alpha3 = 0.1

    # Run 1D, 2D, 3D, 4D
    shapes = [
        (20,),
        (10, 10),
        (6, 6, 6),
        (4, 4, 4, 4)
    ]

    results = {}

    for sh in shapes:
        dim_label = f"{len(sh)}D"
        print(f"Simulating {dim_label} PDE...")
        hist = run_pde_sim_nd(sh, depth, theta, D, gamma, alpha3)

        # Record mean entropy over time
        mean_S = np.mean(hist, axis=tuple(range(1, len(sh)+1)))
        results[dim_label] = mean_S

    # Plot
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(data, label=label)

    plt.xlabel('Time Step')
    plt.ylabel('Mean Entropy')
    plt.title('Recursive Entropy Growth across Dimensions')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/artifacts/entropy_pde_nd.png')
    print("PDE Simulation complete.")

if __name__ == "__main__":
    main()
