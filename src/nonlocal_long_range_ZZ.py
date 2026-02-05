import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Extended to support N-Dimensional lattices

def construct_hamiltonian_nd(dims, alpha, J=1.0, g=0.5):
    # dims: tuple of dimensions, e.g., (3,3) for 3x3 grid
    n = np.prod(dims)
    dim_hilbert = 2**n

    # Pauli matrices
    Sz = np.array([[1,0],[0,-1]])
    Sx = np.array([[0,1],[1,0]])
    Id = np.eye(2)

    # Map index 0..N-1 to coordinate tuple
    coords = list(itertools.product(*[range(d) for d in dims]))

    def get_dist(i, j):
        c1 = np.array(coords[i])
        c2 = np.array(coords[j])
        return np.linalg.norm(c1 - c2)

    # Recursive Kronecker product helper
    # Note: For large N this is memory intensive (2^12 x 2^12 is max practical)
    def pauli_at(op, idx, size):
        # Optimized: Sparse matrices would be better, but dense for small N is okay
        # Actually, using a list of ops and functools.reduce might be cleaner,
        # but let's stick to the previous pattern for consistency, just optimized logic.

        # Construct list then kron
        op_list = [Id] * size
        op_list[idx] = op

        res = op_list[0]
        for k in range(1, size):
            res = np.kron(res, op_list[k])
        return res

    H = np.zeros((dim_hilbert, dim_hilbert), dtype=np.complex128)

    # ZZ terms
    for i in range(n):
        for j in range(i+1, n):
            dist = get_dist(i, j)
            if dist == 0: continue

            coupling = J / (dist**alpha)

            # This kron product is the bottleneck.
            # Optimization: Precompute pauli_at results? No, memory limit.
            # Just compute on fly.
            term = np.dot(pauli_at(Sz, i, n), pauli_at(Sz, j, n))
            H += coupling * term

    # X terms
    for i in range(n):
        H += g * pauli_at(Sx, i, n)

    return H

def get_entropy(psi, n):
    cut = n // 2
    dim_A = 2**cut
    dim_B = 2**(n-cut)

    # Check if reshape is valid
    if psi.size != dim_A * dim_B:
        # Pad or truncate? Should not happen if logic is correct
        return 0

    psi_mat = psi.reshape((dim_A, dim_B))
    # SVD
    s = la.svd(psi_mat, compute_uv=False)
    # Normalize probabilities just in case
    prob = s**2
    prob = prob / np.sum(prob)

    # Entropy
    return -np.sum(prob * np.log2(prob + 1e-12))

def run_simulation_nd(dims_list, alphas):
    # dims_list: list of dimension tuples, e.g. [(8,), (3,3), (2,2,2)]

    results = {}
    dt = 0.05
    steps = 30 # Reduced steps for speed in multi-dim sweep

    for dims in dims_list:
        n = np.prod(dims)
        label = f"{len(dims)}D_{'x'.join(map(str, dims))}"
        print(f"Running simulation for {label} (n={n})...")

        slopes = []

        for alpha in alphas:
            H = construct_hamiltonian_nd(dims, alpha)

            # Start |+>
            psi = np.ones(2**n) / np.sqrt(2**n)

            # Exact Diag
            vals, vecs = la.eigh(H)
            coeffs = np.dot(vecs.conj().T, psi)

            S_t = []
            times = np.arange(steps) * dt

            for t in times:
                psi_t = np.dot(vecs, coeffs * np.exp(-1j * vals * t))
                S_t.append(get_entropy(psi_t, n))

            # Fit slope
            fit = np.polyfit(times[:10], S_t[:10], 1)
            slopes.append(fit[0])

        results[label] = slopes

    return alphas, results

if __name__ == "__main__":
    # 1D: 8 spins
    # 2D: 3x3 (9 spins) - might be slow, let's try 3x2 (6) or 2x4 (8) to keep N constant
    # 3D: 2x2x2 (8 spins)
    # 4D: 2x2x2x1? No, 2x2x1x2. Let's do hypercube 2x2x2x2 = 16 is too big (65k matrix).
    # Keep N ~ 8-10.

    # Geometries to test:
    # 1D Chain: 8
    # 2D Grid: 4x2
    # 3D Cube: 2x2x2

    geometries = [
        (8,),       # 1D
        (4, 2),     # 2D
        (2, 2, 2)   # 3D
    ]

    alphas = [0.5, 1.0, 2.0, 3.0]

    a_vals, res_dict = run_simulation_nd(geometries, alphas)

    # Plotting
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']

    # Prepare CSV data
    import pandas as pd
    csv_data = []

    for i, (label, slopes) in enumerate(res_dict.items()):
        plt.plot(a_vals, slopes, marker=markers[i], label=label)
        for alpha, slope in zip(a_vals, slopes):
            csv_data.append({'Geometry': label, 'Alpha': alpha, 'Slope': slope})

    pd.DataFrame(csv_data).to_csv('data/csv/nonlocal_slopes.csv', index=False)

    plt.xlabel('Alpha (Nonlocality decay exponent)')
    plt.ylabel('Entanglement Growth Slope')
    plt.title('Scrambling Rate vs Geometry and Nonlocality')
    plt.grid(True)
    plt.legend()
    plt.savefig('data/artifacts/nonlocal_slope_nd.png')

    print("Simulation complete. Slopes saved to plot.")
    for k, v in res_dict.items():
        print(f"{k}: {v}")
