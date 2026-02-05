import argparse
import time

import numpy as np

# A toy benchmark script for Surreal K=8 matrix multiplication
# (Interpreted as block-matrix multiply with Octonion-like channels)

def float32_matmul_bench(N, iters):
    """
    Standard float32 matmul baseline (using BLAS via numpy).
    """
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # Warmup
    np.dot(A, B)

    t0 = time.perf_counter()
    for _ in range(iters):
        np.dot(A, B)
    t1 = time.perf_counter()

    return (t1 - t0) / iters

def surreal_matmul_bench(N, iters, check_correctness=False):
    """
    Surreal K=8 matmul simulation.
    Since we don't have a compiled kernel, we simulate the workload:
    - Treat input matrices as having shape (N, N, 8)
    - Implement the 8x8 -> 8 basis multiplication (Octonion logic)
    - This involves 64 channel-wise matmuls if done naively via basis expansion.

    Octonion multiplication rule for basis elements e_i, e_j:
    e_i * e_j = gamma_ijk * e_k

    We represent this as a tensor Gamma (8, 8, 8).
    Ideally we precompute Gamma.

    The operation is C_k = sum_{i,j} Gamma_{ijk} (A_i @ B_j)
    where A_i is the i-th channel matrix.

    For benchmarking, we explicitly execute the 64 (or fewer non-zero) matmuls
    and accumulations.
    """

    # Precompute Octonion multiplication table Gamma
    # Convention: standard Fano plane
    # (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
    # Note: Using standard (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
    # indices 1..7. 0 is identity.

    # Construct sparse Gamma
    triples = [
        (1,2,3), (1,4,5), (1,7,6),
        (2,4,6), (2,5,7),
        (3,4,7), (3,6,5)
    ]

    # Gamma[i, j] -> (sign, k)
    # We store as list of instructions: (i, j, sign, k)
    ops = []

    # e0 is identity
    for i in range(8):
        ops.append((0, i, 1.0, i))
        if i != 0:
            ops.append((i, 0, 1.0, i))

    # Squares
    for i in range(1, 8):
        ops.append((i, i, -1.0, 0))

    # Triples
    for (i,j,k) in triples:
        # e_i e_j = e_k
        ops.append((i, j, 1.0, k))
        # e_j e_i = -e_k
        ops.append((j, i, -1.0, k))

        # cycles
        # e_j e_k = e_i
        ops.append((j, k, 1.0, i))
        ops.append((k, j, -1.0, i))

        # e_k e_i = e_j
        ops.append((k, i, 1.0, j))
        ops.append((i, k, -1.0, j))

    # Prepare Data
    # 8 channels of NxN matrices
    A_channels = [np.random.randn(N, N).astype(np.float32) for _ in range(8)]
    B_channels = [np.random.randn(N, N).astype(np.float32) for _ in range(8)]
    C_channels = [np.zeros((N, N), dtype=np.float32) for _ in range(8)]

    # Warmup (1 op)
    np.dot(A_channels[0], B_channels[0])

    t0 = time.perf_counter()
    for _ in range(iters):
        # Reset C
        for k in range(8):
            C_channels[k].fill(0.0)

        # Execute the channel mix
        for (i, j, sign, k) in ops:
            # C_k += sign * (A_i @ B_j)
            prod = np.dot(A_channels[i], B_channels[j])
            if sign == 1.0:
                C_channels[k] += prod
            else:
                C_channels[k] -= prod

    t1 = time.perf_counter()

    # Optional Correctness Check vs Reference (only first iter)
    if check_correctness:
        # No ground truth to check against unless we define one,
        # but we can check if it ran without error.
        pass

    return (t1 - t0) / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    print(f"Benchmarking Matrix Multiplication (N={args.N})")

    t_f32 = float32_matmul_bench(args.N, args.iters * 10) # Run more iterations for fast baseline
    print(f"Float32 (BLAS) time: {t_f32*1000:.4f} ms")

    t_sur = surreal_matmul_bench(args.N, args.iters, args.check)
    print(f"Surreal K=8 time:    {t_sur*1000:.4f} ms")

    ratio = t_sur / t_f32
    print(f"Ratio (Overhead):    {ratio:.2f}x")
    print("(Note: Theoretical ratio for naive channel ops is ~64x + overhead)")

if __name__ == "__main__":
    main()
