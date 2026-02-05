import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace

# 4D Slicer / Hyper-Mosaic Generator

def generate_4d_entropy_data(shape=(8, 8, 8, 8), steps=20):
    # Simulate a simple diffusion-reaction in 4D to get structured data
    S = np.zeros(shape)
    center = tuple(s // 2 for s in shape)
    S[center] = 1.0

    # Simple Kernel
    for t in range(steps):
        lap = laplace(S, mode='wrap')
        S += 0.1 * lap + 0.01 * S * (1 - S)

    return S

def create_hyper_mosaic(data_4d, output_path):
    # data_4d shape: (W, Z, Y, X) -> Tiling W, Z. Image Y, X.
    W, Z, Y, X = data_4d.shape

    # We want a grid of Z slices, indexed by W.
    # Or better: A grid where rows=W, cols=Z.
    # Each cell is an (Y, X) image.

    # Canvas size
    canvas_h = W * Y
    canvas_w = Z * X

    mosaic = np.zeros((canvas_h, canvas_w))

    for w in range(W):
        for z in range(Z):
            # Extract 2D slice
            slice_2d = data_4d[w, z, :, :]

            # Place in mosaic
            y_start = w * Y
            y_end = (w + 1) * Y
            x_start = z * X
            x_end = (z + 1) * X

            mosaic[y_start:y_end, x_start:x_end] = slice_2d

    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(mosaic, cmap='inferno')
    plt.colorbar(label='Entropy / Field Value')
    plt.title(f"4D Hyper-Mosaic (Rows=W, Cols=Z)")
    plt.xlabel("X (local) / Z (global)")
    plt.ylabel("Y (local) / W (global)")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved 4D Mosaic to {output_path}")

if __name__ == "__main__":
    print("Generating 4D Data...")
    # 4D Grid: 4x4x16x16 (Small W,Z to keep mosaic readable, large X,Y for detail)
    # Or balanced: 5x5x10x10
    shape = (5, 5, 20, 20)
    S_field = generate_4d_entropy_data(shape=shape, steps=30)

    create_hyper_mosaic(S_field, 'data/artifacts/4d_entropy_mosaic.png')

    # Also save a 'time' evolution mosaic (3D + Time)
    # T, Z, Y, X -> Rows=T, Cols=Z
    # Reuse the function treating W as T
    create_hyper_mosaic(S_field, 'data/artifacts/3d_time_evolution_mosaic.png')
