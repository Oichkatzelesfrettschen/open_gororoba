import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 16D Sedenion Engine (Vectorized) ---

class SedenionVectorized:
    def __init__(self, shape):
        # shape: (X, Y, Z, ...) spatial grid
        # self.data: (16, X, Y, Z, ...)
        self.shape = shape
        self.data = np.zeros((16,) + shape, dtype=np.float32)

    def set_random(self, scale=1.0):
        self.data = np.random.normal(0, scale, size=(16,) + self.shape).astype(np.float32)

    def norm(self):
        return np.sqrt(np.sum(self.data**2, axis=0))

    def multiply(self, other):
        # Implements (a,b)(c,d) = (ac - d*b, da + bc*) recursively
        # Ideally we'd use a lookup table for 16x16, but for simplicity/speed in Python:
        # We can treat S as Pairs of Octonions, Octonions as Pairs of Quaternions...
        # Or just use the precomputed multiplication table approach if grid is small.
        # For a simulation, let's do the Recursive split for vectorized ops.
        return self._mul_recursive(self.data, other.data, 16)

    def _mul_recursive(self, A, B, dim):
        if dim == 1:
            return A * B

        half = dim // 2
        a = A[:half]; b = A[half:]
        c = B[:half]; d = B[half:]

        # Conjugate d and c for the formula
        # conj(x): if dim=1 x; else (re, -im)
        # Note: Recursive conjugation

        # (a,b)(c,d) = (ac - d*b, da + bc*)

        # We need a robust conjugate function
        def conj(v, d):
            if d == 1: return v
            h = d // 2
            re = v[:h]
            im = v[h:]
            return np.concatenate((conj(re, h), -im), axis=0)

        d_star = conj(d, half)
        c_star = conj(c, half)

        ac = self._mul_recursive(a, c, half)
        db = self._mul_recursive(d_star, b, half)
        da = self._mul_recursive(d, a, half)
        bc = self._mul_recursive(b, c_star, half)

        res_real = ac - db
        res_imag = da + bc

        return np.concatenate((res_real, res_imag), axis=0)

# --- Simulation: 3D Field Evolution ---

def run_sedenion_field_sim():
    # Grid size (keep small for Python overhead)
    L = 6

    # Run 3D and 4D
    dimensions_to_test = [(L, L, L), (5, 5, 5, 5)]

    for dims in dimensions_to_test:
        dim_label = f"{len(dims)}D"
        print(f"Running {dim_label} Sedenion Field Sim {dims}...")

        phi = SedenionVectorized(dims)
        phi.set_random(scale=0.1)

        pi = SedenionVectorized(dims)
        pi.set_random(scale=0.01)

        dt = 0.05
        steps = 50

        associator_history = []
        energy_history = []

        for t in range(steps):
            # Laplacian
            lap = np.zeros_like(phi.data)
            for axis in range(1, len(dims)+1):
                lap += np.roll(phi.data, 1, axis=axis) + np.roll(phi.data, -1, axis=axis) - 2*phi.data

            phi_sq = SedenionVectorized(dims)
            phi_sq.data = phi.multiply(phi)

            force = lap + 0.1 * phi_sq.data
            pi.data += dt * force
            phi.data += dt * pi.data

            # Associator Norm (A(x,y,z))
            # Just take 3 shifted copies along first 3 axes
            phi_x = phi
            phi_y = SedenionVectorized(dims)
            phi_y.data = np.roll(phi.data, 1, axis=1)
            phi_z = SedenionVectorized(dims)
            phi_z.data = np.roll(phi.data, 1, axis=2)

            # (xy)z
            xy = SedenionVectorized(dims)
            xy.data = phi_x.multiply(phi_y)
            xyz = xy.multiply(phi_z)

            # x(yz)
            yz = SedenionVectorized(dims)
            yz.data = phi_y.multiply(phi_z)
            x_yz_val = phi_x.multiply(yz)

            # Associator Norm
            diff = xyz - x_yz_val
            assoc_norm = np.linalg.norm(diff, axis=0)
            mean_assoc = np.mean(assoc_norm)
            associator_history.append(mean_assoc)

            en = np.mean(phi.norm())
            energy_history.append(en)

        # Save Data
        df = pd.DataFrame({
            'step': range(steps),
            'mean_associator': associator_history,
            'mean_energy': energy_history
        })
        df.to_csv(f'data/csv/sedenion_field_metrics_{dim_label}.csv', index=False)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(associator_history, label='Mean Associator Norm')
        plt.plot(energy_history, label='Mean Field Energy')
        plt.title(f'{dim_label} Sedenion Field Evolution')
        plt.xlabel('Time Step')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'data/artifacts/sedenion_field_{dim_label}_plot.png')
        plt.close()

    print("Sedenion Field Sims Complete.")

if __name__ == "__main__":
    run_sedenion_field_sim()
