import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la


# Re-import Sedenion Vectorized logic locally to avoid circular imports if path issues
class SedenionVectorized:
    def __init__(self, shape):
        self.shape = shape
        self.data = np.zeros((16,) + shape, dtype=np.float32)

    def set_random(self, scale=1.0):
        self.data = np.random.normal(0, scale, size=(16,) + self.shape).astype(np.float32)

    def multiply(self, other):
        # Simplified recursive mul for tracking
        return self._mul_recursive(self.data, other.data, 16)

    def _mul_recursive(self, A, B, dim):
        if dim == 1: return A * B
        half = dim // 2
        a, b = A[:half], A[half:]
        c, d = B[:half], B[half:]

        def conj(v, d):
            if d == 1: return v
            h = d // 2
            re, im = v[:h], v[h:]
            return np.concatenate((conj(re, h), -im), axis=0)

        ac = self._mul_recursive(a, c, half)
        db = self._mul_recursive(conj(d, half), b, half)
        da = self._mul_recursive(d, a, half)
        bc = self._mul_recursive(b, conj(c, half), half)
        return np.concatenate((ac - db, da + bc), axis=0)

def run_spectral_flow(steps=30):
    print("Running Spectral Flow Simulation...")
    L = 6 # Small grid for SVD speed
    dims = (L, L, L)

    phi = SedenionVectorized(dims)
    phi.set_random(0.1)

    spectra = []

    for t in range(steps):
        # 1. Evolve (Simple Wave Eq)
        phi_sq_data = phi.multiply(phi)
        phi.data += 0.01 * phi_sq_data # Drift

        # 2. Compute Spectrum
        # Unfold tensor: (16, L, L, L) -> (16, L*L*L)
        # We want the singular values of the field configuration
        # which represents the "energy modes" of the Sedenion matter.
        flat = phi.data.reshape(16, -1)

        # SVD
        s = la.svd(flat, compute_uv=False)
        spectra.append(s)

    # Save
    spectra = np.array(spectra)
    df = pd.DataFrame(spectra, columns=[f'mode_{i}' for i in range(16)])
    df['step'] = range(steps)
    df.to_csv('data/csv/spectral_flow.csv', index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    for i in range(16):
        plt.plot(df['step'], df[f'mode_{i}'], label=f'Mode {i}' if i%4==0 else "")

    plt.xlabel('Time Step')
    plt.ylabel('Singular Value (Energy Mode)')
    plt.title('Spectral Flow of Sedenion Field')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/artifacts/spectral_flow_plot.png')
    print("Spectral Flow Complete.")

if __name__ == "__main__":
    run_spectral_flow()
