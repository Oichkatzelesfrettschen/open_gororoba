import matplotlib.pyplot as plt
import numpy as np


class RenormalizationGroup:
    """
    Simulates the Renormalization Group (RG) Flow of a field.
    Previous 'Fractal AI' terminology is standardized to:
    - Weight Space -> Scalar Field (Phi)
    - Fractal Dimension -> Anomalous Dimension (gamma)
    - Laplacian -> Kinetic Term (Spatial derivatives)
    - Nonlinearity -> Interaction Potential (lambda * Phi^n)
    """
    def __init__(self, size=50):
        self.field = np.random.randn(size, size)
        self.size = size

    def anomalous_dimension(self, field):
        """
        Calculates the scaling dimension based on field gradients.
        gamma = d(log(grad_Phi)) / d(log(Phi))
        """
        grad = np.gradient(field)
        mag_grad = np.sqrt(grad[0]**2 + grad[1]**2)

        avg_grad = np.mean(mag_grad) + 1e-9
        avg_field = np.mean(np.abs(field)) + 1e-9

        return 1 + np.log(avg_grad) / np.log(avg_field)

    def laplacian(self, Z):
        return np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) + \
               np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z

    def evolve(self, steps=100, eta=0.01, coupling=0.01):
        history = []
        field = self.field.copy()

        for _ in range(steps):
            gamma = self.anomalous_dimension(field)

            # dPhi/dt = - Laplacian(Phi) - lambda * Phi^(gamma)
            # This mimics the Callan-Symanzik equation flow

            diffusion = self.laplacian(field)
            interaction = coupling * np.sign(field) * np.abs(field)**gamma

            dPhi = diffusion - interaction

            # Update (Gradient Flow)
            field += eta * dPhi

            # Regularize (Renormalize) to keep finite
            if np.max(np.abs(field)) > 10:
                field /= np.max(np.abs(field))

            history.append(field.copy())

        return history

def simulate_rg_flow():
    print("Simulating Renormalization Group Flow (Scale-Dependent Evolution)...")
    rg = RenormalizationGroup()
    history = rg.evolve()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(history[0], cmap='inferno')
    ax[0].set_title("UV Scale (Initial State)")

    ax[1].imshow(history[50], cmap='inferno')
    ax[1].set_title("Intermediate Scale")

    ax[2].imshow(history[-1], cmap='inferno')
    ax[2].set_title("IR Scale (Fixed Point)")

    plt.savefig("curated/02_simulations_pde_quantum/rg_flow_evolution.png")
    print("Saved RG Flow visualization.")

if __name__ == "__main__":
    simulate_rg_flow()
