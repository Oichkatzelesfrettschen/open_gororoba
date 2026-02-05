import matplotlib.pyplot as plt
import numpy as np


class TensorNetworkClassical:
    """
    Implements Classical Approximations of Quantum Tensor Networks (TN).

    Standardized Terminology:
    - Tensor Nodes: Weights/States
    - Contraction: Interaction/Gate Application
    - Hadamard/CNOT: Unitary Transformations on Tensor Indices
    """
    def __init__(self, shape=(2, 2, 2, 2)):
        self.state = np.random.randn(*shape)
        # Normalize
        self.state /= np.linalg.norm(self.state)

    def apply_hadamard(self, axis=0):
        """
        Applies a Hadamard Gate (Superposition) to a specific tensor axis.
        H = 1/sqrt(2) * [[1, 1], [1, -1]]
        """
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # Tensor contraction: H_ij * T_j... -> T'_i...
        self.state = np.tensordot(H, self.state, axes=[1, axis])
        # Move the new axis back to position 'axis' to maintain structure
        self.state = np.moveaxis(self.state, 0, axis)

    def apply_cnot(self, control=0, target=1):
        """
        Applies CNOT Gate (Entanglement).
        Control qubit determines if Target is flipped.
        tensor contraction on two indices.
        """
        # CNOT tensor shape (2,2,2,2) -> (out_c, out_t, in_c, in_t)
        CNOT = np.zeros((2, 2, 2, 2))
        CNOT[0, 0, 0, 0] = 1 # |00> -> |00>
        CNOT[0, 1, 0, 1] = 1 # |01> -> |01>
        CNOT[1, 1, 1, 0] = 1 # |10> -> |11> (Flip)
        CNOT[1, 0, 1, 1] = 1 # |11> -> |10> (Flip)

        # Contract: CNOT_xyuv * T_...uv...
        self.state = np.tensordot(CNOT, self.state, axes=[[2, 3], [control, target]])
        # Reorder axes is complex here, simplifying for 2-qubit case or assuming specific structure
        # For general N-dim, we'd need rigorous index tracking (e.g. ncon)

    def measure_entropy(self):
        """
        Calculates Von Neumann Entropy of the tensor state (Entanglement measure).
        """
        # SVD on bipartition
        shape = self.state.shape
        flattened = self.state.reshape(shape[0]*shape[1], -1)
        _, S, _ = np.linalg.svd(flattened, full_matrices=False)
        S = S[S > 1e-15] # Filter zeros
        norm_S = S / np.sqrt(np.sum(S**2)) # Normalize singular values
        probs = norm_S**2
        return -np.sum(probs * np.log(probs))

def simulate_tensor_network_evolution():
    print("Simulating Tensor Network Evolution (Quantum-Classical Bridge)...")
    tn = TensorNetworkClassical(shape=(2, 2, 2, 2))

    entropies = []
    for _ in range(50):
        tn.apply_hadamard(axis=np.random.randint(0, 4))
        tn.apply_cnot(control=0, target=1) # Fixed pair for simplicity
        entropies.append(tn.measure_entropy())

    plt.figure(figsize=(8, 5))
    plt.plot(entropies, color='purple')
    plt.title("Entanglement Entropy Evolution in Tensor Network")
    plt.xlabel("Gate Operations")
    plt.ylabel("Von Neumann Entropy")
    plt.savefig("curated/02_simulations_pde_quantum/tensor_network_entropy.png")
    print("Saved Entropy visualization.")

if __name__ == "__main__":
    simulate_tensor_network_evolution()
