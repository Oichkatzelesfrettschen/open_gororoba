import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_nilpotents():
    print("--- Advanced Visualization: Nilpotent Candidates ---")

    # Load Data
    try:
        df = pd.read_csv("data/csv/sedenion_nilpotent_candidates.csv")
    except FileNotFoundError:
        print("Data not found. Skipping.")
        return

    # Parse indices from string "(1, 2)" -> [1, 2]
    # The csv saved tuples as strings likely. Let's inspect or just handle it.
    # Actually the previous script saved them as columns "Element_A_Indices", "Element_B_Indices"
    # containing tuples like "(1, 5)".

    # Let's just create a synthetic high-dim representation for visualization
    # We have 48 unique components. Let's one-hot encode them into 16D space
    # and project.

    # Re-extract unique vectors
    unique_vecs = []

    # We need to parse the CSV properly
    # For now, let's just generate the vectors from the indices we know exist
    # (Since we can't easily parse the specific string format without ast.literal_eval)
    import re

    vectors = []
    labels = []

    for _, row in df.iterrows():
        # Clean up numpy string representation: "[15  4]" -> "15 4" -> [15, 4]
        # Handle cases with or without commas, brackets, etc.
        s_a = row['Element_A_Indices'].replace('[', '').replace(']', '').replace(',', ' ')
        s_b = row['Element_B_Indices'].replace('[', '').replace(']', '').replace(',', ' ')

        # Split by whitespace and convert to int
        idx_a = [int(x) for x in s_a.split() if x.strip()]
        idx_b = [int(x) for x in s_b.split() if x.strip()]

        # Vector A (16D)
        va = np.zeros(16)
        va[idx_a] = 1
        vectors.append(va)
        labels.append("A")

        # Vector B (16D)
        vb = np.zeros(16)
        vb[idx_b] = 1
        vectors.append(vb)
        labels.append("B")

    X = np.array(vectors)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Plot PCA
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c='cyan', alpha=0.6, s=50)
    ax.set_title("PCA Projection of Sedenion Zero-Divisors (48 Components)")
    plt.savefig("data/artifacts/images/sedenion_zd_pca.png")
    print("Saved PCA plot.")

    # Graph Visualization of Pairs
    G = nx.Graph()
    for i in range(len(df)):
        G.add_edge(f"A_{i}", f"B_{i}")

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, node_size=20, node_color="magenta", alpha=0.5)
    plt.title("Zero-Divisor Pairing Graph")
    plt.savefig("data/artifacts/images/sedenion_zd_network.png")
    print("Saved Network plot.")

if __name__ == "__main__":
    visualize_nilpotents()
