import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def analyze_box_kites():
    print("--- Analyzing 42 Assessors for Box-Kite Structure ---")

    # 1. Load Data
    try:
        df = pd.read_csv("data/csv/sedenion_nilpotent_candidates.csv")
    except FileNotFoundError:
        print("Data file not found.")
        return

    # 2. Parse Vectors
    # We need the actual 16D vectors to cluster them.
    unique_vectors = {} # map tuple -> vector array

    for _, row in df.iterrows():
        for col in ['Element_A_Indices', 'Element_B_Indices']:
            s = str(row[col]).replace('[', '').replace(']', '').replace(',', ' ')
            indices = tuple(sorted([int(x) for x in s.split() if x.strip()]))

            if indices not in unique_vectors:
                v = np.zeros(16)
                v[list(indices)] = 1
                unique_vectors[indices] = v

    roots = list(unique_vectors.keys())
    vectors = np.array(list(unique_vectors.values()))

    print(f"Loaded {len(roots)} unique vectors.")

    if len(roots) < 7:
        print("Not enough roots to cluster into 7 Box-Kites.")
        return

    # 3. K-Means Clustering (Target: 7 Clusters)
    # The theory says there are 7 Box-Kites.
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    # 4. Visualization (PCA)
    pca = PCA(n_components=3)
    vecs_pca = pca.fit_transform(vectors)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color map for 7 clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, 7))

    for i in range(7):
        cluster_mask = (labels == i)
        ax.scatter(vecs_pca[cluster_mask, 0],
                   vecs_pca[cluster_mask, 1],
                   vecs_pca[cluster_mask, 2],
                   c=[colors[i]], s=100, label=f'Box-Kite {i+1}')

    ax.set_title(f"The 7 Box-Kites (K-Means Clustering of {len(roots)} Assessors)")
    ax.legend()
    plt.savefig("data/artifacts/images/box_kites_clustering.png")
    print("Saved Box-Kite Clustering Plot.")

    # 5. Save Clustered Data
    # Map index back to original vector representation
    cluster_data = []
    for i, root in enumerate(roots):
        cluster_data.append({
            'Root_Indices': root,
            'Cluster_ID': labels[i]
        })

    df_out = pd.DataFrame(cluster_data)
    df_out.sort_values('Cluster_ID').to_csv("data/csv/sedenion_box_kites_clustered.csv", index=False)
    print("Saved clustered data to data/csv/sedenion_box_kites_clustered.csv")

    # 6. Verify Cluster Sizes
    counts = df_out['Cluster_ID'].value_counts()
    print("\nCluster Sizes (Ideally 6 per cluster):")
    print(counts)

if __name__ == "__main__":
    analyze_box_kites()
