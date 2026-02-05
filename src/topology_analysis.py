import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from persim import plot_diagrams
from ripser import ripser


def analyze_topology(edge_file, label):
    print(f"--- Analyzing Topology for {label} ---")

    # 1. Load Graph
    if not os.path.exists(edge_file):
        print(f"File {edge_file} not found. Skipping.")
        return

    df = pd.read_csv(edge_file)
    G = nx.from_pandas_edgelist(df, 'source', 'target')

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Nodes: {n_nodes}, Edges: {n_edges}")

    # 2. Construct Metric Space
    # We use shortest path distance.
    # For large graphs, we might need to sample or use sparse distance matrix.
    # Chingon graph might be 27k nodes -> Distance matrix is 27k x 27k (float32) ~ 3GB.
    # This fits in 32GB RAM.

    if n_nodes > 5000:
        print("Graph large; using sparse distance / subsampling for TDA.")
        # Subsample largest component
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        subG = G.subgraph(comps[0])
        print(f"Largest Component: {len(subG)} nodes.")

        # Limit to first 2000 nodes for Ripser (O(N^2) or O(N^3))
        # Ripser is fast but 27k is pushing it for 'distance_matrix=False' (using sparse)
        # Actually ripser takes distance matrix.

        nodes = list(subG.nodes())[:2000] # Subsample
        subG = subG.subgraph(nodes)

        # Re-label for matrix
        G_idx = nx.convert_node_labels_to_integers(subG)

        # Distance Matrix
        D = dict(nx.all_pairs_shortest_path_length(G_idx))
        n_sub = len(G_idx)
        dist_mat = np.inf * np.ones((n_sub, n_sub))
        np.fill_diagonal(dist_mat, 0)

        for u, dists in D.items():
            for v, d in dists.items():
                dist_mat[u, v] = d

    else:
        # Full Distance Matrix
        D = dict(nx.all_pairs_shortest_path_length(G))
        nodes = list(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}
        n_sub = len(nodes)
        dist_mat = np.inf * np.ones((n_sub, n_sub))
        np.fill_diagonal(dist_mat, 0)

        for u, dists in D.items():
            u_idx = node_map[u]
            for v, d in dists.items():
                v_idx = node_map[v]
                dist_mat[u_idx, v_idx] = d

    # 3. Persistent Homology
    print("Running Ripser...")
    # Max dimension 3 (H0, H1, H2, H3)
    # Threshold: Graph diameter is usually small (log N).
    result = ripser(dist_mat, maxdim=3, distance_matrix=True)
    diagrams = result['dgms']

    # 4. Extract Betti Numbers
    # Betti number at scale epsilon is number of bars existing at that scale.
    # We want "Static" Betti numbers of the graph (Clique Complex).
    # This corresponds to epsilon ~ 1.0 in shortest path metric (edges have length 1).
    # If we use distance matrix, '1' means connected.
    # So we look at persistence at threshold 1.1 (just after edge formation).

    bettis = []
    threshold = 1.1
    for i, dgm in enumerate(diagrams):
        # Count points (birth <= 1.1 and death > 1.1)
        # Note: death = inf implies it lasts forever.
        count = 0
        for pt in dgm:
            birth, death = pt
            if birth <= threshold and death > threshold:
                count += 1
        bettis.append(count)
        print(f"Betti-{i} (epsilon={threshold}): {count}")

    # Save Diagrams
    plt.figure(figsize=(10, 5))
    plot_diagrams(diagrams, show=True)
    plt.title(f"Persistence Diagram: {label}")
    plt.savefig(f'data/artifacts/persistence_{label}.png')
    plt.close()

    # Save Betti sequence
    with open(f'data/csv/betti_sequence_{label}.csv', 'w') as f:
        f.write("dimension,betti_number,epsilon\n")
        for i, b in enumerate(bettis):
            f.write(f"{i},{b},{threshold}\n")

if __name__ == "__main__":
    # Analyze Sedenion, Pathion, Chingon
    # Note: Chingon is large, subsampling applied.

    # Check for files
    files = [
        ('data/csv/sedenion_zd_edges.csv', 'Sedenion_ZD'),
        ('data/csv/pathion_zd_edges.csv', 'Pathion_ZD')
    ]

    for fpath, lbl in files:
        analyze_topology(fpath, lbl)
