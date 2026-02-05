import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def generate_hyper_mera_vis():
    print("--- Generating Hyper-Resolution MERA Network (3160x2820) ---")

    W, H = 3160, 2820
    DPI = 300

    # 1. Construct MERA Graph
    depth = 7
    N_leaves = 2**depth
    G = nx.Graph()
    pos = {}

    node_id = 0
    layer_nodes = list(range(N_leaves))

    # Physical Layer (Circle Layout)
    radius = 10.0
    for i in range(N_leaves):
        theta = 2 * np.pi * i / N_leaves
        G.add_node(node_id, layer=0, type='phys')
        pos[node_id] = (radius * np.cos(theta), radius * np.sin(theta))
        node_id += 1

    # Renormalization Layers
    current_nodes = layer_nodes
    current_radius = radius

    for l in range(1, depth + 1):
        current_radius *= 0.85 # Shrink radius
        next_nodes = []

        # Disentanglers & Isometries mixed for visual simplicity
        # We just pair up nodes
        for i in range(0, len(current_nodes)-1, 2):
            parent = node_id
            theta = 2 * np.pi * (i + 0.5) * (2**(l-1)) / N_leaves

            G.add_node(parent, layer=l, type='renorm')
            pos[parent] = (current_radius * np.cos(theta), current_radius * np.sin(theta))

            G.add_edge(current_nodes[i], parent, type='tree')
            G.add_edge(current_nodes[i+1], parent, type='tree')

            next_nodes.append(parent)
            node_id += 1
        current_nodes = next_nodes

    # 2. Add Sedenion Wormholes (Non-Local Links)
    # Connect distant branches
    wormholes = []
    import random
    random.seed(42)
    leaves = [n for n,d in G.nodes(data=True) if d['layer']==0]

    for _ in range(40):
        u, v = random.sample(leaves, 2)
        # Only long distance
        dist = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        if dist > radius:
            G.add_edge(u, v, type='wormhole')
            wormholes.append((u,v))

    # 3. Plotting
    plt.rcParams.update({
        "figure.facecolor": "#0d0f14",
        "axes.facecolor": "#0d0f14",
        "text.color": "white",
        "font.family": "sans-serif",
        "font.weight": "light"
    })

    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    ax.set_facecolor("#0d0f14")

    # Draw Tree Edges (Neon Cyan)
    tree_edges = [e for e in G.edges() if G.edges[e].get('type') == 'tree']
    nx.draw_networkx_edges(G, pos, edgelist=tree_edges, edge_color='#00f7ff',
                           alpha=0.4, width=0.8, ax=ax)

    # Draw Wormholes (Neon Magenta with Glow)
    wh_edges = [e for e in G.edges() if G.edges[e].get('type') == 'wormhole']
    nx.draw_networkx_edges(G, pos, edgelist=wh_edges, edge_color='#ff00ff',
                           alpha=0.8, width=1.5, style='dashed', ax=ax)

    # Draw Nodes (Glowing Points)
    # Layers by color
    node_colors = []
    node_sizes = []
    for n, d in G.nodes(data=True):
        if d['layer'] == 0:
            node_colors.append('#ffffff') # Physical boundary
            node_sizes.append(10)
        else:
            node_colors.append('#00f7ff') # Bulk
            node_sizes.append(5)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, ax=ax)

    # Aesthetics
    ax.axis('off')
    ax.set_aspect('equal')

    # Title with Glow Effect
    title_text = ax.text(0, radius*1.2, "HOLOGRAPHIC TENSOR NETWORK\nSEDENION BULK GEOMETRY",
            ha='center', va='center', fontsize=28, color='white', fontweight='bold')
    title_text.set_path_effects([pe.withStroke(linewidth=3, foreground='#00f7ff')])

    # Annotations
    ax.text(-radius, -radius*1.1, "BOUNDARY: 1D CFT (L=128)", color='white', fontsize=16)
    ax.text(radius, -radius*1.1, "BULK: D = -1.5 (AdS + Wormholes)", color='magenta', fontsize=16, ha='right')

    output_path = "data/artifacts/images/hyper_mera_network.png"
    plt.savefig(output_path, facecolor='#0d0f14', dpi=DPI, bbox_inches='tight')
    print(f"Saved Hyper-MERA Visualization to {output_path}")

if __name__ == "__main__":
    generate_hyper_mera_vis()
