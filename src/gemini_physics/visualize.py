import matplotlib.pyplot as plt


def plot_fano_plane():
    """
    Visualizes the Fano Plane, the mnemonic for Octonion multiplication.
    This fulfills Step 5 of the Gemini Roadmap.
    """
    print("Generating Fano Plane Visualization...")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Vertices of the equilateral triangle
    v = [
        (0, 1),          # e1 (top)
        (-0.866, -0.5),  # e2 (bottom left)
        (0.866, -0.5)    # e3 (bottom right)
    ]

    # Midpoints
    m = [
        ((v[0][0] + v[1][0]) / 2, (v[0][1] + v[1][1]) / 2),  # e4
        ((v[1][0] + v[2][0]) / 2, (v[1][1] + v[2][1]) / 2),  # e5
        ((v[2][0] + v[0][0]) / 2, (v[2][1] + v[0][1]) / 2),  # e6
    ]

    # Centroid
    c = (0, 0) # e7

    # Label mapping
    labels = {
        v[0]: "e1",
        v[1]: "e2",
        v[2]: "e3",
        m[0]: "e4",
        m[1]: "e5",
        m[2]: "e6",
        c: "e7",
    }

    # Draw circle
    circle = plt.Circle((0, -0.15), 0.45, color='blue', fill=False, linestyle='--')
    ax.add_patch(circle)

    # Draw triangle lines
    lines = [
        (v[0], v[1]),
        (v[1], v[2]),
        (v[2], v[0]),  # Outer
        (v[0], m[1]),
        (v[1], m[2]),
        (v[2], m[0]),  # Medians
    ]

    for segment in lines:
        ax.plot(
            [segment[0][0], segment[1][0]],
            [segment[0][1], segment[1][1]],
            color="black",
            alpha=0.6,
        )

    # Plot points
    all_pts = v + m + [c]
    for pt in all_pts:
        ax.scatter(pt[0], pt[1], s=500, color="white", edgecolors="black", zorder=5)
        ax.text(pt[0], pt[1], labels[pt], ha="center", va="center", fontweight="bold")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_title("The Fano Plane: 8D Octonion Multiplicative Structure", fontsize=14)

    plt.savefig("curated/01_theory_frameworks/fano_plane_structure.png")
    print("Fano Plane visualization saved to curated/01_theory_frameworks/fano_plane_structure.png")

if __name__ == "__main__":
    plot_fano_plane()
