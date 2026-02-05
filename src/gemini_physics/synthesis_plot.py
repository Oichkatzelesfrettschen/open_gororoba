import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_synthesis_report():
    """
    Generates a 4-pane cross-domain synthesis plot.
    Links: Algebra | Lattice | Cosmology | Transport
    """
    print("Generating Unified Gemini Synthesis Report...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Algebra (Fano Plane)
    img_fano = plt.imread("curated/01_theory_frameworks/fano_plane_structure.png")
    axs[0, 0].imshow(img_fano)
    axs[0, 0].set_title("I. Algebraic Foundation (Octonions)", fontsize=16)
    axs[0, 0].axis('off')

    # 2. Lattice (E8)
    img_e8 = plt.imread("curated/01_theory_frameworks/e8_quasicrystal_projection.png")
    axs[0, 1].imshow(img_e8)
    axs[0, 1].set_title("II. Physical Substrate (E8 Quasicrystal)", fontsize=16)
    axs[0, 1].axis('off')

    # 3. Cosmology (FLRW)
    cosmo_path = "curated/02_simulations_pde_quantum/quantum_cosmology_statefinders.csv"
    if os.path.exists(cosmo_path):
        df_c = pd.read_csv(cosmo_path)
        axs[1, 0].plot(df_c['time_Gyr'], df_c['scale_factor_a'], color='red', lw=2)
        axs[1, 0].set_title("III. Macroscopic Evolution (Quantum FLRW)", fontsize=16)
        axs[1, 0].set_xlabel("Time (Gyr)")
        axs[1, 0].set_ylabel("Scale Factor a(t)")
        axs[1, 0].grid(True, alpha=0.3)

    # 4. Transport (Strange Metals)
    holo_path = "curated/02_simulations_pde_quantum/strange_metal_holographic_transport.csv"
    if os.path.exists(holo_path):
        df_h = pd.read_csv(holo_path)
        axs[1, 1].plot(df_h['temperature_K'], df_h['resistivity_ohm_m'], color='green', lw=2)
        axs[1, 1].set_title("IV. Material Transport (Holographic Metals)", fontsize=16)
        axs[1, 1].set_xlabel("Temperature (K)")
        axs[1, 1].set_ylabel("Resistivity (Linear-T)")
        axs[1, 1].grid(True, alpha=0.3)

    plt.suptitle("THE GEMINI PROTOCOL: Unified Multi-Scale Physics Synthesis", fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    final_path = "curated/05_summaries_indexes/gemini_unified_synthesis.png"
    plt.savefig(final_path)
    print(f"Unified Synthesis Report saved to {final_path}")

if __name__ == "__main__":
    generate_synthesis_report()
