import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def correlate_real_physics():
    print("--- Correlating Real Observatory Data ---")

    # 1. Load SDSS Quasars (Real Data)
    try:
        sdss = pd.read_csv("data/external/SDSS_Quasars_Astroquery.csv")
        z_real = sdss['z']
    except FileNotFoundError:
        print("SDSS Data missing. Skipping.")
        return

    # 2. Load TESS Data (Real Metadata)
    try:
        tess = pd.read_csv("data/external/TESS_Metadata.csv")
        # Extract exposure time or some physical param
        exptime = tess['t_exptime']
    except:
        exptime = []

    # 3. Load Synthetic GWTC-3 (Since we still rely on the synthetic for BHs)
    try:
        gw = pd.read_csv("data/external/GWTC-3_synthetic.csv")
        masses = gw['mass_1_source']
    except:
        masses = []

    # 4. Correlation Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Real Quasar Redshifts
    ax[0].hist(z_real, bins=30, color='crimson', alpha=0.8, edgecolor='black')
    ax[0].set_title(f"SDSS Quasars (N={len(z_real)})")
    ax[0].set_xlabel("Redshift (z)")
    ax[0].set_ylabel("Count")

    # Plot 2: TESS Exposure Times (Observational Cadence)
    if len(exptime) > 0:
        ax[1].hist(exptime, bins=30, color='orange', alpha=0.8, edgecolor='black')
        ax[1].set_title(f"TESS Observations (N={len(exptime)})")
        ax[1].set_xlabel("Exposure Time (s)")
        ax[1].set_yscale('log')

    # Plot 3: Synthetic Black Hole Masses
    if len(masses) > 0:
        ax[2].hist(masses, bins=20, color='black', alpha=0.8, edgecolor='white')
        ax[2].set_title("GWTC-3 (Synthetic) BH Masses")
        ax[2].set_xlabel(r"Mass ($M_{\odot}$)")

    plt.suptitle("Multi-Messenger Correlation: Quasars, Exoplanets, and Black Holes")
    plt.savefig("data/artifacts/images/real_physics_correlation.png")
    print("Saved Correlation Plot.")

if __name__ == "__main__":
    correlate_real_physics()
