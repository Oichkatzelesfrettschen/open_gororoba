import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def correlate_real_physics_gwtc():
    print("--- Correlating Real LIGO O3 Data ---")

    # 1. Load LIGO Data (Real Download)
    try:
        gwtc = pd.read_csv("data/external/GWTC-3_confident.csv")
        # Filter for valid masses
        masses = gwtc['mass_1_source'].dropna()
        distances = gwtc['luminosity_distance'].dropna()
        redshifts = gwtc['redshift'].dropna()
        print(f"Loaded {len(masses)} Confident BH Events from LIGO O3.")
    except Exception as e:
        print(f"LIGO Data failed to load: {e}")
        return

    # 2. Load SDSS Quasars (Already fetched)
    try:
        sdss = pd.read_csv("data/external/SDSS_Quasars_Astroquery.csv")
        z_qsos = sdss['z']
    except:
        print("SDSS Data missing.")
        z_qsos = []

    # 3. Load TESS Data (Already fetched)
    try:
        tess = pd.read_csv("data/external/TESS_Metadata.csv")
        exptime = tess['t_exptime']
    except:
        exptime = []

    # 4. Correlation Plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Quasar Redshifts (The Cosmic Boundary)
    if len(z_qsos) > 0:
        ax[0].hist(z_qsos, bins=30, color='crimson', alpha=0.8, edgecolor='black', density=True)
        ax[0].set_title(f"SDSS Quasars (z > 0.1)")
        ax[0].set_xlabel("Redshift (z)")
        ax[0].set_ylabel("Density")

    # Plot 2: LIGO BH Masses (The Bulk)
    if len(masses) > 0:
        ax[1].hist(masses, bins=20, color='black', alpha=0.8, edgecolor='white', density=True)
        ax[1].set_title(f"LIGO GWTC-3: BH Masses (N={len(masses)})")
        ax[1].set_xlabel(r"Primary Mass ($M_{\odot}$)")
        # Overlay a Sedenion 'Double Peak' prediction curve (Toy Model)
        # Prediction: Mass distribution peaks at 30 and 60 due to 'Box-Kite' resonance
        # Just a visual guide for the theory
        x = np.linspace(0, 100, 100)
        y = 0.03 * np.exp(-(x-30)**2/100) + 0.02 * np.exp(-(x-60)**2/100)
        ax[1].plot(x, y, color='cyan', linestyle='--', label='Sedenion Resonance Prediction')
        ax[1].legend()

    # Plot 3: LIGO Redshift vs Distance (The Hubble Flow)
    if len(redshifts) > 0:
        ax[2].scatter(redshifts, distances, color='blue', alpha=0.6, s=30)
        ax[2].set_title("LIGO Source Redshift vs Distance")
        ax[2].set_xlabel("Redshift (z)")
        ax[2].set_ylabel("Luminosity Distance (Mpc)")
        ax[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.suptitle("Multi-Messenger Correlation: Real O3 Data Analysis")
    plt.savefig("data/artifacts/images/real_physics_correlation_gwtc3.png")
    print("Saved Correlation Plot.")

if __name__ == "__main__":
    correlate_real_physics_gwtc()
